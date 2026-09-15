"""
NSGA-II Multi-Objective Architecture Optimization for DeepGuard AI.

Searches for Pareto-optimal trade-offs between validation accuracy and
real ONNX Runtime CPU inference latency across DualStreamNet hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import DeepfakeDataset, SUPPORTED_EXTENSIONS
from engine.trainer import Trainer
from export.export_onnx import export_to_onnx
from models.dual_stream_net import DualStreamNet
from optimization.search_space import (
    decode_genome,
    get_search_space_bounds,
)
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize


def verify_real_dataset_presence(config: dict, base_dir: Path | None = None) -> tuple[bool, str]:
    """Inspects the paths configured in config['data'] to ensure real data exists.

    Returns (is_present, status_message).
    """
    data_cfg = config.get("data", {})
    train_dir_str = data_cfg.get("train_dir", "data/processed/train")
    val_dir_str = data_cfg.get("val_dir", "data/processed/val")

    if base_dir is None:
        base_dir = Path.cwd()

    train_dir = Path(train_dir_str)
    if not train_dir.is_absolute():
        train_dir = base_dir / train_dir

    val_dir = Path(val_dir_str)
    if not val_dir.is_absolute():
        val_dir = base_dir / val_dir

    def _count_samples(d: Path) -> int:
        if not d.is_dir():
            return 0
        count = 0
        for cls_name in ("real", "fake"):
            c_dir = d / cls_name
            if c_dir.is_dir():
                count += sum(1 for f in c_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)
        return count

    train_count = _count_samples(train_dir)
    val_count = _count_samples(val_dir)

    if train_count == 0 or val_count == 0:
        msg = (
            f"Configured dataset paths do not contain real training/validation images:\n"
            f"  - Train path '{train_dir}' has {train_count} valid image samples\n"
            f"  - Val path '{val_dir}' has {val_count} valid image samples\n"
            f"Expected '{train_dir}/real' and '{train_dir}/fake' (and val counterpart) "
            f"containing image files."
        )
        return False, msg

    return True, f"Found {train_count} training samples and {val_count} validation samples."


class DualStreamNetProblem(ElementwiseProblem):
    """Pymoo ElementwiseProblem for DualStreamNet multi-objective architecture search.

    Objectives (both minimized):
      f1 = -validation_accuracy (maximize accuracy)
      f2 = median_cpu_latency_ms (minimize latency)
    """

    def __init__(
        self,
        config: dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        epochs: int = 3,
        temp_dir: Path | None = None,
        num_warmup: int = 5,
        num_latency_runs: int = 30,
        **kwargs: Any,
    ) -> None:
        xl, xu = get_search_space_bounds()
        super().__init__(n_var=4, n_obj=2, n_ieq_constr=0, xl=xl, xu=xu, **kwargs)

        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.num_warmup = num_warmup
        self.num_latency_runs = num_latency_runs
        self.eval_count = 0
        self.history: list[dict[str, Any]] = []

        if temp_dir is None:
            self.temp_root = Path(tempfile.mkdtemp(prefix="deepguard_nsga2_"))
            self._own_temp = True
        else:
            self.temp_root = temp_dir
            self.temp_root.mkdir(parents=True, exist_ok=True)
            self._own_temp = False

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        self.eval_count += 1
        candidate_model_cfg = decode_genome(x)

        print(
            f"\n[Candidate {self.eval_count}] Backbone: {candidate_model_cfg['spatial_backbone']}, "
            f"FreqDim: {candidate_model_cfg['frequency_feature_dim']}, "
            f"FusionDim: {candidate_model_cfg['fusion_hidden_dim']}, "
            f"Dropout: {candidate_model_cfg['dropout']:.3f}"
        )

        # 1. Build candidate model via from_config
        full_cfg = {"model": candidate_model_cfg}
        model = DualStreamNet.from_config(full_cfg)

        # 2. Train with existing unmodified Trainer writing to isolated checkpoint dir
        candidate_ckpt_dir = self.temp_root / f"candidate_{self.eval_count}_ckpt"
        candidate_ckpt_dir.mkdir(parents=True, exist_ok=True)

        training_cfg = self.config.get("training", {})
        learning_rate = float(training_cfg.get("learning_rate", 1e-4))
        weight_decay = float(training_cfg.get("weight_decay", 1e-5))

        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            checkpoint_dir=str(candidate_ckpt_dir),
            early_stopping_patience=training_cfg.get("early_stopping_patience", 3),
        )
        trainer.fit(epochs=self.epochs)
        val_accuracy = float(trainer.best_val_accuracy)

        # 3. Export to ONNX using existing unmodified export_to_onnx
        candidate_onnx_path = self.temp_root / f"candidate_{self.eval_count}.onnx"
        image_size = int(self.config.get("data", {}).get("image_size", 224))
        export_model = model.to("cpu").eval()

        dummy_rgb, dummy_freq = export_to_onnx(export_model, candidate_onnx_path, image_size)

        # 4. Measure median CPU inference latency via onnxruntime
        ort_session = ort.InferenceSession(str(candidate_onnx_path), providers=["CPUExecutionProvider"])
        rgb_np = dummy_rgb.numpy()
        freq_np = dummy_freq.numpy()

        for _ in range(self.num_warmup):
            ort_session.run(None, {"rgb": rgb_np, "freq": freq_np})

        latencies: list[float] = []
        for _ in range(self.num_latency_runs):
            t0 = time.perf_counter()
            ort_session.run(None, {"rgb": rgb_np, "freq": freq_np})
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        median_latency_ms = float(np.median(latencies))

        # Cleanup candidate files
        if candidate_onnx_path.exists():
            candidate_onnx_path.unlink()
        if candidate_ckpt_dir.exists():
            shutil.rmtree(candidate_ckpt_dir, ignore_errors=True)

        # Objectives (minimize both): f1 = -accuracy, f2 = latency_ms
        f1 = -val_accuracy
        f2 = median_latency_ms
        out["F"] = [f1, f2]

        record = {
            "candidate_id": self.eval_count,
            "config": candidate_model_cfg,
            "accuracy": float(round(val_accuracy, 4)),
            "latency_ms": float(round(median_latency_ms, 2)),
        }
        self.history.append(record)

        print(f" -> Result: Accuracy={val_accuracy:.4f} | Median Latency={median_latency_ms:.2f} ms")

    def close(self) -> None:
        if self._own_temp and self.temp_root.exists():
            shutil.rmtree(self.temp_root, ignore_errors=True)


def plot_pareto_front(
    all_evaluations: list[dict[str, Any]],
    pareto_front: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generates and saves the Pareto trade-off scatter plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_latencies = [e["latency_ms"] for e in all_evaluations]
    all_accuracies = [e["accuracy"] * 100.0 for e in all_evaluations]

    pf_latencies = [e["latency_ms"] for e in pareto_front]
    pf_accuracies = [e["accuracy"] * 100.0 for e in pareto_front]

    # Sort Pareto front by latency for connecting line
    sorted_pf = sorted(zip(pf_latencies, pf_accuracies), key=lambda pt: pt[0])
    pf_lat_sorted = [pt[0] for pt in sorted_pf]
    pf_acc_sorted = [pt[1] for pt in sorted_pf]

    plt.figure(figsize=(9, 6), dpi=150)
    plt.scatter(
        all_latencies,
        all_accuracies,
        color="#3b82f6",
        alpha=0.6,
        s=60,
        label="Evaluated Architectures",
        edgecolors="none",
    )
    plt.plot(
        pf_lat_sorted,
        pf_acc_sorted,
        color="#ef4444",
        linestyle="--",
        linewidth=1.8,
        alpha=0.8,
        label="Pareto Frontier",
    )
    plt.scatter(
        pf_latencies,
        pf_accuracies,
        color="#ef4444",
        s=120,
        marker="*",
        edgecolors="#7f1d1d",
        linewidth=1.2,
        label="Pareto-Optimal Solutions",
        zorder=5,
    )

    for pt in pareto_front:
        backbone_short = pt["config"]["spatial_backbone"].replace("efficientnet_", "eff_")
        plt.annotate(
            f"{backbone_short}\n({pt['latency_ms']:.1f}ms, {pt['accuracy']*100:.1f}%)",
            (pt["latency_ms"], pt["accuracy"] * 100.0),
            textcoords="offset points",
            xytext=(8, -8),
            fontsize=8,
            color="#1e293b",
        )

    plt.title("DeepGuard AI: Accuracy vs. CPU Latency Pareto Front (NSGA-II)", fontsize=13, fontweight="bold")
    plt.xlabel("ONNX Runtime CPU Median Latency (ms) -> lower is better", fontsize=11)
    plt.ylabel("Validation Accuracy (%) -> higher is better", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()


def run_nsga2_optimization(
    config: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    pop_size: int = 8,
    n_gen: int = 5,
    epochs: int = 3,
    output_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Runs NSGA-II search over DualStreamNet hyperparameter space."""
    if output_dir is None:
        output_dir = Path("ml_core/optimization/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = DualStreamNetProblem(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
    )

    algorithm = NSGA2(pop_size=pop_size)

    print(f"\nStarting NSGA-II Multi-Objective Optimization...")
    print(f"Population size: {pop_size} | Generations: {n_gen} | Epochs/candidate: {epochs} | Seed: {seed}")

    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        seed=seed,
        verbose=True,
    )

    all_history = problem.history

    # Filter non-dominated Pareto front from history
    pareto_front: list[dict[str, Any]] = []
    for cand in all_history:
        acc = cand["accuracy"]
        lat = cand["latency_ms"]
        # cand is dominated if another candidate has >= acc and <= lat, with at least one strictly better
        is_dominated = False
        for other in all_history:
            if other is cand:
                continue
            o_acc = other["accuracy"]
            o_lat = other["latency_ms"]
            if (o_acc >= acc and o_lat <= lat) and (o_acc > acc or o_lat < lat):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(cand)

    # Sort Pareto front by latency
    pareto_front = sorted(pareto_front, key=lambda c: c["latency_ms"])

    # Save JSON
    json_path = output_dir / "pareto_front.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pop_size": pop_size,
                "n_gen": n_gen,
                "epochs_per_candidate": epochs,
                "pareto_front": pareto_front,
                "all_evaluations": all_history,
            },
            f,
            indent=2,
        )

    # Save PNG
    png_path = output_dir / "pareto_front.png"
    plot_pareto_front(all_history, pareto_front, png_path)

    problem.close()

    print("\n" + "=" * 70)
    print("NSGA-II OPTIMIZATION COMPLETED")
    print(f"Total evaluated candidates: {len(all_history)}")
    print(f"Pareto-optimal solutions found: {len(pareto_front)}")
    print(f"Results saved to: {json_path}")
    print(f"Plot saved to:    {png_path}")
    print("=" * 70)

    return {
        "pareto_front": pareto_front,
        "all_evaluations": all_history,
        "json_path": str(json_path),
        "png_path": str(png_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NSGA-II hyperparameter search for DeepGuard AI.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--pop-size", type=int, default=8, help="Population size (default: 8)")
    parser.add_argument("--n-gen", type=int, default=5, help="Number of generations (default: 5)")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per candidate (default: 3)")
    parser.add_argument("--output-dir", type=str, default="optimization/results", help="Results output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at '{config_path}'")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Real data check
    is_present, status_msg = verify_real_dataset_presence(config)
    if not is_present:
        print("\n" + "=" * 76)
        print("REAL DATASET CHECK FAILED:")
        print(status_msg)
        print("\nSTOPPING: Optimization cannot produce meaningful validation accuracy without real data.")
        print("No synthetic data will be fabricated for the production search.")
        print("=" * 76 + "\n")
        return

    # Real data exists - construct loaders
    data_cfg = config["data"]
    train_dataset = DeepfakeDataset(data_cfg["train_dir"], data_cfg["image_size"], train=True)
    val_dataset = DeepfakeDataset(data_cfg["val_dir"], data_cfg["image_size"], train=False)

    device = torch.device(config.get("training", {}).get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=0,
    )

    run_nsga2_optimization(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        epochs=args.epochs,
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
