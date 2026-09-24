\# DeepGuard AI — Trained Model Checkpoint



\## 1. Checkpoint Summary



This checkpoint records the first full real-data training run of DeepGuard AI after dataset preparation, identity-safe splitting, GPU training, ONNX export, backend integration, and frontend E2E verification.



Training was performed in Google Colab using an NVIDIA Tesla T4 GPU.



The model architecture was not redesigned or replaced.



\---



\## 2. Codebase



Repository:



`https://github.com/SufiShaikh05/deepguard-ai`



Codebase commit used for the Colab training environment:



`4f76ab2 Add NSGA-II optimization framework`



\---



\## 3. Model Architecture



Model:



`DualStreamNet`



Components:



\- Spatial stream: Xception

\- Frequency stream: RGB-channel DCT frequency representation

\- Fusion head: classification MLP

\- Number of classes: 2



Class mapping:



\- `0 = real`

\- `1 = fake`



Training configuration:



\- spatial\_backbone: `xception`

\- spatial\_pretrained: `true`

\- freeze\_spatial\_backbone: `false`

\- frequency\_in\_channels: `3`

\- frequency\_feature\_dim: `256`

\- fusion\_hidden\_dim: `256`

\- dropout: `0.3`

\- num\_classes: `2`

\- image\_size: `224`



All model parameters were trainable during the final GPU training run.



\---



\## 4. Dataset



Dataset:



`Sowaiba01/Deepfake`



Source format:



\- `fake/`

\- `real/`

\- `metadata.csv`



Raw files downloaded:



\- Fake images: `5,426`

\- Real images: `5,426`



Metadata rows:



`5,352`



Identity analysis:



\- Unique identities: `10,478`

\- Connected identity components: `5,126`

\- Largest component: `4 identities`

\- Components with 2 identities: `4,917`

\- Components with 3 identities: `192`

\- Components with 4 identities: `17`



Identity-safe splitting was performed by keeping every connected identity component entirely within one split.



\---



\## 5. Dataset Exclusions



The raw directory contained:



\- 74 fake files not represented in `metadata.csv`

\- 74 real files not represented in `metadata.csv`



These files were excluded from the model dataset.



No metadata was invented or reconstructed for those files.



Therefore the final dataset contained exactly:



`5,352 fake/real metadata-backed pairs`



\---



\## 6. Identity-Safe Split



Deterministic split seed:



`20260920`



Final split:



| Split | Pairs | Fake | Real | Identity Components |

|---|---:|---:|---:|---:|

| Train | 3,746 | 3,746 | 3,746 | 3,588 |

| Validation | 803 | 803 | 803 | 770 |

| Test | 803 | 803 | 803 | 768 |

| Total | 5,352 | 5,352 | 5,352 | 5,126 |



Identity leakage checks:



\- Train ↔ Validation: NONE

\- Train ↔ Test: NONE

\- Validation ↔ Test: NONE



The split manifest was independently validated against the actual files before materialization.



\---



\## 7. Training Environment



Platform:



Google Colab



Python:



`3.13.15`



PyTorch:



`2.11.0+cu128`



CUDA:



`12.8`



GPU:



`NVIDIA Tesla T4`



GPU memory:



`14.56 GB`



Training batch size:



`32`



DataLoader workers:



`2`



Device:



`cuda`



Learning rate:



`0.0001`



Weight decay:



`0.00001`



Maximum epochs:



`20`



Early stopping patience:



`5`



\---



\## 8. Training Results



Training used the full train split:



`7,492 images`



Validation used:



`1,606 images`



Observed validation accuracy:



| Epoch | Train Loss | Validation Loss | Validation Accuracy |

|---:|---:|---:|---:|

| 1 | 0.1086 | 0.0177 | 0.9963 |

| 2 | 0.0240 | 0.0173 | 0.9963 |

| 3 | 0.0201 | 0.0102 | 0.9988 |

| 4 | 0.0130 | 0.0087 | 0.9988 |

| 5 | 0.0115 | 0.0114 | 0.9975 |

| 6 | 0.0114 | 0.0087 | 0.9988 |

| 7 | 0.0097 | 0.0093 | 0.9988 |

| 8 | 0.0069 | 0.0072 | 0.9975 |



Best validation accuracy:



`99.88%`



The best checkpoint was selected from the validation results using the existing Trainer logic.



Training stopped after epoch 8 because validation accuracy did not improve for 5 consecutive epochs.



\---



\## 9. Held-Out Test Results



The saved best checkpoint was evaluated on the untouched identity-safe test split.



Test samples:



`1,606`



Test composition:



\- Real: `803`

\- Fake: `803`



Results:



| Metric | Result |

|---|---:|

| Loss | `0.017443` |

| Accuracy | `99.7509%` |

| Precision | `99.8752%` |

| Recall | `99.6264%` |

| F1 | `99.7506%` |

| ROC-AUC | `99.9225%` |



Confusion matrix:



```text

&#x20;                Predicted

&#x20;                Real  Fake

Actual Real       802     1

Actual Fake         3   800


Total test errors:

4 / 1,606

10. ONNX Export

Trained checkpoint:

ml_core/checkpoints/best_model.pth

Existing project exporter:

ml_core/export/export_onnx.py

Exported backend model:

backend/app/ml_assets/model.onnx

PyTorch vs ONNX verification:

max difference = 0.00004578

Configured tolerance:

0.0001

Verification result:

PASS

11. Backend Integration

Backend model loading:

PyTorch checkpoint loaded for Grad-CAM/explainable inference
ONNX model loaded for fast inference
ONNX Runtime provider: CPUExecutionProvider

Backend API verification:

Held-out fake image

Input:

test/fake/fake_00000.jpg

Result:

label: fake
confidence: 1.0
real probability: approximately 1.68e-9
fake probability: 1.0

Full /predict endpoint also returned a valid Grad-CAM heatmap data URI.

Held-out real image

Input:

test/real/real_00000.jpg

Result:

label: real
confidence: 0.9964264
fake probability: approximately 0.0035736

Full /predict endpoint also returned a valid Grad-CAM heatmap data URI.

12. Automated Verification

ML test suite:

11 passed

Backend test suite:

4 passed

Frontend Playwright E2E suite:

4 passed

Playwright scenarios verified:

Valid image upload → scan animation → heatmap/confidence
Invalid PDF rejection
Oversized file rejection
Backend-down error handling
13. Reproducibility Notes

Important artifacts:

ml_core/checkpoints/best_model.pth
backend/app/ml_assets/model.onnx
identity-safe split manifest:
identity_safe_split.json
training configuration used in Colab:
/content/deepguard_training_config.yaml

The dataset images and generated model artifacts are intentionally excluded from Git tracking through .gitignore.

The identity-safe split must not be regenerated randomly when reproducing this checkpoint. The existing manifest should be reused.

The 74 unmatched fake files and 74 unmatched real files should remain excluded unless their metadata is independently established.

14. Important Interpretation Note

The 99.7509% test accuracy is the result obtained on this specific identity-safe held-out test split from the Sowaiba01/Deepfake dataset.

It should not be presented as a claim of equivalent real-world performance on arbitrary deepfakes, unseen manipulation techniques, social-media compression, or unrelated datasets.

Additional cross-dataset and cross-manipulation evaluation would be required to make broader generalization claims.

15. Checkpoint Status

Dataset preparation: PASS

Identity-safe split: PASS

Data-loader validation: PASS

Real-data training: PASS

Best checkpoint: AVAILABLE

Held-out test evaluation: PASS

ONNX export: PASS

PyTorch/ONNX numerical verification: PASS

Backend integration: PASS

Grad-CAM integration: PASS

Frontend E2E: PASS

Git working tree after generated artifacts: CLEAN