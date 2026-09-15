import { test, expect } from "@playwright/test";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const FIXTURES_DIR = path.join(__dirname, "fixtures");
const SCREENSHOTS_DIR = path.join(__dirname, "screenshots");

test.describe("DeepGuard AI End-to-End Suite", () => {
  test("(a) Happy Path: Upload valid image, view scan animation, verify heatmap & confidence", async ({ page }) => {
    await page.goto("/");

    // Verify initial idle state
    await expect(page.getByText(/drop an image to scan/i)).toBeVisible();

    // Upload a valid face JPEG
    const fileChooserInput = page.locator('input[type="file"]');
    await fileChooserInput.setInputFiles(path.join(FIXTURES_DIR, "sample_face.jpg"));

    // Verify scanning stage
    await expect(page.getByText(/analyzing pixel and frequency patterns/i)).toBeVisible();

    // Wait for ResultsPanel to render
    await expect(page.getByText(/verdict/i)).toBeVisible();
    await expect(page.locator('p.font-display').filter({ hasText: /^(real|fake)$/i })).toBeVisible();

    // Verify both original and heatmap images rendered
    const originalImg = page.locator('img[alt="Original upload"]');
    await expect(originalImg).toBeVisible();

    const heatmapImg = page.locator('img[alt="Grad-CAM heatmap overlay"]');
    await expect(heatmapImg).toBeVisible();
    const heatmapSrc = await heatmapImg.getAttribute("src");
    expect(heatmapSrc).toMatch(/^data:image\/png;base64,/);

    // Verify confidence percentage renders
    await expect(page.locator("text=/%/").first()).toBeVisible();

    // Take screenshot of the rendered happy path result
    await page.screenshot({ path: path.join(SCREENSHOTS_DIR, "happy_path_result.png"), fullPage: true });

    // Click "Scan another image" and verify reset back to idle
    await page.getByRole("button", { name: /scan another image/i }).click();
    await expect(page.getByText(/drop an image to scan/i)).toBeVisible();
  });

  test("(b) Rejected Upload: Non-image PDF produces clean error and returns to idle", async ({ page }) => {
    await page.goto("/");

    const fileChooserInput = page.locator('input[type="file"]');
    await fileChooserInput.setInputFiles(path.join(FIXTURES_DIR, "invalid_doc.pdf"));

    // Assert error message
    const errorBanner = page.locator("text=/unsupported file type/i");
    await expect(errorBanner).toBeVisible();

    // Assert upload zone is still interactive / ready for next upload
    await expect(page.getByText(/drop an image to scan/i)).toBeVisible();
  });

  test("(c) Oversized File: File > 10MB triggers client-side error without network request", async ({ page }) => {
    await page.goto("/");

    let apiCalled = false;
    page.on("request", (req) => {
      if (req.url().includes("/api/v1/predict")) {
        apiCalled = true;
      }
    });

    const fileChooserInput = page.locator('input[type="file"]');
    await fileChooserInput.setInputFiles(path.join(FIXTURES_DIR, "oversized.jpg"));

    // Assert 10MB limit error message
    await expect(page.getByText(/limit is 10MB/i)).toBeVisible();

    // Assert no API call was dispatched
    expect(apiCalled).toBe(false);
  });

  test("(d) Backend Down: Failed network request displays friendly error without crash", async ({ page }) => {
    await page.goto("/");

    // Simulate backend network failure
    await page.route("**/api/v1/predict", (route) => route.abort("connectionrefused"));

    const fileChooserInput = page.locator('input[type="file"]');
    await fileChooserInput.setInputFiles(path.join(FIXTURES_DIR, "sample_face.jpg"));

    // Verify scanning stage began
    await expect(page.getByText(/analyzing pixel and frequency patterns/i)).toBeVisible();

    // Wait for clean failure message
    await expect(page.getByText(/the scan failed/i)).toBeVisible();

    // Verify user is returned to idle upload zone (not blank screen)
    await expect(page.getByText(/drop an image to scan/i)).toBeVisible();
  });
});
