import { expect, test } from "@playwright/test";

// End-to-end against the real SPA + Flask JSON API (seeded by serve.sh). Mirrors
// the manual verification: render the dashboard, open a transcript, reassign a
// speaker (which merges adjacent turns), undo, and navigate to settings.

test("dashboard lists the seeded recording", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Recent Recordings")).toBeVisible();
  await expect(page.getByText("Quarterly board meeting").first()).toBeVisible();
  // Library summary reflects the one done recording.
  await expect(page.getByText("Library Summary")).toBeVisible();
});

test("reassign a speaker merges turns; undo restores them", async ({ page }) => {
  await page.goto("/sessions/e2e-demo");

  const turns = page.locator(".turn");
  await expect(turns).toHaveCount(3);

  // Open the reassign popover on the middle turn (Speaker 2). The swap button is
  // hover-revealed, so force the click.
  await page.locator(".turn__swap").nth(1).click({ force: true });
  await expect(page.getByText("Reassign to")).toBeVisible();

  // Reassign to Speaker 1 -> all three turns share a speaker -> collapse to one.
  await page.locator(".spk-opt", { hasText: "Speaker 1" }).first().click();
  await expect(turns).toHaveCount(1);

  // Undo (sl-button enabled once an edit exists) restores the three turns.
  await page.locator("sl-button", { hasText: "Undo" }).click();
  await expect(turns).toHaveCount(3);
});

test("navigates from dashboard to settings", async ({ page }) => {
  await page.goto("/");
  await page.locator("a.sb__link", { hasText: "Settings" }).click();
  await expect(page).toHaveURL(/\/settings$/);
  await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();
});
