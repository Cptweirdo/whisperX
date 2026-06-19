import { expect, test, type Page } from "@playwright/test";

// End-to-end against the real SPA + native (C++) JSON API (seeded by serve.sh).
// Mirrors the manual verification: render the dashboard, open a transcript,
// reassign a speaker (which merges adjacent turns), undo, split a passage onto
// another speaker (the edit-mode flow), and navigate to settings.

// Reproduce the edit-mode selection the SPA reads: select word spans
// [from..to] of turn `turnIdx` and fire the contextmenu the handler listens for
// (TranscriptBody.svelte::onTurnContextMenu reads window.getSelection()). This
// is exactly what a user's drag-select + right-click produces.
async function selectAndContextMenu(
  page: Page,
  turnIdx: number,
  from: number,
  to: number,
) {
  await page.evaluate(
    ({ turnIdx, from, to }) => {
      const turn = document.querySelectorAll<HTMLElement>(".turn__text")[turnIdx];
      const w = turn.querySelectorAll<HTMLElement>(".seg");
      const r = document.createRange();
      r.setStartBefore(w[from]);
      r.setEndAfter(w[to]);
      const s = getSelection()!;
      s.removeAllRanges();
      s.addRange(r);
      const box = w[to].getBoundingClientRect();
      turn.dispatchEvent(
        new MouseEvent("contextmenu", {
          bubbles: true,
          cancelable: true,
          clientX: box.right,
          clientY: box.top,
        }),
      );
    },
    { turnIdx, from, to },
  );
  await expect(page.locator(".ctx-menu")).toBeVisible();
}

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

test("split reassigns a passage to another speaker; undo restores", async ({ page }) => {
  await page.goto("/sessions/e2e-demo");
  const turns = page.locator(".turn");
  await expect(turns).toHaveCount(3);

  // Edit mode gates the select->right-click->reassign flow.
  await page.locator(".editmode").click();
  await expect(page.locator(".edit-hint")).toBeVisible();

  // Select "everyone to the" (words 1..3) inside turn 0 ("Welcome everyone to
  // the meeting today.", SPEAKER_00) and move it to Speaker 2.
  await selectAndContextMenu(page, 0, 1, 3);
  await expect(page.locator(".ctx-menu__ptext")).toHaveText("“everyone to the”");
  await page.locator(".ctx-item--primary").click();
  await page.locator(".ctx-menu .ctx-item", { hasText: "Speaker 2" }).first().click();

  // The turn splits 3-way: head + tail keep Speaker 1, the middle moves to
  // Speaker 2 -> the 3 turns become 5.
  await expect(turns).toHaveCount(5);
  await expect(page.locator(".turn").nth(0)).toContainText("Welcome");
  const mid = page.locator(".turn").nth(1);
  await expect(mid.locator(".turn__speaker")).toHaveText("Speaker 2");
  await expect(mid.locator(".turn__text")).toHaveText("everyone to the");
  await expect(page.locator(".turn").nth(2).locator(".turn__text")).toHaveText("meeting today.");

  // Undo restores the original single turn.
  await page.locator("sl-button", { hasText: "Undo" }).click();
  await expect(turns).toHaveCount(3);
  await expect(page.locator(".turn__text").first()).toHaveText(
    "Welcome everyone to the meeting today.",
  );
});

test("split can mint a new speaker for the passage", async ({ page }) => {
  await page.goto("/sessions/e2e-demo");
  const turns = page.locator(".turn");
  await expect(turns).toHaveCount(3);
  await page.locator(".editmode").click();

  // Trailing selection "meeting today." (words 4..5) -> head stays Speaker 1,
  // the passage goes to a brand-new speaker "Alice".
  await selectAndContextMenu(page, 0, 4, 5);
  await page.locator(".ctx-item--primary").click();
  await page.locator(".spk-add input").fill("Alice");
  await page.locator(".spk-add button", { hasText: "Add" }).click();

  // Turn 0 splits into head (Speaker 1) + tail (Alice): 3 -> 4 turns.
  await expect(turns).toHaveCount(4);
  const alice = page.locator(".turn").nth(1);
  await expect(alice.locator(".turn__speaker")).toHaveText("Alice");
  await expect(alice.locator(".turn__text")).toHaveText("meeting today.");
});

test("navigates from dashboard to settings", async ({ page }) => {
  await page.goto("/");
  await page.locator("a.sb__link", { hasText: "Settings" }).click();
  await expect(page).toHaveURL(/\/settings$/);
  await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();
});
