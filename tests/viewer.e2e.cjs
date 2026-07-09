const assert = require("assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { pathToFileURL } = require("url");
const { createRequire } = require("module");

function loadPlaywright() {
  try {
    return require("playwright");
  } catch (error) {
    const bundledModules = path.join(
      os.homedir(),
      ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules"
    );
    try {
      return createRequire(path.join(bundledModules, "noop.js"))("playwright");
    } catch {
      throw error;
    }
  }
}

const { chromium } = loadPlaywright();
const repoRoot = path.resolve(__dirname, "..");
const viewerUrl = `${pathToFileURL(path.join(repoRoot, "index.html")).href}?test=1`;

const tests = [];
function test(name, fn) {
  tests.push({ name, fn });
}

function approx(actual, expected, epsilon = 1e-6) {
  assert.ok(
    Math.abs(actual - expected) <= epsilon,
    `expected ${actual} to be within ${epsilon} of ${expected}`
  );
}

function writeFile(filePath, text) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, text);
  return filePath;
}

function utilizationText(rows) {
  const capacity = "    2     2     4     2     1     3     3     2     2     2";
  const body = rows.map(row => row.join(" ")).join("\n");
  return [
    "== CAPACTIY:",
    "MXU, XLU, VALU, VPOP, EUP, VLOAD, VLOAD:FILL, VSTORE, VSTORE:SPILL, SALU",
    capacity,
    "== UTILIZATION:",
    body,
    ""
  ].join("\n");
}

function smallBundle(kernelName) {
  return [
    "  0   :  { %a = smov 1 }",
    `  1   :  { %b = smov 2 /* entry bundle: %${kernelName} */ }`,
    ""
  ].join("\n");
}

function padToSize(text, size) {
  if (text.length >= size) return text;
  return `${text}\n${"#".repeat(size - text.length)}`;
}

function createFixture() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llo-viewer-test-"));
  const dumpDir = path.join(dir, "dump");
  const sourceDir = path.join(dir, "source");
  const replacementDir = path.join(dir, "replacement");
  fs.mkdirSync(dumpDir, { recursive: true });
  fs.mkdirSync(sourceDir, { recursive: true });
  fs.mkdirSync(replacementDir, { recursive: true });

  const rows = Array.from({ length: 121 }, () => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  rows[0][9] = 1;
  rows[1][9] = 2;
  rows[2][2] = 4;
  rows[4][0] = 2;
  rows[5][0] = 1;
  rows[8][0] = 2;

  const gammaBundle = [
    "  0   :  { %a = smov 1 }",
    "  1   :  { %b = smov 2 }",
    "  2   :  { %c = vadd %a, %b }",
    "  4   :  { %m = vmatprep.mubr.f32.mxu1 %c /* loc(\"/tmp/generated/kernel.py\":2:5 to 2:16) */ }",
    "  5   :  { %n = vmatmul.mubr.f32.mxu0 %m }",
    "  8   :  { %store = vst [vmem:[#allocation1]] %n }",
    "  0x78   :  { }",
    ""
  ].join("\n");

  const alphaBundle = padToSize(smallBundle("alpha_big.1"), 5200);
  const betaBundle = padToSize(smallBundle("beta_target_small.1"), 1000);
  const paddedGammaBundle = padToSize(gammaBundle, 3200);
  const emptyUtil = utilizationText(Array.from({ length: 121 }, () => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));

  const files = [
    writeFile(path.join(dumpDir, "100-alpha_big.1-70-final_bundles.txt"), alphaBundle),
    writeFile(path.join(dumpDir, "100-alpha_big.1-68-final_hlo-static-per-bundle-utilization.txt"), emptyUtil),
    writeFile(path.join(dumpDir, "200-beta_target_small.1-70-final_bundles.txt"), betaBundle),
    writeFile(path.join(dumpDir, "200-beta_target_small.1-68-final_hlo-static-per-bundle-utilization.txt"), emptyUtil),
    writeFile(path.join(dumpDir, "300-gamma_target_large.1-70-final_bundles.txt"), paddedGammaBundle),
    writeFile(path.join(dumpDir, "300-gamma_target_large.1-68-final_hlo-static-per-bundle-utilization.txt"), utilizationText(rows))
  ];

  const kernelSource = writeFile(path.join(sourceDir, "kernel.py"), [
    "def generated_kernel(left, right):",
    "    output_value = left + right",
    "    return output_value",
    ""
  ].join("\n"));
  const helperSource = writeFile(path.join(sourceDir, "helper.py"), [
    "def helper():",
    "    return 1",
    ""
  ].join("\n"));
  const replacementKernelSource = writeFile(path.join(replacementDir, "kernel.py"), [
    "def generated_kernel(left, right):",
    "    output_value = left - right",
    "    replacement_marker = True",
    "    return output_value",
    ""
  ].join("\n"));

  return {
    dir,
    dumpDir,
    dumpFiles: files,
    kernelSource,
    helperSource,
    replacementKernelSource
  };
}

async function openViewer(page) {
  await page.goto(viewerUrl);
  await page.waitForFunction(() => Boolean(window.__lloViewerTest));
}

async function importDump(page, fixture) {
  await openViewer(page);
  await page.setInputFiles("#folderInput", fixture.dumpDir);
  await page.waitForFunction(() => window.__lloViewerTest.getKernelSuggestions().length === 3);
}

async function loadGammaKernel(page, fixture) {
  await importDump(page, fixture);
  await page.fill("#kernelSearchInput", "target large");
  await page.locator(".kernel-suggestion", { hasText: "gamma_target_large.1" }).click();
  await page.waitForFunction(() => document.querySelector("#status")?.textContent.startsWith("Loaded"));
}

async function state(page) {
  return page.evaluate(() => window.__lloViewerTest.getState());
}

async function launchBrowser() {
  try {
    return await chromium.launch({ headless: true });
  } catch (error) {
    const candidates = [
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
      "/Applications/Chromium.app/Contents/MacOS/Chromium",
      "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
    ];
    const executablePath = candidates.find(candidate => fs.existsSync(candidate));
    if (!executablePath) throw error;
    return chromium.launch({ headless: true, executablePath });
  }
}

test("top fuzzy search finds matching final_bundles file", async ({ page, fixture }) => {
  await importDump(page, fixture);
  await page.fill("#kernelSearchInput", "target large");
  const suggestions = await page.evaluate(() => window.__lloViewerTest.getKernelSuggestions());
  assert.deepStrictEqual(suggestions.map(item => item.name), [
    "300-gamma_target_large.1-70-final_bundles.txt"
  ]);
});

test("candidate ordering is size-desc without search and remains size-desc with fuzzy search", async ({ page, fixture }) => {
  await importDump(page, fixture);
  let suggestions = await page.evaluate(() => window.__lloViewerTest.getKernelSuggestions());
  assert.deepStrictEqual(suggestions.map(item => item.name), [
    "100-alpha_big.1-70-final_bundles.txt",
    "300-gamma_target_large.1-70-final_bundles.txt",
    "200-beta_target_small.1-70-final_bundles.txt"
  ]);

  await page.fill("#kernelSearchInput", "target");
  suggestions = await page.evaluate(() => window.__lloViewerTest.getKernelSuggestions());
  assert.deepStrictEqual(suggestions.map(item => item.name), [
    "300-gamma_target_large.1-70-final_bundles.txt",
    "200-beta_target_small.1-70-final_bundles.txt"
  ]);
});

test("dataset title input updates state", async ({ page, fixture }) => {
  await importDump(page, fixture);
  await page.fill("#datasetTitleInput", "q16 smoke baseline");
  const current = await state(page);
  assert.strictEqual(current.datasetTitle, "q16 smoke baseline");
});

test("WASD, zoom slider, and ctrl-wheel change pan/zoom state", async ({ page, fixture }) => {
  await loadGammaKernel(page, fixture);
  await page.locator("#timelineCanvas").click({ position: { x: 700, y: 280 } });
  const initial = await state(page);

  await page.keyboard.press("d");
  let next = await state(page);
  assert.ok(next.offsetCycle > initial.offsetCycle, "D should pan right");

  await page.keyboard.press("a");
  next = await state(page);
  assert.ok(next.offsetCycle <= initial.offsetCycle + 0.01, "A should pan left");

  const beforeW = await state(page);
  await page.keyboard.press("w");
  const afterW = await state(page);
  assert.ok(afterW.cycleWidth > beforeW.cycleWidth, "W should zoom in");

  await page.keyboard.press("s");
  const afterS = await state(page);
  assert.ok(afterS.cycleWidth < afterW.cycleWidth, "S should zoom out");

  await page.locator("#zoomSlider").evaluate(input => {
    input.value = "70";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  const afterSlider = await state(page);
  assert.strictEqual(afterSlider.zoom, 70);
  assert.ok(afterSlider.cycleWidth > afterS.cycleWidth, "slider should zoom in");

  await page.mouse.move(700, 280);
  const beforeWheel = await state(page);
  await page.keyboard.down("Control");
  await page.mouse.wheel(0, -240);
  await page.keyboard.up("Control");
  const afterWheel = await state(page);
  assert.ok(afterWheel.cycleWidth > beforeWheel.cycleWidth, "ctrl-wheel should zoom in");
});

test("color legend click dims non-matching instructions and second click clears it", async ({ page, fixture }) => {
  await loadGammaKernel(page, fixture);
  await page.locator(".legend-filter", { hasText: "MXU" }).click();
  let current = await state(page);
  assert.strictEqual(current.legendFilterKey, "MXU");

  let dimmed = await page.evaluate(() => window.__lloViewerTest.getDimmedInstructions());
  const mxu = dimmed.find(item => item.display.includes("vmatprep"));
  const salu = dimmed.find(item => item.display.includes("smov 1"));
  assert.ok(mxu && !mxu.dimmed, "MXU instruction should remain visible");
  assert.ok(salu && salu.dimmed, "non-MXU instruction should be dimmed");

  await page.locator(".legend-filter", { hasText: "MXU" }).click();
  current = await state(page);
  assert.strictEqual(current.legendFilterKey, "");
  dimmed = await page.evaluate(() => window.__lloViewerTest.getDimmedInstructions());
  assert.ok(dimmed.every(item => !item.dimmed), "second click should clear legend dimming");
});

test("variable search and instruction search report matches and can jump to instructions", async ({ page, fixture }) => {
  await loadGammaKernel(page, fixture);
  await page.fill("#variableSearchInput", "%a");
  await page.waitForFunction(() => document.querySelector("#variableSearchSummary")?.textContent.includes("2 instructions"));
  assert.strictEqual(await page.locator("#variableSearchResults .dep-item").count(), 2);

  await page.fill("#instructionSearchInput", "vmatmul");
  await page.waitForFunction(() => document.querySelector("#instructionSearchSummary")?.textContent.includes("1 matching"));
  await page.locator("#instructionSearchResults .dep-item").first().click();

  const current = await state(page);
  const selected = await page.evaluate(id => {
    return window.__lloViewerTest.getInstructionSummaries().find(instruction => instruction.id === id);
  }, current.selectedId);
  assert.ok(selected.display.includes("vmatmul"), "instruction search result should select the matching instruction");
});

test("utilization stats are numeric and selected cycle range recalculates them", async ({ page, fixture }) => {
  await loadGammaKernel(page, fixture);
  const initialStats = await page.evaluate(() => window.__lloViewerTest.getResourceStats("MXU"));
  assert.strictEqual(initialStats.count, 121);
  assert.strictEqual(initialStats.peak, 1);

  const alignedRow = await page.evaluate(() => window.__lloViewerTest.getResourceRow("MXU", 4));
  assert.deepStrictEqual(alignedRow, { cycle: 4, busy: 2, capacity: 2, ratio: 1 });

  await page.evaluate(() => window.__lloViewerTest.setCycleSelection(4, 5));
  const selectedStats = await page.evaluate(() => window.__lloViewerTest.getResourceStats("MXU"));
  assert.strictEqual(selectedStats.count, 2);
  approx(selectedStats.average, 0.75);
  assert.strictEqual(selectedStats.peak, 1);
});

test("cycle selection summary reports selected range length", async ({ page, fixture }) => {
  await loadGammaKernel(page, fixture);
  await page.evaluate(() => window.__lloViewerTest.setCycleSelection(4, 8));
  const summary = await page.evaluate(() => window.__lloViewerTest.getCycleSelectionSummary());
  assert.strictEqual(summary, "5 cycles");
});

test("empty final_bundles cycles do not shift instruction/utilization alignment", async ({ page, fixture }) => {
  await loadGammaKernel(page, fixture);
  const instructions = await page.evaluate(() => window.__lloViewerTest.getInstructionSummaries());
  const mxuInstruction = instructions.find(item => item.display.includes("vmatprep"));
  assert.strictEqual(mxuInstruction.cycle, 4);

  const row = await page.evaluate(() => window.__lloViewerTest.getResourceRow("MXU", 4));
  assert.strictEqual(row.busy, 2);
  assert.strictEqual(row.ratio, 1);
});

test("code viewer supports multiple files, replacement upload, deletion, and partial-column highlighting", async ({ page, fixture }) => {
  await loadGammaKernel(page, fixture);
  await page.setInputFiles("#codeFileInput", [fixture.kernelSource, fixture.helperSource]);
  await page.waitForFunction(() => window.__lloViewerTest.getState().codeFiles.length === 2);

  let current = await state(page);
  assert.deepStrictEqual(current.codeFiles.map(file => file.basename).sort(), ["helper.py", "kernel.py"]);

  await page.setInputFiles("#codeFileInput", fixture.replacementKernelSource);
  await page.waitForFunction(() => window.__lloViewerTest.getState().codeFiles.find(file => file.basename === "kernel.py")?.lineCount === 4);
  current = await state(page);
  assert.strictEqual(current.codeFiles.length, 2, "same basename upload should replace, not duplicate");
  assert.ok(await page.locator("#codeViewer").textContent().then(text => text.includes("replacement_marker")));

  await page.locator(".code-file-delete", { hasText: "x" }).first().click();
  await page.waitForFunction(() => window.__lloViewerTest.getState().codeFiles.length === 1);

  await page.evaluate(() => window.__lloViewerTest.selectInstructionByText("vmatprep"));
  await page.waitForFunction(() => window.__lloViewerTest.getCodeFocus().focusedLine === "2");
  const focus = await page.evaluate(() => window.__lloViewerTest.getCodeFocus());
  assert.strictEqual(focus.activeFile, "kernel.py");
  assert.strictEqual(focus.focusedLine, "2");
  assert.strictEqual(focus.focusedText, "output_value");
  assert.ok(focus.mutedCount > 0, "non-matching columns should be muted");
});

test("pin root keeps the original dependency root while selecting another instruction", async ({ page, fixture }) => {
  await loadGammaKernel(page, fixture);
  const rootId = await page.evaluate(() => window.__lloViewerTest.selectInstructionByText("vadd"));
  await page.evaluate(() => window.__lloViewerTest.setPinRoot(true));
  const childId = await page.evaluate(() => window.__lloViewerTest.selectInstructionByText("smov 1"));
  const current = await state(page);
  assert.strictEqual(current.pinRoot, true);
  assert.strictEqual(current.selectionRootId, rootId);
  assert.strictEqual(current.selectedId, childId);
  assert.ok(current.selectedDeps.some(dep => dep.related === childId), "root dependency list should still include selected dependency");
});

(async () => {
  const fixture = createFixture();
  const browser = await launchBrowser();
  let failures = 0;
  try {
    for (const item of tests) {
      const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
      try {
        await item.fn({ page, fixture });
        console.log(`ok - ${item.name}`);
      } catch (error) {
        failures += 1;
        console.error(`not ok - ${item.name}`);
        console.error(error);
      } finally {
        await page.close().catch(() => {});
      }
    }
  } finally {
    await browser.close().catch(() => {});
    fs.rmSync(fixture.dir, { recursive: true, force: true });
  }
  if (failures) process.exit(1);
})().catch(error => {
  console.error(error);
  process.exit(1);
});
