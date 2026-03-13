const path = require("path");
const { chromium } = require("playwright");

async function main() {
  const indexPath = path.resolve("node_modules/@insilicall/img2data/index.html");
  const url = "file://" + indexPath;

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1200 } });
  await page.goto(url, { waitUntil: "load" });
  await page.waitForTimeout(1500);

  const status = await page.evaluate(() => {
    return {
      hasWpd: typeof wpd !== "undefined",
      wpdKeys: typeof wpd !== "undefined" ? Object.keys(wpd).length : 0,
      hasImageManager: typeof wpd !== "undefined" && !!wpd.imageManager,
      hasAutoExtraction: typeof wpd !== "undefined" && !!wpd.autoExtraction,
      hasDataTable: typeof wpd !== "undefined" && !!wpd.dataTable,
    };
  });

  const outDir = path.resolve("outputs_wpd_smoke");
  await page.screenshot({ path: path.join(outDir, "wpd_home.png"), fullPage: true }).catch(async () => {
    // Ensure directory exists if screenshot failed due missing parent.
    const fs = require("fs");
    fs.mkdirSync(outDir, { recursive: true });
    await page.screenshot({ path: path.join(outDir, "wpd_home.png"), fullPage: true });
  });

  console.log(JSON.stringify(status, null, 2));
  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
