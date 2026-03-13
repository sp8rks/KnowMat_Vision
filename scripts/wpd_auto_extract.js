const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");
const { chromium } = require("playwright");

function parseArgs(argv) {
  const out = {
    queueDir: "outputs_xy_classified_p10_v2/webplotdigitizer_queue",
    maxImages: 3,
    outDir: "outputs_wpd_auto_extract",
    pythonBin: process.env.PYTHON_BIN || "/Users/tanmayaramachandran/anaconda3/bin/python3",
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--queue-dir" && argv[i + 1]) out.queueDir = argv[++i];
    else if (a === "--max-images" && argv[i + 1]) out.maxImages = parseInt(argv[++i], 10);
    else if (a === "--out-dir" && argv[i + 1]) out.outDir = argv[++i];
    else if (a === "--python-bin" && argv[i + 1]) out.pythonBin = argv[++i];
  }
  return out;
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (ch === "\r") {
      // ignore
    } else {
      field += ch;
    }
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

function toCsv(fields, rows) {
  function esc(v) {
    const s = String(v ?? "");
    if (s.includes(",") || s.includes('"') || s.includes("\n")) {
      return `"${s.replace(/"/g, '""')}"`;
    }
    return s;
  }
  const lines = [fields.map(esc).join(",")];
  for (const r of rows) {
    lines.push(r.map(esc).join(","));
  }
  return lines.join("\n") + "\n";
}

function runPrepare(pythonBin, imagePath) {
  const scriptPath = path.resolve("scripts/wpd_prepare_case.py");
  const proc = spawnSync(pythonBin, [scriptPath, "--image", imagePath], {
    encoding: "utf-8",
    maxBuffer: 10 * 1024 * 1024,
  });
  if (proc.status !== 0) {
    return { ok: false, reason: `prepare_failed:${proc.stderr || proc.stdout}` };
  }
  try {
    return JSON.parse(proc.stdout.trim());
  } catch (e) {
    return { ok: false, reason: `prepare_json_parse_failed:${e}` };
  }
}

async function extractOne(page, indexUrl, imagePath, prep) {
  await page.goto(indexUrl, { waitUntil: "load" });
  await page.waitForTimeout(1200);
  await page.setInputFiles("#fileLoadBox", imagePath);
  await page.waitForFunction(
    () => {
      try {
        const info = wpd.imageManager.getImageInfo();
        return info && info.width > 0 && info.height > 0;
      } catch {
        return false;
      }
    },
    { timeout: 20000 }
  );
  await page.waitForTimeout(500);

  const result = await page.evaluate((payload) => {
    try {
      const pd = wpd.appData.getPlotData();
      pd.reset();

      const axes = new wpd.XYAxes();
      const calib = new wpd.Calibration(2);

      const p0 = payload.calibration.p0;
      const p1 = payload.calibration.p1;
      const p2 = payload.calibration.p2;
      const p3 = payload.calibration.p3;
      calib.addPoint(p0[0], p0[1], String(p0[2]), String(p0[3]));
      calib.addPoint(p1[0], p1[1], String(p1[2]), String(p1[3]));
      calib.addPoint(p2[0], p2[1], String(p2[2]), String(p2[3]));
      calib.addPoint(p3[0], p3[1], String(p3[2]), String(p3[3]));

      const calOk = axes.calibrate(calib, false, false, true);
      if (!calOk) {
        return { ok: false, reason: "axes_calibration_failed" };
      }

      axes.name = "XYAxesAuto";
      pd.addAxes(axes, wpd.appData.isMultipage());

      const ds = new wpd.Dataset(2);
      ds.name = "auto_series";
      pd.addDataset(ds, wpd.appData.isMultipage());
      pd.setAxesForDataset(ds, axes);

      for (const pt of payload.pixel_points) {
        ds.addPixel(pt[0], pt[1], null);
      }

      wpd.plotDataProvider.setDataSource(ds);
      const tbl = wpd.plotDataProvider.getData();
      if (!tbl || !tbl.rawData) {
        return { ok: false, reason: "table_generation_failed" };
      }

      return {
        ok: true,
        fields: tbl.fields || ["x", "y"],
        rows: tbl.rawData,
        n_rows: tbl.rawData.length,
      };
    } catch (err) {
      return { ok: false, reason: `evaluate_exception:${String(err)}` };
    }
  }, prep);

  return result;
}

async function main() {
  const args = parseArgs(process.argv);
  const queueDir = path.resolve(args.queueDir);
  const manifestPath = path.join(queueDir, "manifest.csv");
  const outDir = path.resolve(args.outDir);
  fs.mkdirSync(outDir, { recursive: true });

  if (!fs.existsSync(manifestPath)) {
    throw new Error(`Manifest not found: ${manifestPath}`);
  }

  const csvText = fs.readFileSync(manifestPath, "utf-8");
  const rows = parseCsv(csvText);
  if (rows.length < 2) {
    throw new Error("Manifest has no rows");
  }
  const header = rows[0];
  const qIdx = header.indexOf("queue_image");
  if (qIdx < 0) {
    throw new Error("manifest missing queue_image column");
  }

  const imagePaths = rows
    .slice(1)
    .map((r) => (r[qIdx] || "").trim())
    .filter((x) => x.length > 0)
    .slice(0, Math.max(0, args.maxImages));

  const indexUrl = "file://" + path.resolve("node_modules/@insilicall/img2data/index.html");
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1200 } });

  const summary = [];
  for (let i = 0; i < imagePaths.length; i++) {
    const imgRel = imagePaths[i];
    const imagePath = path.resolve(imgRel);
    const base = path.basename(imagePath, path.extname(imagePath));
    const prep = runPrepare(args.pythonBin, imagePath);

    if (!prep.ok) {
      summary.push({
        image: imagePath,
        status: "failed",
        stage: "prepare",
        reason: prep.reason || "unknown_prepare_error",
      });
      continue;
    }

    const result = await extractOne(page, indexUrl, imagePath, prep);
    if (!result.ok) {
      summary.push({
        image: imagePath,
        status: "failed",
        stage: "extract",
        reason: result.reason || "unknown_extract_error",
      });
      continue;
    }

    const fields = (result.fields || []).slice(0, 2);
    const outRows = (result.rows || []).map((r) => [r[0], r[1]]);
    const outCsv = toCsv(fields.length ? fields : ["x", "y"], outRows);
    const outCsvPath = path.join(outDir, `${base}_wpd.csv`);
    fs.writeFileSync(outCsvPath, outCsv, "utf-8");
    const shotPath = path.join(outDir, `${base}_wpd.png`);
    await page.screenshot({ path: shotPath, fullPage: true });

    summary.push({
      image: imagePath,
      status: "success",
      stage: "done",
      trace_source: prep.trace_source,
      prepared_points: prep.point_count,
      exported_rows: outRows.length,
      csv_file: outCsvPath,
      screenshot_file: shotPath,
    });
  }

  await browser.close();

  const summaryPath = path.join(outDir, "summary.json");
  fs.writeFileSync(summaryPath, JSON.stringify({ queue_dir: queueDir, summary }, null, 2), "utf-8");
  console.log(JSON.stringify({ out_dir: outDir, summary_file: summaryPath, items: summary }, null, 2));
}

main().catch((err) => {
  console.error(String(err));
  process.exit(1);
});
