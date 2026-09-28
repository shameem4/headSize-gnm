// Check gnm-model.js against the Python reference (web_test_case.json).
// Run: node test_web.mjs   (after python3 export_web.py && python3 make_web_test.py)
import fs from "fs";
import { loadGNM } from "../gnm-model.js";

const buf = fs.readFileSync(new URL("../gnm_head_fit.bin", import.meta.url));
const model = await loadGNM(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const tc = JSON.parse(fs.readFileSync(new URL("web_test_case.json", import.meta.url)));

let worst = 0;
const report = (label, js, py, tol) => {
  const d = Math.abs(js - py);
  worst = Math.max(worst, d / tol);
  console.log(`${d <= tol ? "ok  " : "FAIL"} ${label.padEnd(28)} js ${js.toFixed(3).padStart(9)}  py ${py.toFixed(3).padStart(9)}`);
};

const zero = new Float64Array(model.K);
const t0 = model.measure(model.mesh(zero), model.eyeJoints(zero)).values;
for (const k of Object.keys(tc.templateValues)) report(`template ${k}`, t0[k], tc.templateValues[k], 0.01);

const start = performance.now();
const { c, scale } = model.fit(Float64Array.from(tc.obs), tc.scaleSd);
const fitMs = performance.now() - start;
report("fit scale", scale, tc.scale, 1e-4);
const cErr = Math.max(...Array.from(c, (v, i) => Math.abs(v - tc.c[i])));
report("fit max |c - c_py|", cErr, 0, 0.01);
const m = model.measure(model.mesh(c), model.eyeJoints(c)).values;
for (const k of Object.keys(tc.fitValues)) report(`fit ${k}`, m[k], tc.fitValues[k], 0.1);
console.log(`fit took ${fitMs.toFixed(0)} ms; ${worst <= 1 ? "ALL OK" : "MISMATCH"}`);
process.exit(worst <= 1 ? 0 : 1);
