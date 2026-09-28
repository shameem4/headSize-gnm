# GNM Head fit: feasibility for glasses / hat / headphone / earbud sizing

**Question:** if we fit [GNM Head](https://github.com/google/GNM) (a statistical 3D model of the whole head, in real-world units) to the face landmarks the app already has, how well can we predict head dimensions the camera can't see? And what should set the mesh's real-world size: the iris, GNM's own sense of size, or both?

**Short answer:**
- **In simulation:** coarse sizing looks feasible for hats, glasses and headphones, but weak for earbuds predicted from the face. Combining the iris with GNM's own size estimate works best.
- **On a real face:** GNM's own size estimate was off by about 5%, so the demo takes size from the iris (or a calibration) and only shape from GNM. See "First real-world observation".
- **Ears** are now measured directly from side views; see "Ear measurement from side views".
- **What's checked so far:** simulation results are an upper bound, and real checks so far cover one person.

## Method

- **Simulated people:** 400 random GNM heads per scenario (identity drawn from the model's prior), each with a random iris size (11.7 ± 0.45 mm, independent of head size), random head pose (±20°) and landmark noise.
- **Observed landmarks:** the 166 skull-fixed MediaPipe↔GNM points from [XR Blocks](https://github.com/google/xrblocks/tree/main/samples/avatar_lab/gnm) (Apache-2.0). The XR Blocks "differential" model accounts for MediaPipe's depth bias: `obs = ref + (X_true − T)[vx] + noise`. They're then scaled by `11.7 / iris`, as the app does.
- **Fit:** rotation by Kabsch, then one joint linear solve for identity, translation and scale, with the scale on the model side.
- **Scale strategies:**
  - **iris:** trust the 11.7 mm iris (scale fixed).
  - **gnm:** GNM's shape prior decides size (scale free).
  - **fused:** scale prior centred on the iris with its real-world spread (SD 3.8%).
- **Measurements** (tape-measure style, see `gnm_measure.py`): head circumference, length, breadth, bitragion breadth, ear-to-ear arc over the head, temple width, eye-to-ear distance, ear length. On the GNM average head they land close to adult ANSUR II averages (e.g. circumference 562 mm, head length 194 mm).

Two bugs were caught along the way and are fixed here:
1. Alternating scale and shape solves didn't converge.
2. Scaling the noisy observations biased size 20% low. That's errors-in-variables; the scale now sits on the model side.

## Results (nominal scenario, RMSE in mm)

| Measure | Population SD | iris | gnm | **fused** | Use |
|---|---|---|---|---|---|
| Head circumference | 26.5 | 22.7 | 13.9 | **12.6** | hats |
| Head length | 10.4 | 7.8 | 5.1 | **4.8** | hats, helmets |
| Head breadth | 8.2 | 6.6 | 4.2 | **3.8** | hats, helmets |
| Bitragion breadth | 8.2 | 6.2 | 3.7 | **3.3** | headphones, glasses |
| Ear-to-ear arc | 14.5 | 13.0 | 8.5 | **7.8** | headphones |
| Temple width | 8.5 | 6.1 | 3.2 | **2.9** | glasses frame width |
| Eye-to-ear | 6.9 | 6.1 | 5.0 | **5.0** | glasses temple length |
| Ear length | 5.7 | 3.9 | 3.8 | **3.8** | earbuds (coarse) |
| IPD (far, eyeball centres) | 3.1 | 2.4 | 1.3 | **1.1** | glasses |
| Face width (127–356) | 8.6 | 5.9 | 3.1 | **2.7** | glasses |
| Eye width (corners) | 1.3 | 0.9 | 0.5 | **0.5** | — |
| Bridge / pad width | 2.9 / 2.2 | 1.5 / 1.3 | 0.9 / 0.8 | **0.8 / 0.7** | glasses |

Size (scale) error: iris 4.1%, gnm 2.1%, **fused 1.8%**.

**Stress tests** (fused circumference RMSE / size error). Full tables are in `results.txt`.

| Scenario | Circumference | Size |
|---|---|---|
| A. Nominal | 12.6 mm | 1.8% |
| B. Perfect iris (limit of predicting shape from the face) | 11.7 mm | 1.6% |
| C. + 3% head size not reflected in face proportions | 16.8 mm | 2.7% |
| D. MediaPipe captures only 70% of individual shape | 15.9 mm | 1.6% |
| E. Double landmark noise | 17.6 mm | 2.7% |

**Real-face sanity check:** fitting GNM (free scale) to MediaPipe's canonical face gives a scale 2.9% larger than GNM's average head. MediaPipe's canonical IPD is 3.1% larger than GNM's, so the two independent models agree.

## What it means for product fit (coarse)

Sizing steps below are approximate industry conventions.

- **Hats:** fitted hat sizes step about 10 mm in circumference; S/M/L bands are about 20 mm. At 12–18 mm RMSE, **S/M/L is realistic, an exact fitted size isn't.** GNM heads are bald, so hair needs an allowance.
- **Glasses:** frame-width categories are about 5–10 mm apart and temple lengths step by 5 mm. Temple width (3–4 mm) is **good enough for narrow/medium/wide**. Temple length (about 5 mm, R² ≈ 0.5) is **about ±1 step**.
- **Headphones:** band arc (8–11 mm) and ear spacing (3–5 mm) are **fine**, since headbands adjust over a much wider range.
- **Earbuds:** only overall ear length is predicted, and weakly (R² ≈ 0.5). Ear canal and concha size aren't observable from face landmarks, so **only a coarse ear-size category**. `ear_check.py` ([ear_results.txt](ear_results.txt)) tried more ear measures on the fitted head. The concha is a fixed region picked on the template. Even with an exact scale none is useful:

  | Measure | Population SD (mm) | RMSE (mm) | R² |
  |---|---|---|---|
  | Ear length | 5.6 | 3.8 | 0.54 |
  | Ear width | 4.3 | 3.5 | 0.31 |
  | Ear protrusion | 5.3 | 4.5 | 0.28 |
  | Concha height | 2.2 | 1.7 | 0.43 |
  | Concha width | 2.0 | 1.5 | 0.42 |
  | Concha depth | 5.1 | 4.6 | 0.20 |

  Errors are 70–90% of the spread between people, so these are close to guessing the average. This is also an upper bound: it assumes real ears vary exactly as GNM's model says and follow the face as it predicts. Real earbud sizing needs the ear itself to be measured (side view).

## Caveats

- **Best case.** The simulated people come from GNM itself, so their proportions relate to size exactly as GNM assumes. Scenario C shows how results degrade when they don't.
- **The GNM prior may be wider than the real population.** Its circumference SD is 26.5 mm versus roughly 20 mm for ANSUR adults, so the R² values are likely flattering. The absolute RMSEs are the better guide.
- **Assumed noise levels.** Landmark noise (1 mm across, 3 mm in depth, after averaging frames) is an assumption; scenario E doubles it.
- **Not modelled:** expression (the fit uses skull-fixed points only), glasses, hair, and occlusion.

## Multi-view sweep vs single-pose fit

`multiview_check.py` ([multiview_results.txt](multiview_results.txt)) compares today's fit (3D MediaPipe landmarks from a near-frontal hold) with a Face-ID-style sweep. The sweep is 40 frames at up to ±25° yaw and ±10° pitch, fitting one head shape plus a pose per frame to the 2D landmark positions only. Scale is exact in both, and the sweep fixes each frame's distance to the face, as the iris would. BETA is how faithfully MediaPipe's depth follows the real face.

| RMSE (mm) | 3D, BETA 1 | Sweep | 3D, BETA 0.5 | Sweep ±25° | Sweep ±15° |
|---|---|---|---|---|---|
| Circumference | 8.7 | 9.4 | 11.5 | 9.4 | 10.6 |
| Length | 3.7 | 4.3 | 5.7 | 4.3 | 5.0 |
| Breadth | 2.8 | 2.9 | 2.8 | 2.9 | 3.0 |
| Nose bridge projection | 0.9 | 0.8 | 1.9 | 0.8 | 1.1 |

The sweep doesn't depend on MediaPipe's depth. It roughly matches the 3D fit when that depth is faithful, and is clearly better when it isn't. Real MediaPipe depth fidelity is unknown.

Caveats:
- It assumes each landmark stays on the same spot of skin at every angle. MediaPipe's face-contour points slide to the silhouette as the head turns, so a real sweep fit should leave them out.
- Poses start near the truth.
- Per-frame noise (~1.8 px) is a guess.

**Real data contradicted it (2026-09-27, one sweep, tape circumference ~580 mm).** The browser version (commit 2db1931, reverted) gave:

| Fit | Circumference | Length | Face width |
|---|---|---|---|
| Single-pose | 568 | 192 | 151.5 |
| Sweep | 575 | 200.5 | 141.2 |
| Sweep, reference-cloud x/y (see below) | 603 | 209 | 151.5 |
| Straight-ahead frames only, reference-cloud x/y | 584 | 197 | 152.9 |

- MediaPipe landmarks sit ~0.7 mm (x/y) off their GNM vertices. The 3D fit's reference cloud absorbs that; a vertex-based 2D fit shrinks the face ~7%.
- With reference-cloud x/y, the turned views still pushed the head ~12 mm longer, most likely because MediaPipe places landmarks differently when the face is turned.
- Both simulation assumptions (landmarks on vertices, the same at every angle) fail for real MediaPipe output, so the sweep was reverted.

## Nose bridge (glasses fit)

`nose_check.py` ([nose_results.txt](nose_results.txt)) asks whether the fitted head can tell a low nose bridge from a high one:
- **Bridge projection:** how far the sellion (MediaPipe 168) sits in front of the inner eye corners.
- **Bridge height:** how high the sellion sits relative to the pupils.

| Scenario | Projection RMSE (mm) | Projection R² | Height RMSE (mm) | Height R² | Right third |
|---|---|---|---|---|---|
| A: iris scale | 0.9 | 0.95 | 0.5 | 0.90 | 84% |
| D: MediaPipe captures 70% of shape | 1.5 | 0.86 | 0.8 | 0.78 | 78% |

The population SDs are 4.1 mm (projection) and 1.7 mm (height). No simulated person was put in the wrong extreme third (low for high). The demo's low / medium / high category uses projection thirds of the GNM population (below 11.0 mm, above 14.6 mm). That's a population split, not an industry standard, and real nose depth hasn't been checked.

## Ear measurement from side views

The demo's "Measure ears" step runs [Ear_Landmarker](https://github.com/shameem4/Ear_Landmarker): 55 points in iBUG numbering, found on a crop around the head. Each point is projected onto the fitted head's ear plane.

What real runs on one person taught (ruler: right ear ~68 mm long, >36 mm wide):
- **Scale at the ear is right.** A printed marker held beside the ear measured 45.3 mm against its 45 mm, and its own distance matched the ear plane's to within about 2.5%.
- **The lobe arc runs onto the cheek.** Its last points (17–19) follow the lobe up to where it joins the cheek, so length stops at point 16, the lobe's lowest point. Length is measured along the ear's principal axis, and width across it.
- **Width depends on how squarely the ear is seen.** Frames seen more obliquely measure wider (up to 38 mm at ~27° off square), so only frames within 20° of square are used.
- **Measure width the way the demo does,** or the numbers won't agree. The front edge is where the top of the ear joins the head (point 0), not the tragus. Measured that way the ruler gave >36 mm; from the tragus it gave 33.
- **Results after each run's scale error:** length 66.6–70.5 mm, width 37.5–39.8 mm over the last three runs. The ± shown in the app is only the spread between frames within a run; run-to-run variation is larger, about ±2 mm.
- **Label placement:**
  - Point 0 sometimes lands on the face in front of the ear, which inflates width.
  - Ear_Landmarker's own README labels the point groups differently from iBUG. The demo follows iBUG (see ears.js).

## Next steps

1. **Real validation (essential):** 5–10 people, tape-measured circumference, head length/breadth and ear length, compared against the fit from a webcam session.
2. **Browser prototype:** done, see below.

## First real-world observation

On the first real test (one person, compared against their own measurements), the fused-scale results read **3–4% small**:

- **GNM alone** put the head **4.9% smaller** than the iris scale did.
- The **iris scale** on its own was about right.
- On a generated test face, GNM alone went the other way (+3.3%).

So GNM's size-from-proportions estimate looks much less reliable on real faces than the simulation suggests. That's expected, since the simulated faces obey GNM's size rules by construction. The reference cloud's scale was checked and matches GNM's mesh within about 1%, so it isn't the cause.

The demo therefore uses **iris scale only**: GNM gives shape, the iris gives size. The fused/GNM-scale results above are kept as simulation history.

## Browser demo

The demo at the repo root is built on this: MediaPipe tracks the face, GNM Head is fitted over 90 frames with the scale set by the iris, and every measurement is taken on the fitted head.

- `export_web.py` writes `gnm_head_fit.bin` (repo root) (4.3 MB). It holds 100 identity components (99.8% of shape variance; circumference RMSE 13.1 vs 12.9 mm with all 170). They're stored as int8, except at the 166 fitting points, which are float32: int8 there shifted the fitted scale by about 0.3%.
- `make_web_test.py` + `test_web.mjs` check `gnm-model.js` against this Python code. Scale and identity match; measurements agree within 0.05 mm.

## Reproduce

```bash
python3 fetch_data.py      # downloads gnm_head.npz (53 MB) and builds corr.npz
python3 prior.py           # prior spread of each measurement
python3 experiment.py 400  # all scenarios, ~25 s on 32 cores
python3 ear_check.py 400   # ear measures predicted from the face
python3 nose_check.py 400  # nose bridge
python3 multiview_check.py 200   # sweep vs single-pose fit (~3 min)
python3 export_web.py      # browser model -> ../gnm_head_fit.bin
python3 make_web_test.py && node test_web.mjs   # JS vs Python check
```

Needs numpy and scipy.
