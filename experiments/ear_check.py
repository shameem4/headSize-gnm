"""Option 1 check for earbuds: how well does GNM, fitted to face landmarks, predict
ear dimensions beyond ear length? Same simulation as experiment.py (scenario A: iris
scale; B: exact scale), ear measurements only.

Ear regions are picked on the template (fixed mesh topology), left ear:
  ear            - GNM 'ears' vertex group
  concha         - the deep bowl in front of the ear canal: ear vertices at most 74 mm
                   from the midline, 256-275 mm high, more than 15 mm forward
Measurements (mm, averaged over both ears where symmetric):
  ear_length     - top to bottom of the ear
  ear_width      - front to back of the ear
  ear_protrusion - how far the ear stands out from the head (lateral extent)
  concha_height, concha_width - extent of the concha region
  concha_depth   - ear rim's outermost point to the concha's deepest point (lateral)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from gnm_measure import T, EAR_L, EAR_R
from experiment import Bf, REF, VX, IRIS_MM, IRIS_SD, SIG_XY, SIG_Z, fit, rand_rot


def concha(ear, side):
    x = side * T[:, 0]
    return ear & (x < 0.074) & (T[:, 1] > 0.256) & (T[:, 1] < 0.275) & (T[:, 2] > 0.015)


CONCHA_L, CONCHA_R = concha(EAR_L, 1), concha(EAR_R, -1)


def ear_measure(V):
    out = {}
    for name, fn in [
        ('ear_length', lambda e, c, s: np.ptp(V[e, 1])),
        ('ear_width', lambda e, c, s: np.ptp(V[e, 2])),
        ('ear_protrusion', lambda e, c, s: np.ptp(s * V[e, 0])),
        ('concha_height', lambda e, c, s: np.ptp(V[c, 1])),
        ('concha_width', lambda e, c, s: np.ptp(V[c, 2])),
        ('concha_depth', lambda e, c, s: (s * V[e, 0]).max() - (s * V[c, 0]).min()),
    ]:
        out[name] = 1000 * (fn(EAR_L, CONCHA_L, 1) + fn(EAR_R, CONCHA_R, -1)) / 2
    return out


def person(args):
    seed, iris_on = args
    rng = np.random.default_rng(seed)
    c_true = rng.standard_normal(170)
    X = T + (c_true @ Bf).reshape(-1, 3)
    iris = IRIS_MM + IRIS_SD * rng.standard_normal() if iris_on else IRIS_MM
    k = IRIS_MM / iris
    noise = rng.standard_normal((len(VX), 3)) * [SIG_XY, SIG_XY, SIG_Z]
    obs = REF + (X[VX] - T[VX]) + noise
    obs = k * (obs @ rand_rot(rng, 20).T) + rng.uniform(-0.1, 0.1, 3)
    c, s = fit(obs, 0.0)
    # the fitted head is drawn/measured at the iris scale: real size = model size * s/k
    return ear_measure(X), ear_measure((T + (c @ Bf).reshape(-1, 3)) / k)


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    print(f'Concha region: {CONCHA_L.sum()} + {CONCHA_R.sum()} vertices')
    print('Template:', {k: round(v, 1) for k, v in ear_measure(T).items()})
    for iris_on, label in [(True, 'A. iris scale (demo, uncalibrated)'), (False, 'B. exact scale (calibrated)')]:
        with ProcessPoolExecutor() as ex:
            R = list(ex.map(person, [(s, iris_on) for s in range(1000, 1000 + N)], chunksize=10))
        print(f'\n=== {label}  (N={N})')
        print(f"  {'measure':16s} {'pop SD':>7s} {'RMSE':>6s} {'R²':>6s}")
        for key in R[0][0]:
            t = np.array([r[0][key] for r in R])
            p = np.array([r[1][key] for r in R])
            rmse = np.sqrt(np.mean((p - t) ** 2))
            print(f'  {key:16s} {t.std():7.1f} {rmse:6.1f} {1 - rmse**2 / t.var():6.2f}')
