"""Glasses nose-bridge check: can GNM, fitted to face landmarks, tell a low nose bridge
from a high one? Same simulation as experiment.py (A: iris scale, B: exact scale).

Measures (mm, model axes: y up, z out of the face):
  bridge_projection - how far the nose root (sellion, MediaPipe 168) stands in front of
                      the inner eye corners (133, 362): small = low bridge
  bridge_height     - how far the sellion sits above the pupils (eyeball centres):
                      small or negative = low bridge
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from gnm_measure import T, MP, joints
from experiment import Bf, REF, VX, IRIS_MM, IRIS_SD, SIG_XY, SIG_Z, fit, rand_rot


def nose(V, J):
    sel = V[MP[168]]
    canthi = (V[MP[133]] + V[MP[362]]) / 2
    pupils = (J[2] + J[3]) / 2
    return {'bridge_projection': 1000 * (sel[2] - canthi[2]), 'bridge_height': 1000 * (sel[1] - pupils[1])}


def person(args):
    seed, iris_on, alpha = args
    rng = np.random.default_rng(seed)
    c_true = rng.standard_normal(170)
    X = T + (c_true @ Bf).reshape(-1, 3)
    iris = IRIS_MM + IRIS_SD * rng.standard_normal() if iris_on else IRIS_MM
    k = IRIS_MM / iris
    noise = rng.standard_normal((len(VX), 3)) * [SIG_XY, SIG_XY, SIG_Z]
    obs = REF + alpha * (X[VX] - T[VX]) + noise
    obs = k * (obs @ rand_rot(rng, 20).T) + rng.uniform(-0.1, 0.1, 3)
    c, s = fit(obs, 0.0)
    return nose(X, joints(c_true)), nose((T + (c @ Bf).reshape(-1, 3)) / k, joints(c) / k)


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    print('Template:', {k: round(v, 1) for k, v in nose(T, joints(np.zeros(1))).items()})
    for iris_on, alpha, label in [(True, 1.0, 'A. iris scale'), (False, 1.0, 'B. exact scale'),
                                  (True, 0.7, 'D. MediaPipe captures 70% of individual shape')]:
        with ProcessPoolExecutor() as ex:
            R = list(ex.map(person, [(s, iris_on, alpha) for s in range(1000, 1000 + N)], chunksize=10))
        print(f'\n=== {label}  (N={N})')
        print(f"  {'measure':18s} {'pop mean':>8s} {'pop SD':>7s} {'RMSE':>6s} {'R²':>6s}  tercile agreement")
        for key in R[0][0]:
            t = np.array([r[0][key] for r in R]); p = np.array([r[1][key] for r in R])
            rmse = np.sqrt(np.mean((p - t) ** 2))
            lo, hi = np.percentile(t, [33.3, 66.7])
            cat = lambda x: np.digitize(x, [lo, hi])
            agree = np.mean(cat(t) == cat(p)); gross = np.mean(np.abs(cat(t) - cat(p)) == 2)
            print(f'  {key:18s} {t.mean():8.1f} {t.std():7.1f} {rmse:6.1f} {1 - rmse**2 / t.var():6.2f}  '
                  f'{100*agree:.0f}% same third, {100*gross:.1f}% low<->high')
