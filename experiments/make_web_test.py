"""Write web_test_case.json: a synthetic observation plus the Python fit (K components,
fused scale) and measurements, for checking gnm-model.js (see test_web.mjs)."""
import json, os
from pathlib import Path
import numpy as np
os.chdir(Path(__file__).parent)
import experiment as E
from gnm_measure import T, measure, joints
from export_web import K

rng = np.random.default_rng(7)
c_true = rng.standard_normal(170)
X = T + (c_true @ E.Bf).reshape(-1, 3)
obs = E.REF + (X - T)[E.VX] + rng.standard_normal((len(E.VX), 3)) * [E.SIG_XY, E.SIG_XY, E.SIG_Z]
obs = 1.03 * (obs @ E.rand_rot(rng, 20).T) + [0.02, -0.01, -0.5]

E.M = E.B[:, E.VX, :].reshape(170, -1).T[:, :K]               # fit with the first K components only
def fit_k(obs, sd, iters=6):
    n = len(obs); c = np.zeros(K); s = 1.0; tile = np.tile(np.eye(3), (n, 1)); ref = E.REF.ravel()
    for _ in range(iters):
        target = s * E.REF + (E.M @ (s * c)).reshape(-1, 3)
        a, b = obs - obs.mean(0), target - target.mean(0)
        U, _, Vt = np.linalg.svd(a.T @ b); R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
        u = (obs @ R.T).ravel(); A = np.hstack([E.M, ref[:, None], -tile]); ps = 1 / sd**2
        P = np.diag(np.r_[np.ones(K), ps, np.zeros(3)]); p0 = np.r_[np.zeros(K), ps, np.zeros(3)]
        th = np.linalg.solve(A.T @ (E.W[:, None] * A) + P, A.T @ (E.W * u) + p0); s = th[K]; c = th[:K] / s
    return c, s
c, s = fit_k(obs, E.IRIS_SD / E.IRIS_MM)
cfull = np.r_[c, np.zeros(170 - K)]
case = {
    'obs': obs.ravel().tolist(), 'scaleSd': E.IRIS_SD / E.IRIS_MM,
    'c': c.tolist(), 'scale': s,
    'templateValues': measure(T),
    'fitValues': measure(T + (cfull @ E.Bf).reshape(-1, 3), joints(cfull)),
}
Path('web_test_case.json').write_text(json.dumps(case))
print('scale', s, 'fit circumference', case['fitValues']['circumference'])
