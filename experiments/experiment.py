"""Can GNM fitted to MediaPipe-style face landmarks predict head/ear dimensions,
and how should the mesh be scaled?

Per synthetic person: identity c ~ N(0, I) over GNM's 170 head components,
horizontal iris diameter ~ N(11.7, 0.45) mm (independent of head), random pose.
Observed landmarks follow the XR Blocks differential model:
    obs = ref + alpha * (X_true - T)[vx] + noise      (real mm)
then scaled by k = 11.7 / iris (what the app does when it assumes 11.7 mm).
Fit (166 skull-fixed landmarks): Kabsch rotation, then a joint linear MAP solve for
identity, translation and scale (scale on the model side; see fit()).
Scenarios B-E stress the assumptions (see results.txt / README.md).
Scale handling:
    iris   - scale fixed to 1 (trust the 11.7 mm iris assumption)
    gnm    - scale free (GNM's shape prior decides size)
    fused  - scale prior centred on the iris estimate with SD 3.8%
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')   # one BLAS thread per worker process
import numpy as np, sys
from concurrent.futures import ProcessPoolExecutor
from gnm_measure import T, D, measure, joints

B = D['vertex_identity_basis'][:170].astype(np.float64)
Bf = B.reshape(170, -1)
corr = np.load('corr.npz')
rig = corr['rigid'] == 1
VX = corr['vx'][rig].astype(int)
REF = corr['ref'][rig].astype(np.float64)
M = B[:, VX, :].reshape(170, -1).T                   # (3n, 170)
IRIS_MM, IRIS_SD = 11.7, 0.45
SIG_XY, SIG_Z = 0.001, 0.003                          # per-person residual landmark error (m)
W = np.tile([1 / SIG_XY**2, 1 / SIG_XY**2, 1 / SIG_Z**2], len(VX))


def rand_rot(rng, deg):
    a = np.radians(rng.uniform(-deg, deg, 3))
    cx, sx, cy, sy, cz, sz = np.cos(a[0]), np.sin(a[0]), np.cos(a[1]), np.sin(a[1]), np.cos(a[2]), np.sin(a[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def fit(obs, scale_sd, iters=6):
    """Joint MAP fit with the scale on the model side (scaling the noisy observations
    instead biases scale low: shrinking them shrinks the noise too).
        R obs + t  ~  s (ref + M c)  =  s ref + M c'     with c' = s c
    With R fixed this is linear in (c', s, t). Prior |c'|^2 (~|c|^2 since s ~ 1).
    scale_sd: 0 = s fixed at 1 (trust iris), inf = free (GNM decides size),
              else Gaussian prior on s around 1 (fuse iris and GNM).
    Returns c = c'/s (identity of the real head) and s (observation units per real unit).
    """
    n = len(obs)
    c, s = np.zeros(170), 1.0
    tile_t = np.tile(np.eye(3), (n, 1))
    ref = REF.ravel()
    for _ in range(iters):
        target = s * REF + (M @ (s * c)).reshape(-1, 3)
        a, b = obs - obs.mean(0), target - target.mean(0)
        U, _, Vt = np.linalg.svd(a.T @ b)
        R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
        u = (obs @ R.T).ravel()
        if scale_sd == 0:
            A = np.hstack([M, -tile_t]); rhs = u - ref
            P = np.diag(np.r_[np.ones(170), np.zeros(3)]); p0 = np.zeros(173)
        else:
            A = np.hstack([M, ref[:, None], -tile_t]); rhs = u
            ps = 0.0 if np.isinf(scale_sd) else 1 / scale_sd**2
            P = np.diag(np.r_[np.ones(170), ps, np.zeros(3)]); p0 = np.r_[np.zeros(170), ps, np.zeros(3)]
        theta = np.linalg.solve(A.T @ (W[:, None] * A) + P, A.T @ (W * rhs) + p0)
        s = 1.0 if scale_sd == 0 else theta[170]
        c = theta[:170] / s
    return c, s


def person(args):
    seed, alpha, iris_on, extra_size_sd, noise_mult = args
    rng = np.random.default_rng(seed)
    c_true = rng.standard_normal(170)
    X = T + (c_true @ Bf).reshape(-1, 3)
    g = np.exp(extra_size_sd * rng.standard_normal())        # size not explained by proportions
    X = X * g
    iris = IRIS_MM + IRIS_SD * rng.standard_normal() if iris_on else IRIS_MM
    k = IRIS_MM / iris
    noise = noise_mult * rng.standard_normal((len(VX), 3)) * [SIG_XY, SIG_XY, SIG_Z]
    obs = REF + alpha * (X[VX] - T[VX]) + noise
    obs = k * (obs @ rand_rot(rng, 20).T) + rng.uniform(-0.1, 0.1, 3)
    true = measure(X, joints(c_true) * g)
    out = {'true': true, 'k': k}
    for name, sd in [('iris', 0.0), ('gnm', np.inf), ('fused', IRIS_SD / IRIS_MM)]:
        c, s = fit(obs, sd)
        out[name] = measure(T + (c @ Bf).reshape(-1, 3), joints(c))
        out[name + '_scale'] = k / s                      # estimated / true size (1.0 = exact)
    return out


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    prior = np.load('prior_samples.npz')
    keys = list(prior['keys'])
    sd_prior = prior['A'].std(0)
    configs = [
        (1.0, True, 0.0, 1, 'A. nominal: iris SD 3.8%, landmark noise 1 mm (3 mm depth)'),
        (1.0, False, 0.0, 1, 'B. oracle iris (no iris error): shape-prediction limit'),
        (1.0, True, 0.03, 1, 'C. + 3% head size independent of face proportions'),
        (0.7, True, 0.0, 1, 'D. MediaPipe captures only 70% of individual shape'),
        (1.0, True, 0.0, 2, 'E. double landmark noise'),
    ]
    for alpha, iris_on, extra, nm, label in configs:
        with ProcessPoolExecutor() as ex:
            R = list(ex.map(person, [(s, alpha, iris_on, extra, nm) for s in range(1000, 1000 + N)], chunksize=10))
        print(f'\n=== {label}  (N={N})')
        for mname in ['iris', 'gnm', 'fused']:
            sc = np.array([r[mname + '_scale'] for r in R])
            print(f'  size error [{mname:5s}]: SD {100*np.std(np.log(sc)):.1f}%  bias {100*np.mean(np.log(sc)):+.1f}%')
        print(f"  {'measure':16s} {'pop SD':>7s} | {'RMSE iris':>9s} {'RMSE gnm':>9s} {'RMSE fused':>10s} | R² fused")
        for j, key in enumerate(keys):
            t = np.array([r['true'][key] for r in R])
            row = []
            for mname in ['iris', 'gnm', 'fused']:
                p = np.array([r[mname][key] for r in R])
                row.append(np.sqrt(np.mean((p - t) ** 2)))
            p = np.array([r['fused'][key] for r in R])
            r2 = 1 - np.mean((p - t) ** 2) / np.var(t)
            print(f'  {key:16s} {sd_prior[j]:7.1f} | {row[0]:9.1f} {row[1]:9.1f} {row[2]:10.1f} | {r2:6.2f}')
