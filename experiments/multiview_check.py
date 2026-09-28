"""Does a Face-ID-style sweep (many head angles) fit the head better than today's
single-pose fit to MediaPipe 3D landmarks? Simulation, exact scale in both.

Today ("3D fit", experiment.fit): MediaPipe-style 3D landmarks, averaged over frames:
    obs = REF + (X - T)[VX] + persistent noise (1 mm xy, 3 mm z)
  with the depth part of each person's deviation scaled by BETA: how faithfully
  MediaPipe's predicted depth follows the real face (1 = fully, as the other
  simulations assume; 0.5 = halfway to the average face).
Sweep ("2D multi-view fit"): F frames at random yaw/pitch within +-range; each gives
  the 2D image positions of the landmarks (true 3D landmark = its GNM vertex, plus a
  persistent 1 mm offset per person, plus per-frame noise). Fit: one identity c and a
  pose per frame, minimising reprojection error (Gauss-Newton, alternating pose and
  shape), no MediaPipe depth used.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from gnm_measure import T, measure, joints
from experiment import Bf, B, REF, VX, SIG_XY, SIG_Z, fit, rand_rot
from nose_check import nose

K = 170
M3 = B[:, VX, :].transpose(1, 2, 0)          # (n, 3, K) basis at landmark vertices
TV = T[VX]                                   # (n, 3) template landmark vertices
DIST = 0.6                                   # camera distance (m)
SIG_FRAME = 0.0007                           # per-frame 2D landmark noise (m at the face; ~1.8 px at 1080p)
FRAMES = 40
KEYS = ['circumference', 'length', 'breadth', 'bitragion', 'ear_to_ear_arc', 'temple_width', 'bridge_projection']


def rot(yaw, pitch):
    a, b = np.radians(yaw), np.radians(pitch)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return Rx @ Ry


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def project(P):
    """Camera looks down +z (CV convention): normalized image coords"""
    return P[:, :2] / P[:, 2:3]


def proj_jac(P):
    """d(u, v)/d(X, Y, Z) for each point: (n, 2, 3)"""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    J = np.zeros((len(P), 2, 3))
    J[:, 0, 0] = 1 / z; J[:, 0, 2] = -x / z**2
    J[:, 1, 1] = 1 / z; J[:, 1, 2] = -y / z**2
    return J


def multiview_fit(obs2d, poses0, depths, iters=60):
    """obs2d: (F, n, 2); poses0: list of (R, t) initial model->camera poses;
    depths: (F,) distance of the landmarks' centroid in each frame (what the iris gives).
    Returns identity c. Weights: per-frame noise + persistent offset (which doesn't
    average out across frames)."""
    F = len(obs2d)
    sig2 = (SIG_FRAME**2 + F * SIG_XY**2) / DIST**2
    c = np.zeros(K)
    poses = [(R.copy(), t.copy()) for R, t in poses0]
    for _ in range(iters):
        L = TV + M3 @ c                                            # (n, 3) model landmarks
        # pose step: Gauss-Newton per frame
        for f in range(F):
            R, t = poses[f]
            for _ in range(3):
                A = L @ R.T; P = A + t
                J = proj_jac(P)
                r = (obs2d[f] - project(P)).reshape(-1)
                dPdw = -np.stack([skew(a) for a in A])              # (n, 3, 3)
                Jw = np.einsum('nij,njk->nik', J, dPdw); Jt = J
                # distance fixed (exact scale: the iris/calibration fixes it; from 2D alone a
                # bigger head further away looks the same): update rotation and x, y only
                G = np.concatenate([Jw, Jt[:, :, :2]], axis=2).reshape(-1, 5)
                d = np.linalg.lstsq(G, r, rcond=None)[0]
                d = np.r_[d, 0.0]
                w = d[:3]; th = np.linalg.norm(w)
                if th > 1e-12:
                    k = w / th; Kx = skew(k)
                    R = (np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx) @ R
                t = t + d[3:]
                # keep the face at the known distance (the iris fixes the distance to the
                # face, not to the model's origin down at the neck)
                t[2] = depths[f] - (R @ L.mean(0))[2]
            poses[f] = (R, t)
        # shape step: linearize projections around current shape, MAP solve for c
        AtA = np.eye(K); Atb = np.zeros(K)
        for f in range(F):
            R, t = poses[f]
            P = L @ R.T + t
            J = proj_jac(P)                                        # (n, 2, 3)
            Ac = np.einsum('nij,jk,nkl->nil', J, R, M3).reshape(-1, K)
            r = (obs2d[f] - project(P)).reshape(-1) + Ac @ c
            AtA += Ac.T @ Ac / sig2; Atb += Ac.T @ r / sig2
        c = np.linalg.solve(AtA, Atb)
    return c


def all_measures(c_or_X, is_c=True):
    if is_c:
        X, J = T + (c_or_X @ Bf).reshape(-1, 3), joints(c_or_X)
    else:
        X, J = c_or_X
    m = measure(X, J); m.update(nose(X, J))
    return {k: m[k] for k in KEYS}


def person(args):
    seed, beta, yaw_range, pitch_range, iters = (*args, 60)[:5] if len(args) == 4 else args
    rng = np.random.default_rng(seed)
    c_true = rng.standard_normal(K)
    X = T + (c_true @ Bf).reshape(-1, 3)
    true = all_measures((X, joints(c_true)), is_c=False)

    # Today: 3D fit to MediaPipe-style landmarks (exact scale)
    dev = X[VX] - T[VX]
    dev[:, 2] *= beta
    obs = REF + dev + rng.standard_normal((len(VX), 3)) * [SIG_XY, SIG_XY, SIG_Z]
    obs = obs @ rand_rot(rng, 20).T + rng.uniform(-0.1, 0.1, 3)
    c3d, _ = fit(obs, 0.0)

    # Sweep: 2D multi-view fit
    Ltrue = X[VX] + rng.standard_normal((len(VX), 3)) * SIG_XY      # persistent per person
    centre = TV.mean(0)
    obs2d, poses0, depths = [], [], []
    for f in range(FRAMES):
        yaw, pitch = rng.uniform(-yaw_range, yaw_range), rng.uniform(-pitch_range, pitch_range)
        # model (y up, z out of face) -> camera (CV: y down, z forward, face looking at camera)
        R = np.diag([1, -1, -1.0]) @ rot(yaw, pitch)
        t = np.array([0, 0, DIST]) - R @ centre
        P = Ltrue @ R.T + t
        depths.append(P.mean(0)[2])
        obs2d.append(project(P) + rng.standard_normal((len(VX), 2)) * SIG_FRAME / DIST)
        # initial pose: truth perturbed (in practice: today's per-frame pose tracking)
        Rn = rot(rng.normal(0, 3), rng.normal(0, 3))
        poses0.append((R @ Rn, t + np.r_[rng.normal(0, 0.005, 2), 0.0]))
    cmv = multiview_fit(np.array(obs2d), poses0, np.array(depths), iters)

    return true, all_measures(c3d), all_measures(cmv)


def report(R, label):
    print(f'\n=== {label}  (N={len(R)})')
    print(f"  {'measure':18s} {'pop SD':>7s} | {'RMSE 3D':>8s} {'R² 3D':>6s} | {'RMSE sweep':>10s} {'R² sweep':>8s}")
    for k in KEYS:
        t = np.array([r[0][k] for r in R]); a = np.array([r[1][k] for r in R]); b = np.array([r[2][k] for r in R])
        ra, rb = np.sqrt(np.mean((a - t)**2)), np.sqrt(np.mean((b - t)**2))
        print(f'  {k:18s} {t.std():7.1f} | {ra:8.1f} {1 - ra**2 / t.var():6.2f} | {rb:10.1f} {1 - rb**2 / t.var():8.2f}')


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    for beta, yr, pr in [(1.0, 25, 10), (0.5, 25, 10), (0.5, 15, 10)]:
        with ProcessPoolExecutor() as ex:
            R = list(ex.map(person, [(s, beta, yr, pr) for s in range(1000, 1000 + N)], chunksize=4))
        report(R, f'MediaPipe depth fidelity {beta}; sweep yaw +-{yr} deg, pitch +-{pr} deg, {FRAMES} frames')
