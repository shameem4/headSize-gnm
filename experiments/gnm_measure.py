"""Anthropometric measurements on GNM Head meshes (metres in, millimetres out).

Coordinates: x = subject's left(+)/right(-), y up, z forward (face at +z).
All are coarse, tape-measure style definitions intended for sizing, not clinical use.
"""
import numpy as np
from scipy.spatial import ConvexHull

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
D = np.load('gnm_head.npz')
T = D['template_vertex_positions'].astype(np.float64)
GROUPS = {n: g > 0.5 for n, g in zip(D['vertex_group_names'], D['vertex_groups'])}
TRIS = D['triangles']

skin = GROUPS['skin_exterior'] & ~GROUPS['ears']
tri_ok = skin[TRIS].all(1)
_e = np.sort(np.concatenate([TRIS[tri_ok][:, [0, 1]], TRIS[tri_ok][:, [1, 2]], TRIS[tri_ok][:, [2, 0]]]), 1)
EDGES = np.unique(_e, axis=0)                      # skin edges (no ears)
EARS = GROUPS['ears']
EAR_L = EARS & (T[:, 0] > 0)
EAR_R = EARS & (T[:, 0] < 0)
MID_BROW = GROUPS['middle_brow_region']
TEMPLE_L, TEMPLE_R = GROUPS['left_temple_region'], GROUPS['right_temple_region']
# Corneal apex (front of each eye), fixed to the template's most forward eye vertex
_ext = GROUPS['eye_exteriors']
APEX_L = int(np.flatnonzero(_ext & (T[:, 0] > 0))[np.argmax(T[_ext & (T[:, 0] > 0), 2])])
APEX_R = int(np.flatnonzero(_ext & (T[:, 0] < 0))[np.argmax(T[_ext & (T[:, 0] < 0), 2])])

# MediaPipe landmark index -> GNM vertex (XR Blocks correspondence), for face measurements
_corr = np.load('corr.npz')
MP = dict(zip(_corr['lm'].tolist(), _corr['vx'].tolist()))
JT = D['template_joint_positions'].astype(np.float64)       # neck, head, left_eye, right_eye
JB = D['joint_identity_basis'].astype(np.float64)


def joints(c):
    """Joint positions for identity c (first len(c) head components)."""
    return JT + np.tensordot(c, JB[:len(c)], 1)


def cut(V, axis, value):
    """Points where skin edges cross the plane V[:, axis] == value."""
    a, b = V[EDGES[:, 0]], V[EDGES[:, 1]]
    da, db = a[:, axis] - value, b[:, axis] - value
    m = (da * db) < 0
    t = (da[m] / (da[m] - db[m]))[:, None]
    return a[m] + t * (b[m] - a[m])


def hull_perimeter(P2):
    h = ConvexHull(P2)
    Q = P2[h.vertices]
    return np.linalg.norm(Q - np.roll(Q, 1, 0), axis=1).sum()


def tragion(V, ear):
    """Front of the ear at mid-ear height (coarse tragion)."""
    E = V[ear]
    mid = E[:, 1].mean()
    band = E[np.abs(E[:, 1] - mid) < 0.006]
    return band[np.argmax(band[:, 2])]


def measure(V, J=None):
    """Dict of measurements in mm for one mesh V (N, 3) in metres.
    J: joint positions (for far IPD between eyeball centres); defaults to template joints."""
    J = JT if J is None else J
    y0 = V[MID_BROW, 1].mean()                       # brow ridge height
    circ, length = 0.0, 0.0
    for h in y0 + np.arange(0.0, 0.061, 0.003):      # tape band above the brows
        P = cut(V, 1, h)[:, [0, 2]]
        if len(P) < 3:
            continue
        circ = max(circ, hull_perimeter(P))
        length = max(length, np.ptp(P[:, 1]))
    breadth = 0.0
    for h in y0 + np.arange(-0.02, 0.081, 0.004):     # widest point above the ears
        P = cut(V, 1, h)
        if len(P):
            breadth = max(breadth, np.ptp(P[:, 0]))

    tl, tr = tragion(V, EAR_L), tragion(V, EAR_R)
    zc = (tl[2] + tr[2]) / 2                           # coronal plane through tragions
    P = cut(V, 2, zc)
    P = P[P[:, 1] > min(tl[1], tr[1])][:, [0, 1]]
    arc = hull_perimeter(np.vstack([P, tl[[0, 1]], tr[[0, 1]]])) - np.linalg.norm(tl[[0, 1]] - tr[[0, 1]])

    temple = V[TEMPLE_L, 0].max() - V[TEMPLE_R, 0].min()
    # Corneal apex (front of eye) to top of the ear on the same side: sets glasses temple length
    def eye_to_ear(apex, ear):
        top = V[ear][np.argmax(V[ear][:, 1])]
        return np.linalg.norm(V[apex] - top)
    eye_ear = (eye_to_ear(APEX_L, EAR_L) + eye_to_ear(APEX_R, EAR_R)) / 2
    ear_len = (np.ptp(V[EAR_L, 1]) + np.ptp(V[EAR_R, 1])) / 2

    d = lambda a, b: np.linalg.norm(V[MP[a]] - V[MP[b]])
    return {k: v * 1000 for k, v in dict(
        ipd_far=np.linalg.norm(J[2] - J[3]),
        face_width=d(127, 356),
        eye_width=(d(362, 263) + d(33, 133)) / 2,
        bridge_width=d(190, 414),
        pad_width=d(114, 343),
        circumference=circ, length=length, breadth=breadth,
        bitragion=np.linalg.norm(tl - tr), ear_to_ear_arc=arc,
        temple_width=temple, eye_to_ear=eye_ear, ear_length=ear_len).items()}
