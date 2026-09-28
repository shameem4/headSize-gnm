"""Export a trimmed GNM Head for the browser demo (../gnm_head_fit.bin).

Keeps: skin vertices (+ one corneal-apex vertex per eye), the first K head identity
components as int8 with a per-component scale (float32 at the rigid fitting points), eye joint positions and their identity
basis, skin triangles, per-vertex measurement flags, and the MediaPipe correspondence.

Layout: 'GNMF' | uint32 JSON length | JSON header | padding to 4 | sections.
Header lists each section as {name, dtype, shape, offset} (offset from data start).
Units are metres; axes x = subject's left, y up, z out of the face.
"""
import json, os
from pathlib import Path
import numpy as np

K = 100
OUT = Path(__file__).parent / '../gnm_head_fit.bin'


def main():
    os.chdir(Path(__file__).parent)
    D = np.load('gnm_head.npz')
    G = {n: g > 0.5 for n, g in zip(D['vertex_group_names'], D['vertex_groups'])}
    T = D['template_vertex_positions']
    corr = np.load('corr.npz')

    skin = np.flatnonzero(G['skin'])
    apex = [int(np.flatnonzero(G['eye_exteriors'] & side)[np.argmax(T[G['eye_exteriors'] & side, 2])])
            for side in (T[:, 0] > 0, T[:, 0] < 0)]                    # left, right corneal apex
    keep = np.r_[skin, apex]
    remap = np.full(len(T), -1, np.int64)
    remap[keep] = np.arange(len(keep))
    assert (remap[corr['vx']] >= 0).all()

    tris = D['triangles']
    display = G['skin'] & ~G['mouth_sock']                          # hide the mouth interior
    tris = remap[tris[display[tris].all(1)]]

    basis = D['vertex_identity_basis'][:K][:, keep, :]                 # (K, n, 3)
    scale = np.abs(basis).reshape(K, -1).max(1) / 127
    q = np.round(basis / scale[:, None, None]).astype(np.int8)
    print('int8 max abs error (mm):', float(np.abs(q * scale[:, None, None] - basis).max() * 1000))

    x = T[keep, 0]
    flags = (((G['skin_exterior'] & ~G['ears'])[keep]) * 1
             | (G['ears'][keep] & (x > 0)) * 2 | (G['ears'][keep] & (x < 0)) * 4
             | G['middle_brow_region'][keep] * 8
             | G['left_temple_region'][keep] * 16 | G['right_temple_region'][keep] * 32).astype(np.uint8)
    flags[-2:] = 0                                                      # apex vertices: not skin

    sections = {
        'template': T[keep].astype(np.float32),
        'basis': q,
        'basisScale': scale.astype(np.float32),
        'eyeJoints': D['template_joint_positions'][2:4].astype(np.float32),          # left, right
        'eyeJointBasis': D['joint_identity_basis'][:K, 2:4].astype(np.float32),
        'triangles': tris.astype(np.uint16),
        'flags': flags,
        'eyeApex': np.array([len(keep) - 2, len(keep) - 1], np.uint16),             # left, right
        'corrLandmark': corr['lm'].astype(np.uint16),
        'corrVertex': remap[corr['vx']].astype(np.uint16),
        'corrRigid': corr['rigid'].astype(np.uint8),
        'corrRef': corr['ref'].astype(np.float32),
        # Full-precision basis at the rigid fitting points: the size/shape trade-off in the
        # fit is weakly determined, and int8 rounding here shifted scale by ~0.3%
        'rigidBasis': D['vertex_identity_basis'][:K][:, corr['vx'][corr['rigid'] == 1], :].astype(np.float32),
    }
    header, blobs, offset = {'K': K, 'numVertices': len(keep), 'sections': []}, [], 0
    for name, arr in sections.items():
        arr = np.ascontiguousarray(arr)
        header['sections'].append({'name': name, 'dtype': str(arr.dtype), 'shape': list(arr.shape), 'offset': offset})
        raw = arr.tobytes()
        raw += b'\0' * (-len(raw) % 4)
        blobs.append(raw)
        offset += len(raw)
    js = json.dumps(header).encode()
    js += b' ' * (-(8 + len(js)) % 4)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes(b'GNMF' + np.uint32(len(js)).tobytes() + js + b''.join(blobs))
    print(f'{OUT} {OUT.stat().st_size / 1e6:.2f} MB, {len(keep)} vertices, {len(tris)} triangles, K={K}')


if __name__ == '__main__':
    main()
