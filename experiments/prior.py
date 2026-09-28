import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')   # one BLAS thread per worker process
import numpy as np, time
from concurrent.futures import ProcessPoolExecutor
from gnm_measure import T, D, measure, joints
B = D['vertex_identity_basis'][:170].astype(np.float64)   # head components only
Bf = B.reshape(170, -1)
def one(c):
    return measure(T + (c @ Bf).reshape(-1, 3), joints(c))
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    C = rng.standard_normal((600, 170))
    t = time.time()
    with ProcessPoolExecutor() as ex: M = list(ex.map(one, C, chunksize=20))
    keys = M[0].keys()
    A = np.array([[m[k] for k in keys] for m in M])
    np.savez('prior_samples.npz', C=C, A=A, keys=list(keys))
    print(f'{len(C)} heads in {time.time()-t:.1f}s')
    for k, col in zip(keys, A.T): print(f'{k:16s} mean {col.mean():6.1f}  SD {col.std():5.1f} mm')
    ci = list(keys).index('circumference')
    print('corr(circumference, others):', dict(zip(keys, np.corrcoef(A.T)[ci].round(2))))
