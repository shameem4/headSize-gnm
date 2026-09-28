"""Download GNM Head (google/GNM) and the MediaPipe<->GNM landmark correspondence
(google/xrblocks, derived from edualvarado/gnm-webcam-puppet). Both Apache-2.0.
Writes gnm_head.npz (53 MB) and corr.npz next to this script."""
import base64, re, urllib.request
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
GNM_URL = 'https://raw.githubusercontent.com/google/GNM/main/gnm/shape/data/versions/v3_0/gnm_head.npz'
CORR_URL = 'https://raw.githubusercontent.com/google/xrblocks/main/samples/avatar_lab/gnm/FaceCorrespondence.js'

if not (HERE / 'gnm_head.npz').exists():
    urllib.request.urlretrieve(GNM_URL, HERE / 'gnm_head.npz')

js = urllib.request.urlopen(CORR_URL).read().decode()
b = base64.b64decode(re.search(r"PACKED =\s*'([^']+)'", js).group(1))
n = int(re.search(r'const COUNT = (\d+)', js).group(1))
np.savez(HERE / 'corr.npz',
         ref=np.frombuffer(b, np.float32, n * 3, 0).reshape(n, 3),
         lm=np.frombuffer(b, np.uint16, n, 5676),
         vx=np.frombuffer(b, np.uint16, n, 6622),
         rigid=np.frombuffer(b, np.uint8, n, 7568))
print('ok')
