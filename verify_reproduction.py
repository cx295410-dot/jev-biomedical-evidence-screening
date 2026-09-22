import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--out',default='reproduced');args=p.parse_args()
root=Path(__file__).resolve().parent;out=Path(args.out);count=0
for f in root.glob('reference_*.csv'):
 name=f.name.removeprefix('reference_')
 if name in ['integrity_checks.csv','sha256_manifest.csv']:continue
 pd.testing.assert_frame_equal(pd.read_csv(f),pd.read_csv(out/name),check_dtype=False,rtol=1e-11,atol=1e-12);count+=1
a=json.loads((root/'reference_jev_calibration_summary.json').read_text());b=json.loads((out/'jev_calibration_summary.json').read_text())
for k,v in a.items():assert np.isclose(v,b[k],rtol=1e-11,atol=1e-12),k
print(f'Validated {count} tables and calibration summary against frozen references.')
