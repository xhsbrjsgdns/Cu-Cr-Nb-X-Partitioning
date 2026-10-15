"""After stage subprocesses close their logs, finalize and verify every file hash."""
import os
import pandas as pd
import g_common as C
import g_contract as G
count=G.write_manifest(include_logs=True)
path=os.path.join(C.D_MANIFEST,'STEP5G_FILE_HASHES.csv')
for row in pd.read_csv(path).itertuples():
    p=os.path.join(C.ROOT,row.relative_path)
    assert os.path.getsize(p)==row.bytes and C.sha256_file(p)==row.sha256,p
print('FINAL MANIFEST VERIFIED: %d files, including closed stage logs'%count)
