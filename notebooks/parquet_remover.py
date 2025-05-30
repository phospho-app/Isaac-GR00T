import os
import pandas as pd
import numpy as np

parquet_dir = os.path.expanduser(
    "~/phosphobot/recordings/lerobot_v2.1/bounding-box-test1/data/chunk-000"
)

for file in sorted(os.listdir(parquet_dir)):
    if not file.endswith(".parquet"):
        continue

    path = os.path.join(parquet_dir, file)
    df = pd.read_parquet(path)
    print(f"\n📦 {file}")

    for col in df.columns:
        first_val = df[col].iloc[0]
        if isinstance(first_val, list):
            try:
                arr = np.asarray(first_val, dtype=np.float32)
                print(f"✅ {col} ok: shape {arr.shape}")
            except Exception as e:
                print(f"❌ {col} FAILED: {e}")
                print(f"   → value = {first_val}")
        else:
            print(f"ℹ️ {col} not a list: type = {type(first_val)}")
