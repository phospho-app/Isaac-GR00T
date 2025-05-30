#!/usr/bin/env python3
"""
quick_dataset_check.py
------------------------------------------------
• Vérifie la correspondance 1-à-1 vidéo ↔ parquet
• Valide la présence des colonnes box.pickup / box.target
• Détecte les fichiers vides ou illisibles
------------------------------------------------
"""

import os, sys, pandas as pd, cv2
from pathlib import Path
from tqdm import tqdm, trange

DATA_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/home/pa-boss/Isaac-GR00T/bounding-box-test1/"
).expanduser()

video_dir   = DATA_ROOT / "videos/chunk-000/observation.images.secondary_0"
parquet_dir = DATA_ROOT / "data/chunk-000"

issues = []

# 1) liste toutes les vidéos
videos = sorted(p for p in video_dir.glob("episode_*.mp4"))
print(f"🎞️  {len(videos)} vidéos trouvées")

# 2) pour chaque vidéo, chercher le parquet homonyme
for vid in tqdm(videos, desc="scan"):
    ep_id = vid.stem            # episode_XXXX
    pq = parquet_dir / f"{ep_id}.parquet"

    if not pq.exists():
        issues.append(f"❌ parquet manquant pour {ep_id}")
        continue

    # 2.1 lire le parquet
    try:
        df = pd.read_parquet(pq, columns=["box.pickup", "box.target"])
    except Exception as e:
        issues.append(f"❌ parquet illisible {pq.name}: {e}")
        continue

    # 2.2 vérifier colonnes
    for col in ("box.pickup", "box.target"):
        if col not in df.columns:
            issues.append(f"⚠️  colonne {col} absente dans {pq.name}")
            break

    # 2.3 petite stat taille fichier
    if vid.stat().st_size < 10_000:   # <10 kio → suspect
        issues.append(f"⚠️  vidéo minuscule {vid.name} ({vid.stat().st_size} o)")

print("\nRésumé :")
if issues:
    for line in issues:
        print(line)
    print(f"\n🔴 {len(issues)} problème(s) détecté(s)")
else:
    print("✅ Aucune incohérence détectée")
