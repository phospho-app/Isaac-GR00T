import os
import torch
import pandas as pd
import json
import cv2
from PIL import Image, ImageDraw
import numpy as np
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# === Setup PaliGemma ===
model_id = "google/paligemma-3b-mix-224"

if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

processor = AutoProcessor.from_pretrained(model_id)
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
model.to(device)
model.eval()

# === Constantes ===
MAX_OBJECTS = 3

def pad_or_truncate(list_of_4d, max_len=MAX_OBJECTS):
    padded = list_of_4d[:max_len] + [[0.0, 0.0, 0.0, 0.0]] * (max_len - len(list_of_4d))
    return np.array(padded, dtype=np.float32)

def pad_grasps(list_of_2d, max_len=MAX_OBJECTS):
    padded = list_of_2d[:max_len] + [[0.0, 0.0]] * (max_len - len(list_of_2d))
    return np.array(padded, dtype=np.float32)

def parse_paligemma_output(text):
    objects = []
    for part in text.split(";"):
        locs = re.findall(r"<loc(\d{3,4})>", part)
        if len(locs) == 4:
            ymin, xmin, ymax, xmax = map(float, locs)
            center_y = (ymin + ymax) / 2
            center_x = (xmin + xmax) / 2
            objects.append({
                "box_2d": [ymin, xmin, ymax, xmax],
                "grasp_point": [center_y, center_x],
            })
    return objects

def get_2D_bbox(img, prompt=None, frame_id=None):
    if prompt is None:
        prompt = "<image> detect orange circle ; black box"

    img_proc = img.convert("RGB").resize((224, 224), resample=Image.Resampling.LANCZOS)
    inputs = processor(text=prompt, images=img_proc, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        in_len = inputs["input_ids"].shape[-1]
        generated_ids = model.generate(**inputs, max_length=in_len + 100)
        generated_text = processor.batch_decode(generated_ids[:, in_len:], skip_special_tokens=True)[0]

    if frame_id is not None:
        print(f"[Frame {frame_id}] 🧠 PaliGemma output:\n{generated_text}")
    return generated_text

def plot_bbox_grasp(img, bbox, grasp_point=None, label=None):
    width, height = img.size
    draw = ImageDraw.Draw(img)

    abs_y1 = int(bbox[0] / 1000 * height)
    abs_x1 = int(bbox[1] / 1000 * width)
    abs_y2 = int(bbox[2] / 1000 * height)
    abs_x2 = int(bbox[3] / 1000 * width)
    abs_x1, abs_x2 = sorted([abs_x1, abs_x2])
    abs_y1, abs_y2 = sorted([abs_y1, abs_y2])
    draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline='red', width=4)

    if grasp_point is not None:
        grasp_y, grasp_x = grasp_point
        abs_grasp_y = int(grasp_y / 1000 * height)
        abs_grasp_x = int(grasp_x / 1000 * width)
        draw.ellipse((abs_grasp_x - 5, abs_grasp_y - 5, abs_grasp_x + 5, abs_grasp_y + 5), fill='blue')

    if label:
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("Arial.ttf", 20)
        except:
            font = None
        text_bbox = draw.textbbox((abs_x1, abs_y1 - 25), label, font=font)
        draw.rectangle(text_bbox, fill='black')
        draw.text((abs_x1, abs_y1 - 25), label, fill='white', font=font)

    return img

# === Dossiers ===
video_dir = os.path.expanduser("~/phosphobot/recordings/lerobot_v2.1/bounding-box-test1/videos/chunk-000/observation.images.secondary_0")
parquet_dir = os.path.expanduser("~/phosphobot/recordings/lerobot_v2.1/bounding-box-test1/data/chunk-000")
os.makedirs("outputs", exist_ok=True)

# Trouver tous les fichiers vidéo
video_files = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4") and f.startswith("episode_"))

for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    episode_id = video_file.replace(".mp4", "")
    
    print(f"\n📁 Processing {episode_id}...")
    
    # Lire uniquement la première frame
    cap = cv2.VideoCapture(video_path)
    success, frame_bgr = cap.read()
    cap.release()

    if not success:
        print(f"❌ Could not read first frame of {video_file}")
        continue

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)

    try:
        raw_response = get_2D_bbox(img_pil, frame_id=0).strip()
        annotations = parse_paligemma_output(raw_response)
    except Exception as e:
        print(f"❌ Parsing failed:", e)
        continue

    annotated = img_pil.copy()
    for obj in annotations:
        annotated = plot_bbox_grasp(
            annotated,
            bbox=obj["box_2d"],
            grasp_point=obj["grasp_point"],
            label=obj["label"]
        )

    out_path = f"outputs/{episode_id}_frame_000.png"
    annotated.save(out_path)
    print(f"✅ Saved to {out_path}")

    # Mettre à jour le fichier parquet
    parquet_path = os.path.join(parquet_dir, f"{episode_id}.parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        
        box_2d_coords = [obj["box_2d"] for obj in annotations]
        grasp_points = [obj["grasp_point"] for obj in annotations]

        df.at[0, "box_2d_coords"] = pad_or_truncate(box_2d_coords)
        df.at[0, "grasp_points"] = pad_grasps(grasp_points)

        df.to_parquet(parquet_path)
        print(f"💾 Updated parquet file")
    else:
        print(f"⚠️ No parquet file found for {episode_id}")

# === Nettoyage final ===
def clean_column(df, column_name, expected_dim):
    try:
        val = df[column_name].iloc[0]
        if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] == expected_dim:
            return df
        else:
            print(f"⚠️ Invalid structure in {column_name}, dropping.")
            return df.drop(columns=[column_name])
    except Exception as e:
        print(f"⚠️ Error reading {column_name}: {e}")
        return df.drop(columns=[column_name])

for file in sorted(os.listdir(parquet_dir)):
    if not file.endswith(".parquet"):
        continue

    path = os.path.join(parquet_dir, file)
    df = pd.read_parquet(path)
    updated = False

    expected_shapes = {
        "box_2d_coords": 4,
        "grasp_points": 2,
    }

    for col, size in expected_shapes.items():
        if col in df.columns:
            new_df = clean_column(df, col, size)
            if new_df is not df:
                df = new_df
                updated = True

    if updated:
        df.to_parquet(path)
        print(f"🧼 {file} cleaned.")
    else:
        print(f"✅ {file} already clean.") 