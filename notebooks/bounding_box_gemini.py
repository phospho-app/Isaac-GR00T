from dotenv import load_dotenv
import os
import torch
import pandas as pd
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from PIL import Image, ImageDraw
import numpy as np

# Charger la clé API depuis le .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

MODEL_ID = "gemini-2.0-flash"
model = genai.GenerativeModel(model_name=MODEL_ID)

# === Utils ===

def tensor_to_pil(tensor):
    if isinstance(tensor, torch.Tensor):
        img = tensor.clone().detach()
        if img.dtype != torch.uint8:
            img = (img * 255).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
    return tensor  # déjà PIL.Image

def parse_json(text):
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1:
        return text[start:end+1]
    return '[]'

def get_2D_bbox(img, prompt=None) -> str:
    """Utilise Gemini pour détecter des bounding boxes 2D."""
    bounding_box_system_instructions = """
    You are an expert at analyzing images to identify and locate objects.
    Return bounding boxes as a JSON array. Each object in the array should have a "label" (string) and "box_2d" (array of 4 numbers).
    The "box_2d" coordinates must be [ymin, xmin, ymax, xmax], normalized to a 0-1000 scale.
    Never return Python code fencing (```python ... ```) or general markdown fencing (``` ... ```) around the JSON. Only output the raw JSON array.
    If an object is present multiple times, name them uniquely (e.g., "orange circle 1", "orange circle 2").
    """

    if prompt is None:
        prompt = """Analyze the provided image. Detect all distinct lego bricks, small toys, and any items that could be considered a 'blue bin' or a 'yellow bin' present on the desk.
        Ignore the robot arm itself if visible.
        Return your findings strictly as a JSON array, following the format specified in the system instructions.
        Example of the expected JSON output format: [{"label": "blue lego brick", "box_2d": [100, 200, 150, 280]}, {"label": "yellow bin", "box_2d": [500, 600, 700, 850]}]"""

    contents = [
        {"role": "system", "parts": [bounding_box_system_instructions]},
        {"role": "user", "parts": [img, prompt]}
    ]

    response = model.generate_content(
        contents=contents,
        generation_config=GenerationConfig(temperature=0.5),
    )

    return response.text

def plot_bbox_grasp(im_tensor, bbox, grasp_point=None, grasp_angle=0, label=None):
    im = tensor_to_pil(im_tensor)
    img = im.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)

    # Convert normalized bbox to absolute coordinates
    abs_y1 = int(bbox[0] / 1000 * height)
    abs_x1 = int(bbox[1] / 1000 * width)
    abs_y2 = int(bbox[2] / 1000 * height)
    abs_x2 = int(bbox[3] / 1000 * width)

    if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1
    if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

    draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline='red', width=4)

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

# === Chemin vers le dossier des épisodes ===
episodes_dir = os.path.expanduser(
    "~/phosphobot/recordings/lerobot_v2.1/bounding-box-test1/data/chunk-000"
)

# Créer le dossier de sortie
os.makedirs("outputs", exist_ok=True)

# Trouver tous les fichiers parquet dans le dossier
episode_files = [f for f in os.listdir(episodes_dir) if f.endswith('.parquet')]
episode_files.sort()  # Trier pour avoir un ordre cohérent

for episode_file in episode_files:
    episode_path = os.path.join(episodes_dir, episode_file)
    episode_id = episode_file.replace('episode_', '').replace('.parquet', '')
    
    print(f"\n📁 Processing episode {episode_id}...")
    
    # Charger les données de l'épisode
    df = pd.read_parquet(episode_path)
    
    # Prendre uniquement la première frame
    if len(df) > 0:
        frame_tensor = torch.tensor(df.iloc[0]["observations.images.secondary_0"])  # format [C, H, W]
        img_pil = tensor_to_pil(frame_tensor)

        # Appel Gemini
        raw_response = get_2D_bbox(img_pil)
        try:
            annotations = json.loads(parse_json(raw_response))
        except Exception as e:
            print(f"❌ Parsing Gemini response failed:", e)
            continue

        # Annoter image
        annotated = img_pil.copy()
        for obj in annotations:
            box = obj.get("box_2d", [0, 0, 0, 0])
            label = obj.get("label", "")
            annotated = plot_bbox_grasp(frame_tensor, box, grasp_point=None, grasp_angle=0, label=label)

        # Sauvegarde
        out_path = f"outputs/episode_{episode_id}_frame_000.png"
        annotated.save(out_path)
        print(f"✅ Saved to {out_path}")
    else:
        print(f"⚠️ Episode {episode_id} is empty")
