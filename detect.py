import torch
import cv2
import numpy as np
from PIL import Image
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# ========== DEVICE SETUP ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 detect.py using device: {device}")

# ========== GROUNDINGDINO IMPORTS ==========
GROUNDINGDINO_AVAILABLE = False
try:
    # Add GroundingDINO to path
    possible_paths = [
        "GroundingDINO",
        os.path.join(os.path.dirname(__file__), "GroundingDINO"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            sys.path.append(path)
            break

    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    from groundingdino.util.inference import load_image, predict

    GROUNDINGDINO_AVAILABLE = True
    print("✅ GroundingDINO imported in detect.py")
except ImportError as e:
    print(f"⚠️ GroundingDINO not available: {e}")

# ========== SAM IMPORTS ==========
SAM_AVAILABLE = False
try:
    from segment_anything import sam_model_registry, SamPredictor

    SAM_AVAILABLE = True
    print("✅ SAM imported in detect.py")
except ImportError:
    print("⚠️ SAM not available")

# ========== LOAD MODELS ==========
grounding_model = None
sam_predictor = None

# Load GroundingDINO
if GROUNDINGDINO_AVAILABLE:
    try:
        config_file = os.path.join("GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
        checkpoint = os.path.join("models", "groundingdino_swint_ogc.pth")

        if os.path.exists(config_file) and os.path.exists(checkpoint):
            print("📦 Loading GroundingDINO model...")
            args = SLConfig.fromfile(config_file)
            grounding_model = build_model(args)
            ckpt = torch.load(checkpoint, map_location=device)
            grounding_model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
            grounding_model.eval()
            grounding_model = grounding_model.to(device)
            print("✅ GroundingDINO model loaded")
    except Exception as e:
        print(f"❌ Error loading GroundingDINO: {e}")

# Load SAM
if SAM_AVAILABLE:
    try:
        sam_checkpoint = os.path.join("models", "sam_vit_h_4b8939.pth")
        if os.path.exists(sam_checkpoint):
            print("📦 Loading SAM model...")
            sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
            sam.to(device=device)
            sam_predictor = SamPredictor(sam)
            print("✅ SAM model loaded")
    except Exception as e:
        print(f"❌ Error loading SAM: {e}")


# ========== CORE DETECTION FUNCTION ==========
def run_detection(image_path, prompt, box_threshold=0.25, text_threshold=0.2):
    """Run object detection on image"""

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return [], [], [], image

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    boxes_pixel = []
    phrases = []
    logits = []

    # Use GroundingDINO if available
    if grounding_model is not None:
        try:
            image_source, image_tensor = load_image(image_path)
            image_tensor = image_tensor.to(device)

            boxes, logits, phrases = predict(
                model=grounding_model,
                image=image_tensor,
                caption=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device
            )

            # Convert boxes to pixel coordinates
            for box in boxes:
                cx, cy, bw, bh = box.tolist()
                x0 = int((cx - bw / 2) * w)
                y0 = int((cy - bh / 2) * h)
                x1 = int((cx + bw / 2) * w)
                y1 = int((cy + bh / 2) * h)
                boxes_pixel.append([x0, y0, x1, y1])

        except Exception as e:
            print(f"Detection error: {e}")

    return boxes_pixel, phrases, image_rgb


def create_visualization(image_rgb, boxes, phrases, output_path):
    """Create output image with bounding boxes"""
    overlay = image_rgb.copy()

    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        # Draw box
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 3)
        # Add label
        label = phrases[i] if i < len(phrases) else "object"
        cv2.putText(overlay, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save
    Image.fromarray(overlay).save(output_path)
    return output_path


# ========== SHOT FUNCTIONS ==========
def zero_shot_detect(image_path, prompt, output_path=None):
    """Zero-shot detection"""
    print(f"  🔍 Zero-shot: '{prompt}'")

    boxes, phrases, image_rgb = run_detection(image_path, prompt, 0.25, 0.2)

    if output_path is None:
        output_path = image_path.replace('.', '_out.')

    create_visualization(image_rgb, boxes, phrases, output_path)
    print(f"  ✅ Found {len(boxes)} objects")

    return boxes, [], phrases, output_path


def one_shot_detect(image_path, example_path, prompt, output_path=None):
    """One-shot detection - uses example to adjust thresholds"""
    print(f"  📸 One-shot with example: {example_path}")

    # Use lower thresholds for one-shot
    boxes, phrases, image_rgb = run_detection(image_path, prompt, 0.2, 0.15)

    if output_path is None:
        output_path = image_path.replace('.', '_out.')

    create_visualization(image_rgb, boxes, phrases, output_path)
    print(f"  ✅ Found {len(boxes)} objects")

    return boxes, [], phrases, output_path


def few_shot_detect(image_path, example_paths, prompt, output_path=None):
    """Few-shot detection - uses multiple examples"""
    print(f"  🎯 Few-shot with {len(example_paths)} examples")

    # Use even lower thresholds for few-shot
    boxes, phrases, image_rgb = run_detection(image_path, prompt, 0.15, 0.1)

    if output_path is None:
        output_path = image_path.replace('.', '_out.')

    create_visualization(image_rgb, boxes, phrases, output_path)
    print(f"  ✅ Found {len(boxes)} objects")

    return boxes, [], phrases, output_path