import sys
import cv2
from pathlib import Path

# Add project root to sys path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from perception_engine.api.pipeline import run_pipeline

image_path = _PROJECT_ROOT / "perception_engine" / "Offroad_Segmentation_testImages" / "Color_Images" / "0000096.png"
with open(image_path, "rb") as f:
    img_bytes = f.read()

weights_path = _PROJECT_ROOT / "weights" / "best_model_v6.pth"
print("Running pipeline...")
res = run_pipeline(img_bytes, str(weights_path), original_filename="0000096.png")

print("Finished!")
if res.get("error"):
    print("ERROR:", res.get("error"))
else:
    print("Path found:", res.get("path_found"))
    print("path cost:", res.get("path_cost"))
    print("Check outputs/api_infer/path_*.png")
