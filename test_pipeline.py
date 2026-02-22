import sys
import cv2
from pathlib import Path

# Add project root to sys path
sys.path.append("/Users/anuragx/Desktop/Monospace")

from perception_engine.api.pipeline import run_pipeline

image_path = "/Users/anuragx/Desktop/Monospace/perception_engine/Offroad_Segmentation_testImages/Color_Images/0000096.png"
with open(image_path, "rb") as f:
    img_bytes = f.read()

weights_path = "/Users/anuragx/Desktop/Monospace/best_model_v6.pth"
print("Running pipeline...")
res = run_pipeline(img_bytes, weights_path, original_filename="0000096.png")

print("Finished!")
if res.get("error"):
    print("ERROR:", res.get("error"))
else:
    print("Path found:", res.get("path_found"))
    print("path cost:", res.get("path_cost"))
    print("Check outputs/api_infer/path_*.png")
