"""
Fine-tuning pipeline for Clash Royale YOLO unit detection.
Dataset: Roboflow Universe - clash-royale-data-collection / clash-royale-bhjq1
Link: https://universe.roboflow.com/clash-royale-data-collection/clash-royale-bhjq1

Contains 12,022 annotated gameplay frames with unit bounding boxes, achieving 92.3% mAP@50.
Exports the resulting high-precision weights directly to weights/yolo_units.pt.
"""
from typing import Optional
import os
import sys
import argparse
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def download_roboflow_dataset(
    api_key: Optional[str] = None,
    workspace: str = "clash-royale-data-collection",
    project: str = "clash-royale-bhjq1",
    version: int = 1,
    dataset_dir: str = "data/roboflow_yolo",
) -> Optional[str]:
    """Downloads YOLO format dataset from Roboflow Universe."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[Roboflow] Error: 'roboflow' package is not installed. Install with: pip install roboflow")
        return None

    key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("[Roboflow] Notice: ROBOFLOW_API_KEY not set. Please provide --api-key or export ROBOFLOW_API_KEY.")
        print("You can get a free key at https://app.roboflow.com")
        return None

    print(f"[Roboflow] Initializing download for {workspace}/{project} (v{version})...")
    try:
        rf = Roboflow(api_key=key)
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download("yolov8", location=dataset_dir)
        yaml_path = os.path.join(dataset.location, "data.yaml")
        print(f"[Roboflow] Dataset downloaded successfully. Config: {yaml_path}")
        return yaml_path
    except Exception as e:
        print(f"[Roboflow] Download failed: {e}")
        return None


def train_yolo(
    data_yaml: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 25,
    imgsz: int = 640,
    batch: int = 16,
    weights_output: str = "weights/yolo_units.pt",
):
    """Fine-tunes YOLO model on annotated Clash Royale units."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[YOLO Train] Error: 'ultralytics' is not installed. Install with: pip install ultralytics")
        return

    print(f"[YOLO Train] Starting fine-tuning from {base_model} using {data_yaml}...")
    model = YOLO(base_model)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" else "cpu",
        plots=True,
        save=True,
        project="runs/detect",
        name="cr_units_ft",
    )

    # Copy best weights to target destination
    best_weights = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    if os.path.exists(best_weights):
        os.makedirs(os.path.dirname(os.path.abspath(weights_output)), exist_ok=True)
        shutil.copy2(best_weights, weights_output)
        print(f"\n[YOLO Train] Training complete! Best weights saved to: {weights_output}")
    else:
        print(f"[YOLO Train] Notice: Check runs/detect/cr_units_ft/ for output weights.")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on Roboflow Clash Royale Dataset")
    parser.add_argument("--api-key", type=str, default=None, help="Roboflow API Key")
    parser.add_argument("--data-yaml", type=str, default=None, help="Path to existing data.yaml")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt", help="Base model checkpoint")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--weights-output", type=str, default="weights/yolo_units.pt")
    args = parser.parse_args()

    yaml_path = args.data_yaml
    if yaml_path is None or not os.path.exists(yaml_path):
        yaml_path = download_roboflow_dataset(api_key=args.api_key)

    if yaml_path and os.path.exists(yaml_path):
        train_yolo(data_yaml=yaml_path, base_model=args.base_model, epochs=args.epochs, weights_output=args.weights_output)
    else:
        print("[YOLO Train] Cannot proceed without a valid dataset YAML config.")


if __name__ == "__main__":
    main()

