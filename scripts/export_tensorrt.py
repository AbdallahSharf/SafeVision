import os
from ultralytics import YOLO
import tensorrt as trt

def export_models():
    print("Exporting YOLO to TensorRT...")
    yolo_path = "models/best.pt"
    if os.path.exists(yolo_path):
        engine_path = yolo_path.replace('.pt', '.engine')
        if os.path.exists(engine_path):
            print(f"TensorRT engine already exists at {engine_path}, skipping export.")
            return

        print("Exporting YOLO to TensorRT...")
        model = YOLO(yolo_path)
        try:
            # Export to ONNX first, then TensorRT
            model.export(format="engine", device="0", half=True, dynamic=True)
            print("YOLO exported successfully!")
        except Exception as e:
            print(f"Failed to export YOLO to TensorRT: {e}")
            print("The system will fall back to using PyTorch/ONNX if supported.")
    else:
        print(f"Model not found at {yolo_path}")

    print("\nNote: ArcFace w600k_r50.onnx will be compiled automatically by ONNXRuntime TensorrtExecutionProvider at runtime. No manual conversion needed!")

if __name__ == "__main__":
    export_models()
