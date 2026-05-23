import os
from ultralytics import YOLO
import tensorrt as trt

def export_models():
    print("Exporting YOLO to TensorRT...")
    yolo_path = "models/best.pt"
    if os.path.exists(yolo_path):
        model = YOLO(yolo_path)
        # Export to ONNX first, then TensorRT
        model.export(format="engine", device="0", half=True, dynamic=True)
        print("YOLO exported successfully!")
    else:
        print(f"Model not found at {yolo_path}")

    print("\nNote: ArcFace w600k_r50.onnx will be compiled automatically by ONNXRuntime TensorrtExecutionProvider at runtime. No manual conversion needed!")

if __name__ == "__main__":
    export_models()
