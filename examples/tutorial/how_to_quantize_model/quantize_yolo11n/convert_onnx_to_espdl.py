import os
from esp_ppq.api import espdl_interface
from onnxsim import simplify
import onnx
import torch
from torch.utils.data import DataLoader, TensorDataset


def convert_yolo11n_to_espdl(imgsz):
    """Convert ONNX YOLO11n model to ESP-DL format with minimal quantization (FP16 oder höchste Qualität)"""
    
    INPUT_SHAPE = [3, imgsz, imgsz]
    DEVICE = "cpu"
    TARGET = "esp32s3"
    
    # Model paths  
    ONNX_PATH = r"C:\Users\rapha\Documents\GitHub\esp-dl\OWN_FILES\yolo11n_detect_train2_best.onnx"
    ESPDL_MODLE_PATH = r"C:\Users\rapha\Documents\GitHub\esp-dl\OWN_FILES\yolo11n_detect_train2_best_fp16.espdl"

    # Load and simplify ONNX model
    print(f"Loading ONNX model from: {ONNX_PATH}")
    model = onnx.load(ONNX_PATH)
    print("Simplifying ONNX model...")
    model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated" 
    onnx.save(onnx.shape_inference.infer_shapes(model), ONNX_PATH)
    print("ONNX model loaded and simplified successfully.")

    # Create dummy calibration data (minimal requirement)
    print("Creating minimal calibration data...")
    dummy_data = torch.randn(8, 3, imgsz, imgsz)  # 8 dummy samples
    dataset = TensorDataset(dummy_data)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    def collate_fn(batch):
        return batch[0].to(DEVICE)

    # Convert ONNX to ESP-DL format with FP16 (higher quality than INT8)
    print(f"Converting ONNX to ESP-DL format (FP16 - High Quality)...")
    print(f"Output file: {ESPDL_MODLE_PATH}")
    
    espdl_graph = espdl_interface.espdl_quantize_onnx(
        onnx_import_file=ONNX_PATH,
        espdl_export_file=ESPDL_MODLE_PATH,
        calib_dataloader=dataloader,
        calib_steps=2,  # Minimal calibration steps
        input_shape=[1] + INPUT_SHAPE,
        target=TARGET,
        num_of_bits=16,  # FP16 instead of INT8 for better quality
        collate_fn=collate_fn,
        device=DEVICE,
    )
    
    print(f"✅ Conversion completed successfully!")
    print(f"📁 ESP-DL model saved to: {ESPDL_MODLE_PATH}")
    return espdl_graph


if __name__ == "__main__":
    convert_yolo11n_to_espdl(imgsz=640)