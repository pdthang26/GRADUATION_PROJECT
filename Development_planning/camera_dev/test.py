import torch

if torch.cuda.is_available():
    device_id = 0  # Chọn GPU đầu tiên. Nếu bạn có nhiều GPU, bạn có thể chọn một GPU khác.
    device = f"cuda:{device_id}"
    print(f"GPU: {torch.cuda.get_device_name(device_id)}")
else:
    print("Không tìm thấy GPU. Hãy sử dụng CPU hoặc cài đặt driver CUDA.")
    device = 'cpu'

# model_seg = YOLO('YOLO_pretrain_models\yolov8n.pt', device=device)