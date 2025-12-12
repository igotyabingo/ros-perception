from ultralytics import YOLO

model = YOLO("experiments/hotdog/weights/best.pt")
model.val(data="datasets/data.yaml")
