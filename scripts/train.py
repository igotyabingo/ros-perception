from ultralytics import YOLO
import os

def main():
    # adjust model size depending on trained result: n, s, m, l, x
    model = YOLO("yolov8n.pt")   

    # dataset yaml path
    data_yaml = "datasets/data.yaml"

    # model training
    model.train(
        data=data_yaml,
        imgsz=640,
        epochs=100,
        batch=16,
        name="hotdog",
        project="experiments",
        pretrained=True,
        optimizer='SGD',
        lr0=0.01,
        mosaic=1.0,
    )

    print("Train Completed!")
    print("Weights saved under: experiments/hotdog/weights/")

if __name__ == "__main__":
    main()
