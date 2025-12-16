from ultralytics import YOLO

def main():
    model = YOLO("experiments/hotdog/weights/best.pt")

    model.train(
        data="datasets/data.yaml",
        imgsz=640,
        epochs=30,
        batch=16,
        lr0=0.0003,
        freeze=10,
        mosaic=0.3,
        name="hotdog_pizza_ft",
        project="experiments",
    )

    print("Pizza fine-tuning completed!")
    print("Saved under: experiments/hotdog_pizza_ft/weights/best.pt")

if __name__ == "__main__":
    main()