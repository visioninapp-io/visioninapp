from __future__ import annotations

from AI.model_trainer import build_model, train_model


def main() -> None:
    # Choose a base model by alias ("yolov8n" -> resolves to yolov8n.pt)
    model = build_model("yolov8n", {"epochs": 3, "imgsz": 640, "batch": 8})

    # Train by passing ultralytics YOLO args via fit_params (X/y unused)
    # Ensure you provide a proper dataset YAML file path
    model, _ = train_model(
        model,
        X_train=None,
        y_train=None,
        fit_params={
            "data": "path/to/data.yaml",  # TODO: replace with your dataset YAML
            # You can also set/override training args here
            # e.g., "epochs": 50, "imgsz": 640, "batch": 16
        },
    )

    # Export or run inference as needed
    # e.g., model.predict(source="path/to/images_or_video")


if __name__ == "__main__":
    main()


