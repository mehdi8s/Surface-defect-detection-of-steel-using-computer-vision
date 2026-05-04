import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from steel_defect.model import build_model


def main():
    parser = argparse.ArgumentParser(description="Predict class for one steel defect image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument(
        "--class-map", type=str, required=True, help="Path to class_to_idx.json from training run"
    )
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    with Path(args.class_map).open("r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = build_model(args.model_name, num_classes=len(class_to_idx))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = Image.open(args.image)
    x = tfm(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(probs.argmax().item())

    print(f"Prediction: {idx_to_class[pred_idx]}")
    print(f"Confidence: {float(probs[pred_idx]):.4f}")


if __name__ == "__main__":
    main()
