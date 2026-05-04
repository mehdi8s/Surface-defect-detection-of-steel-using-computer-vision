import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from steel_defect.data import build_dataloaders
from steel_defect.model import build_model
from steel_defect.trainer import evaluate, save_eval_reports
from steel_defect.utils import resolve_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained steel defect model")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--class-map", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    with Path(args.class_map).open("r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    _, val_loader, _ = build_dataloaders(
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args.model_name, num_classes=len(class_to_idx))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    save_eval_reports(y_true, y_pred, idx_to_class, Path(args.output_dir))
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
