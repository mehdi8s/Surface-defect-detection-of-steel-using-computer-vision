import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from steel_defect.data import build_dataloaders
from steel_defect.model import build_model
from steel_defect.trainer import evaluate, save_eval_reports, save_training_curves, train_one_epoch
from steel_defect.utils import load_config, resolve_device, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train steel defect classifier on NEU-DET")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])

    run_name = f'{cfg["output"]["run_name"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    output_dir = Path(cfg["output"]["output_root"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_to_idx = build_dataloaders(
        dataset_root=cfg["data"]["dataset_root"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["num_workers"],
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model(cfg["train"]["model_name"], num_classes=len(class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (output_dir / "class_to_idx.json").open("w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)
    with (output_dir / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Reload best model for final report artifacts.
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    _, _, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    save_training_curves(history, output_dir)
    save_eval_reports(y_true, y_pred, idx_to_class, output_dir)

    print(f"Training completed. Best val_acc={best_val_acc:.4f}")
    print(f"Artifacts saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
