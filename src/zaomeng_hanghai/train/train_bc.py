"""专家动作交叉熵预训练：python -m zaomeng_hanghai.train.train_bc。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import uuid

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .expert_data import concatenate, load_episodes, split_episodes
from zaomeng_hanghai.policy import ACTION_NAMES, ActorCritic
from ..core.observation import OBSERVATION_DIM, OBSERVATION_SCHEMA_VERSION


def make_loader(episodes, batch_size, shuffle, seed):
    obs, acts = concatenate(episodes)
    return DataLoader(
        TensorDataset(torch.from_numpy(obs), torch.from_numpy(acts)),
        batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum, count = 0.0, 0
    confusion = torch.zeros((9, 9), dtype=torch.int64)
    with torch.set_grad_enabled(optimizer is not None):
        for observations, actions in loader:
            observations, actions = observations.to(device), actions.to(device)
            logits = model.actor(observations)
            loss = nn.functional.cross_entropy(logits, actions)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5)
                optimizer.step()
            loss_sum += loss.item() * len(actions)
            count += len(actions)
            bins = (actions * 9 + logits.detach().argmax(-1)).cpu()
            confusion += torch.bincount(bins, minlength=81).reshape(9, 9)
    support = confusion.sum(1)
    return {
        "loss": loss_sum / count,
        "accuracy": confusion.diag().sum().item() / count,
        "samples": count,
        "per_action_recall": [
            confusion[i, i].item() / support[i].item() if support[i] else None for i in range(9)
        ],
        "action_counts": support.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def save_checkpoint(path, model, optimizer, epoch, metrics):
    checkpoint = {
        "format_version": 1, "training_stage": "behavioral_cloning", "critic_trained": False,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_names": list(ACTION_NAMES), "model_config": model.model_config,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "metrics": metrics,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/expert"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 ...")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10, help="验证损失连续不改善的轮数；0 禁用")
    args = parser.parse_args(argv)
    if (min(args.epochs, args.batch_size, args.threads, *args.hidden_sizes) <= 0
            or not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.patience < 0):
        parser.error("Epochs, batch size, threads, widths and learning rate must be positive; patience >= 0")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    train, validation = split_episodes(load_episodes(args.data), args.validation_fraction, args.seed)
    train_loader = make_loader(train, args.batch_size, True, args.seed)
    val_loader = make_loader(validation, args.batch_size, False, args.seed) if validation else None
    output = args.output or Path("runs/bc") / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(device=str(device), output=str(output.resolve()))
    config.update(observation_schema_version=OBSERVATION_SCHEMA_VERSION, model_config={
        "observation_dim": OBSERVATION_DIM, "hidden_sizes": args.hidden_sizes, "action_dim": len(ACTION_NAMES)
    })
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "split.json").write_text(json.dumps({
        "train": [str(episode.path) for episode in train],
        "validation": [str(episode.path) for episode in validation],
    }, indent=2), encoding="utf-8")
    model = ActorCritic(hidden_sizes=args.hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.learning_rate)
    print(f"Output: {output.resolve()} | device={device}", flush=True)
    print(f"Train: {len(train)} episodes / {len(train_loader.dataset)} samples; "
          f"validation: {len(validation)} episodes / {len(val_loader.dataset) if val_loader else 0} samples", flush=True)
    best_loss, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        metrics = {"train": run_epoch(model, train_loader, device, optimizer)}
        if val_loader:
            metrics["validation"] = run_epoch(model, val_loader, device)
        selected = metrics.get("validation", metrics["train"])
        save_checkpoint(output / "last.pt", model, optimizer, epoch, metrics)
        if selected["loss"] < best_loss:
            best_loss, stale = selected["loss"], 0
            save_checkpoint(output / "best.pt", model, optimizer, epoch, metrics)
        else:
            stale += 1
        with (output / "history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps({"epoch": epoch, **metrics}) + "\n")
        summary = " | ".join(f"{key}: loss={value['loss']:.4f} acc={value['accuracy']:.2%}" for key, value in metrics.items())
        print(f"Epoch {epoch}/{args.epochs} | {summary}", flush=True)
        if val_loader and args.patience and stale >= args.patience:
            print("Early stopping", flush=True)
            break
    print(f"Best checkpoint: {(output / 'best.pt').resolve()}", flush=True)


if __name__ == "__main__":
    main()
