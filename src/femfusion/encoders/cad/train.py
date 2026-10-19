"""Train CAD-only masked Face reconstruction and log-volume regression."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .dataset import CADDataset, loader
from .model import MaskedCADTask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def select_face_mask(batch, ratio, generator):
    if not 0 < ratio <= 1:
        raise ValueError('mask_ratio must be in (0, 1]')
    node_counts = torch.bincount(batch.batch, minlength=batch.num_graphs).tolist()
    mask = torch.zeros(batch.num_nodes, dtype=torch.bool)
    start = 0
    for n in node_counts:
        k = max(1, min(n, round(n * ratio)))
        picked = torch.randperm(n, generator=generator)[:k]
        mask[start + picked] = True
        start += n
    return mask.to(batch.x.device)


def step(task, batch, face_mask, volume_weight):
    tokens, pred, volume = task(batch, face_mask)
    target = batch.x[face_mask]

    if len(pred) != len(target):
        raise ValueError('Masked-token mapping is not aligned to node features')
    face_mse = F.mse_loss(pred, target)
    volume_target = batch.volume_log.reshape(-1)
    volume_mse = F.mse_loss(volume, volume_target)
    loss = face_mse + volume_weight * volume_mse
    with torch.no_grad():
        err = (pred - target).abs()
        vol_error = (volume - volume_target).abs()
    return loss, {'face_mse': face_mse.detach().item(),
                  'face_mae': err.mean().item(), 'volume_mse': volume_mse.detach().item(),
                  'volume_mae': vol_error.mean().item(), 'loss': loss.detach().item()}, len(target)


def epoch(task, dataloader, device, ratio, weight, seed, optimizer=None):
    task.train(optimizer is not None)
    generator = torch.Generator().manual_seed(seed)
    totals = {k: 0.0 for k in ('loss', 'face_mse', 'face_mae', 'volume_mse', 'volume_mae')}
    seen_graphs = 0
    for batch in dataloader:
        batch = batch.to(device)
        face_mask = select_face_mask(batch, ratio, generator)
        with torch.set_grad_enabled(optimizer is not None):
            loss, metrics, count = step(task, batch, face_mask, weight)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        for key, value in metrics.items():
            totals[key] += value * batch.num_graphs
        seen_graphs += batch.num_graphs
    if seen_graphs == 0:
        raise RuntimeError('No CAD graphs in dataloader')
    return {k: v / seen_graphs for k, v in totals.items()}


def train(config):
    seed = int(config['seed'])
    set_seed(seed)
    device = torch.device(config.get('device', 'cpu'))
    cached = config['cache_dir']
    train_ds, val_ds = CADDataset('train', cached), CADDataset('validation', cached)
    batch_size = int(config['batch_size'])
    train_loader = loader(train_ds, batch_size, shuffle=True, seed=seed)
    val_loader = loader(val_ds, batch_size, shuffle=False, seed=seed)
    model_cfg = config['model']
    task = MaskedCADTask(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(task.parameters(), lr=float(config['learning_rate']),
                                  weight_decay=float(config.get('weight_decay', 1e-4)))
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    (output/'config.json').write_text(json.dumps(config, indent=2))
    (output/'normalizer.json').write_text(json.dumps(train_ds.normalizer, indent=2))
    best, no_improve, history = float('inf'), 0, []
    for index in range(int(config['epochs'])):
        train_metrics = epoch(task, train_loader, device, config['mask_ratio'],
                              config['volume_weight'], seed + index, optimizer)
        val_metrics = epoch(task, val_loader, device, config['mask_ratio'],
                            config['volume_weight'], seed + 100000)
        record = {'epoch': index + 1, **{f'train_{k}': v for k,v in train_metrics.items()},
                  **{f'val_{k}': v for k,v in val_metrics.items()}}
        history.append(record)
        print(f"Epoch {index+1:03d}: train_loss={train_metrics['loss']:.6f} "
              f"val_loss={val_metrics['loss']:.6f} val_face_MAE={val_metrics['face_mae']:.6f}", flush=True)
        if val_metrics['loss'] < best - 1e-8:
            best, no_improve = val_metrics['loss'], 0
            torch.save({'model': task.state_dict(), 'config': config, 'epoch': index+1,
                        'val_metrics': val_metrics, 'normalizer': train_ds.normalizer}, output/'best.pt')
        else:
            no_improve += 1
        if no_improve >= int(config.get('patience', 20)):
            break
    with (output/'history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (output/'metrics.json').write_text(json.dumps({'best_val_loss': best,
                                                  'epochs_ran': len(history)}, indent=2))
    return output/'best.pt'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/encoders/cad.yaml')
    p.add_argument('--cache-dir')
    p.add_argument('--output-dir')
    p.add_argument('--device')
    p.add_argument('--epochs', type=int)
    args = p.parse_args()
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    for key in ('cache_dir','output_dir','device','epochs'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    print(f'Best checkpoint: {train(config)}')


if __name__ == '__main__':
    main()
