"""Single-modality masked field reconstruction. No solver surrogate/alignment."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import random
import numpy as np
import torch
from .dataset import FEMDataset, make_loader, fit_normalizer
from .model import FEMEncoder, mask_nodes, reconstruction_loss, physical_metrics


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, normalizer, *, optimizer=None, mask_ratio=.25,
              seed=42, device='cpu', fixed_train_mask=False):
    training = optimizer is not None
    model.train(training)
    weighted_loss = displacement_sum = stress_sum = 0.
    total_nodes = batches = 0
    with torch.set_grad_enabled(training):
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            # In fixed-mask overfit mode, select nodes by stable case_id hash,
            # independent of epoch, batch order and DataLoader shuffling.
            # Normal training uses fresh masks each epoch/batch.
            hidden = mask_nodes(batch, mask_ratio,
                                seed=seed if (fixed_train_mask and training) or not training else seed+i,
                                deterministic=(not training) or fixed_train_mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
            _, pred = model(batch, hidden)
            loss = reconstruction_loss(pred, batch.targets, hidden)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            metrics = physical_metrics(pred, batch.targets, hidden, normalizer)
            n = metrics['masked_nodes']
            weighted_loss += float(loss.detach()) * n
            displacement_sum += metrics['displacement_abs_sum_m']
            stress_sum += metrics['stress_abs_sum_Pa']
            total_nodes += n; batches += 1
    return {'normalized_masked_MSE': weighted_loss / total_nodes,
            'masked_displacement_MAE_mm': displacement_sum / (3 * total_nodes) * 1000.,
            'masked_stress_MAE_MPa': stress_sum / total_nodes / 1e6,
            'masked_nodes': total_nodes, 'batches': batches}



def save_loss_curve_svg(history, destination):
    """Dependency-free vector loss curve, with literal measured train/val values."""
    from html import escape
    width, height, margin = 680, 340, 44
    xlo, xhi = 1, max(2, len(history))
    values = [r[k] for r in history for k in ('train_normalized_masked_MSE',
                                              'validation_normalized_masked_MSE')]
    ymin, ymax = 0., max(1e-9, max(values)*1.1)
    def x(epoch): return margin + (epoch-xlo)/(xhi-xlo)*(width-2*margin)
    def y(v): return height-margin - (v-ymin)/(ymax-ymin)*(height-2*margin)
    paths=[]
    for field,color in [('train_normalized_masked_MSE','#2763ae'),
                        ('validation_normalized_masked_MSE','#d2792e')]:
        points=' '.join(f'{x(r["epoch"]):.1f},{y(r[field]):.1f}' for r in history)
        if len(history)==1:
            points += f' {x(history[0]["epoch"])+.1:.1f},{y(history[0][field]):.1f}'
        paths.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    svg=(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
         f'<rect width="100%" height="100%" fill="white"/>'
         f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>'
         f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>'
         f'<text x="{margin}" y="22" font-size="15">Masked normalized MSE / epoch</text>'
         f'<text x="{margin}" y="{height-8}" font-size="12">epochs (1 - {len(history)})</text>'
         f'<text x="{margin+180}" y="22" fill="#2763ae" font-size="12">Train</text>'
         f'<text x="{margin+235}" y="22" fill="#d2792e" font-size="12">Validation</text>'
         f'<text x="0" y="{margin}" font-size="11">{ymax:.2g}</text>'
         + ''.join(paths) + '</svg>')
    Path(destination).write_text(svg, encoding='utf-8')


def train_main(argv=None):
    p = argparse.ArgumentParser(description='FEM surface MPNN masked-reconstruction trainer')
    p.add_argument('--data-root', default='data/raw/agentfem')
    p.add_argument('--run-dir', default='outputs/fem/training')
    p.add_argument('--cache-dir', default='data/processed/fem/cache')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--max-regions', type=int, default=32)
    p.add_argument('--max-nodes', type=int, default=4096)
    p.add_argument('--mask-ratio', type=float, default=.25)
    p.add_argument('--fixed-train-mask', action='store_true',
                   help='Overfit diagnostic: same masked nodes per case across epochs; '
                        'recommended with --dropout 0 --weight-decay 0')
    p.add_argument('--reference-load-N', type=float, default=1000.)
    p.add_argument('--train-limit', type=int, default=None, help='smoke only; not representative')
    p.add_argument('--validation-limit', type=int, default=None, help='smoke only')
    p.add_argument('--device', default='cpu')
    p.add_argument('--resume', type=Path, default=None,
                   help='Resume exactly from last.pt in the same run-dir (optimizer/RNG/loader states)')
    p.add_argument('--max-epochs-this-run', type=int, default=None,
                   help='Optional bounded execution; total target remains --epochs')
    a = p.parse_args(argv)
    if a.max_epochs_this_run is not None and a.max_epochs_this_run < 1:
        p.error('--max-epochs-this-run must be positive')
    if a.epochs < 1 or a.batch_size < 1:
        p.error('--epochs and --batch-size must be positive')
    if not 0 < a.mask_ratio < 1:
        p.error('--mask-ratio must be between 0 and 1')
    if not 0 <= a.dropout < 1:
        p.error('--dropout must be in [0,1)')
    seed_everything(a.seed)
    run_dir = Path(a.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    config['model_architecture'] = 'token_conditioned'
    if a.resume is not None and not a.resume.is_file():
        p.error(f'No resume checkpoint at {a.resume}')
    if a.resume is not None and a.resume.parent.resolve() != run_dir.resolve():
        p.error('--resume must point to last.pt in the same --run-dir')
    if a.resume is None:
        (run_dir / 'train_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    normalizer = fit_normalizer(a.data_root, a.reference_load_N)  # always ALL 128 train cases
    normalizer.save(run_dir/'normalizer.json')
    common = dict(normalizer=normalizer, cache_dir=a.cache_dir, max_regions=a.max_regions,
                  max_nodes=a.max_nodes, seed=a.seed)
    train_ds = FEMDataset(a.data_root, 'train', limit_cases=a.train_limit, **common)
    val_ds = FEMDataset(a.data_root, 'validation', limit_cases=a.validation_limit, **common)
    train_dl = make_loader(train_ds, a.batch_size, shuffle=True, seed=a.seed)
    val_dl = make_loader(val_ds, a.batch_size, shuffle=False, seed=a.seed)
    model_config = dict(hidden_dim=a.hidden_dim, output_dim=128, num_layers=a.layers, dropout=a.dropout)
    model = FEMEncoder(**model_config).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    history = []
    best = float('inf')
    first_epoch = 1
    if a.resume is not None:
        state = torch.load(a.resume, map_location='cpu', weights_only=True)
        if state.get('architecture_version') != 'token_conditioned':
            raise ValueError('Resume checkpoint has incompatible architecture')
        original = state['training_config']
        must_match = ('data_root', 'batch_size', 'lr', 'weight_decay', 'seed', 'hidden_dim',
                      'layers', 'dropout', 'max_regions', 'max_nodes', 'mask_ratio',
                      'fixed_train_mask', 'reference_load_N', 'train_limit', 'validation_limit')
        for k in must_match:
            if str(original[k]) != str(config[k]):
                raise ValueError(f'Cannot resume: {k} changed ({original[k]} vs {config[k]})')
        if state['model_config'] != model_config or state['normalizer'] != normalizer.__dict__:
            raise ValueError('Cannot resume: model config or training normalizer changed')
        model.load_state_dict(state['state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        history = json.loads((run_dir/'history.json').read_text(encoding='utf-8'))
        first_epoch = int(state['epoch'])+1
        if len(history) != first_epoch-1:
            raise ValueError('Cannot resume: history length does not match checkpoint epoch')
        best = float(state['best_validation_mse'])
        torch.set_rng_state(state['torch_rng_state'].cpu())
        train_dl.generator.set_state(state['train_loader_rng_state'].cpu())
        val_dl.generator.set_state(state['validation_loader_rng_state'].cpu())
        print(f'Resumed from epoch {first_epoch-1}, target epochs={a.epochs}', flush=True)
    last_epoch = min(a.epochs, first_epoch - 1 + a.max_epochs_this_run) if a.max_epochs_this_run else a.epochs
    for epoch in range(first_epoch, last_epoch+1):
        training = run_epoch(model, train_dl, normalizer, optimizer=optimizer,
                             mask_ratio=a.mask_ratio,
                             seed=a.seed if a.fixed_train_mask else a.seed+epoch*1000,
                             device=a.device, fixed_train_mask=a.fixed_train_mask)
        validation = run_epoch(model, val_dl, normalizer, mask_ratio=a.mask_ratio,
                               seed=a.seed, device=a.device)
        record = {'epoch': epoch, **{f'train_{k}':v for k,v in training.items()},
                  **{f'validation_{k}':v for k,v in validation.items()}}
        history.append(record)
        print(json.dumps(record))
        if validation['normalized_masked_MSE'] < best:
            best = validation['normalized_masked_MSE']
            torch.save({'architecture_version': 'token_conditioned',
                        'state_dict': model.state_dict(), 'model_config': model_config,
                        'normalizer': normalizer.__dict__, 'training_config': config,
                        'best_epoch': epoch, 'validation_metrics': validation},
                       run_dir/'best.pt')
        with (run_dir/'history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
        (run_dir/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        save_loss_curve_svg(history, run_dir/'loss_curve.svg')
        # last.pt stores the complete restartable state, unlike best.pt which
        # is intentionally a lightweight inference/selection checkpoint.
        torch.save({'architecture_version': 'token_conditioned',
                    'state_dict': model.state_dict(), 'model_config': model_config,
                    'normalizer': normalizer.__dict__, 'training_config': config,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch, 'best_validation_mse': best,
                    'torch_rng_state': torch.get_rng_state(),
                    'train_loader_rng_state': train_dl.generator.get_state(),
                    'validation_loader_rng_state': val_dl.generator.get_state()}, run_dir/'last.tmp.pt')
        (run_dir/'last.tmp.pt').replace(run_dir/'last.pt')
    print(f'Best validation masked MSE={best:.6g}, checkpoint={run_dir/"best.pt"}')
    return history


if __name__ == '__main__':
    train_main()
