"""Frozen CAD representation probe. No encoder fitting or test-data tuning.

Run via: python -m femfusion.encoders.cad.probe --checkpoint ...
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SPLITS = ('train', 'validation', 'test')
GEOMETRY_TARGETS = ('depth', 'radius', 'waist', 'bow')
DIAGNOSTIC_TARGETS = ('mass_kg', 'physical_load_N')
ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def read_official_targets(raw_dir, split, targets=GEOMETRY_TARGETS + DIAGNOSTIC_TARGETS):
    """Use case_id keys, never ZIP/CSV/loader row positions for alignment."""
    path = Path(raw_dir) / f'{split}.csv'
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'{path}: empty CSV')
    ids, target_rows = [], []
    for row in rows:
        cid = row['case_id'].strip()
        if not cid or cid in ids:
            raise ValueError(f'{path}: empty or duplicate case_id: {cid!r}')
        ids.append(cid)
        target_rows.append([float(row[key]) for key in targets])
    y = np.asarray(target_rows, dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError(f'{path}: nonfinite target')
    return ids, y


def load_archive(path, expected_ids):
    """Return global embeddings in official CSV case order; reject missing/extras."""
    with np.load(path, allow_pickle=False) as z:
        ids = [str(i) for i in z['case_ids'].tolist()]
        x = np.asarray(z['global_token'], dtype=np.float64)
        if 'local_offsets' in z.files:
            offsets = np.asarray(z['local_offsets'], dtype=np.int64)
            local = z['local_tokens']
            if (len(offsets) != len(ids)+1 or offsets[0] != 0 or
                    offsets[-1] != len(local) or np.any(np.diff(offsets) <= 0) or
                    local.ndim != 2 or local.shape[1] != 128 or not np.isfinite(local).all()):
                raise ValueError(f'{path}: invalid local-token offsets/features')
    if x.shape != (len(ids), 128) or not np.isfinite(x).all():
        raise ValueError(f'{path}: global_token is not finite [N, 128]')
    if len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError(f'{path}: case_id set mismatch, missing={set(expected_ids)-set(ids)}, '
                         f'extra={set(ids)-set(expected_ids)}')
    idx = {cid: i for i, cid in enumerate(ids)}
    return x[[idx[cid] for cid in expected_ids]]


def summarize_array(x):
    """Descriptive only: no outcome leakage into PCA. Requires N >= 2."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('need a finite 2D array with >=2 samples')
    centered = x - x.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    eigen = s**2
    if eigen.sum() < 1e-20:
        ratios = np.zeros(len(s), dtype=float)
        effective = 0.0
    else:
        ratios = eigen / eigen.sum()
        nonzero = ratios[ratios > 0]
        effective = float(np.exp(-(nonzero*np.log(nonzero)).sum()))
    norm = np.linalg.norm(x, axis=1)
    return {'n': int(len(x)), 'dimensions': int(x.shape[1]),
            'pc1_explained_variance': float(ratios[0]),
            'pc5_explained_variance': float(ratios[:5].sum()),
            'effective_rank': effective,
            'centered_total_variance': float((centered**2).sum()/(len(x)-1)),
            'embedding_norm_mean': float(norm.mean()),
            'embedding_norm_std': float(norm.std())}


def _score(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    mae = float(np.mean(np.abs(y-pred)))
    rmse = float(np.sqrt(np.mean((y-pred)**2)))
    denom = float(np.sum((y-y.mean())**2))
    r2 = float(1.0 - np.sum((y-pred)**2)/denom) if denom > 0 else None
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def fit_ridge_with_validation(train_x, val_x, test_x, train_y, val_y, test_y, alphas=ALPHAS):
    """Train-only X/Y scaling, validation-only alpha search, test one-time score."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_x)
    xt, xv, xs = (scaler.transform(a) for a in (train_x, val_x, test_x))
    y_mean, y_scale = float(train_y.mean()), float(train_y.std())
    if y_scale < 1e-15:
        raise ValueError('target constant on train; cannot fit informative ridge')
    best, best_alpha, best_val_mae = None, None, float('inf')
    for alpha in alphas:
        model = Ridge(alpha=float(alpha)).fit(xt, (train_y-y_mean)/y_scale)
        val_predictions = model.predict(xv)*y_scale+y_mean
        val_mae = float(np.mean(np.abs(val_y-val_predictions)))
        if val_mae < best_val_mae:
            best, best_alpha, best_val_mae = model, float(alpha), val_mae
    test_predictions = best.predict(xs)*y_scale+y_mean
    return {'alpha_selected_on_validation': best_alpha,
            'validation_mae': best_val_mae,
            'test': _score(test_y, test_predictions)}


def fit_pc1_with_validation(train_x, val_x, test_x, train_y, val_y, test_y):
    """PCA fitted only to train: explicitly test whether 1D suffices."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train_x)
    xt, xv, xs = (scaler.transform(a) for a in (train_x, val_x, test_x))
    pca = PCA(n_components=1).fit(xt)
    return fit_ridge_with_validation(pca.transform(xt), pca.transform(xv),
                                     pca.transform(xs), train_y, val_y, test_y)


def raw_graph_statistics(g, volume_stats):
    """Hand-crafted CAD-only controls, no CSV target columns are inputs.

    g.x/g.edge_attr are normalized by the train-only CAD normalizer.
    The CAD B-rep volume itself is an available geometry statistic, intentionally
    included as a strong baseline (the mass diagnostic will benefit from it).
    """
    from .graph_builder import SURFACES, CURVES
    node = g.x.detach().cpu().numpy().astype(np.float64)
    edges = g.edge_attr.detach().cpu().numpy().astype(np.float64)
    if len(edges) == 0:
        edges = np.zeros((1, 3), dtype=np.float64)
    face_types = g.face_type.detach().cpu().numpy()
    edge_types = g.edge_type.detach().cpu().numpy()
    summaries = []
    for a in (node, edges):
        summaries.extend((a.mean(axis=0), a.std(axis=0),
                          a.min(axis=0), a.max(axis=0)))
    summaries.extend((np.bincount(face_types, minlength=len(SURFACES))/len(node),
                      np.bincount(edge_types, minlength=len(CURVES))/max(1,len(edge_types))))
    log_v = float(g.volume_log.reshape(-1)[0].item()) * volume_stats['std'] + volume_stats['mean']
    summaries.append(np.array([np.log1p(len(node)), np.log1p(len(edge_types)), log_v]))
    return np.concatenate(summaries).astype(np.float64)


def export_from_checkpoint(checkpoint, cache_dir, split, output_dir, batch_size=4):
    """Frozen, unmasked inference; no training/reconstruction; output CSV-order ids."""
    import torch
    from .dataset import CADDataset, loader
    from .model import MaskedCADTask

    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    ds = CADDataset(split, cache_dir)
    if ckpt['normalizer'] != ds.normalizer:
        raise ValueError('Checkpoint and cached train-only normalizers differ')
    task = MaskedCADTask(**ckpt['config']['model']).cpu().eval()
    task.load_state_dict(ckpt['model'], strict=True)
    ids, globals_, stats = [], [], []
    with torch.inference_mode():
        for batch in loader(ds, batch_size=batch_size, shuffle=False, seed=42):
            tokens, _, _ = task(batch, face_mask=None)
            ids.extend(tokens.sample_ids)
            globals_.append(tokens.global_token.cpu().numpy())
        for i in range(len(ds)):
            stats.append(raw_graph_statistics(ds[i], ds.normalizer['volume']))
    expected_ids = [r['case_id'] for r in ds.records]
    if ids != expected_ids:
        raise ValueError(f'{split}: inference case order differs from manifest')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir/f'{split}_global.npz'
    np.savez_compressed(target, case_ids=np.asarray(ids, dtype=str),
                        global_token=np.concatenate(globals_, axis=0).astype(np.float32),
                        cad_statistics=np.asarray(stats, dtype=np.float32))
    return target


def raw_stats_from_archive(path, expected_ids):
    with np.load(path, allow_pickle=False) as z:
        if 'cad_statistics' not in z.files:
            return None
        ids = [str(i) for i in z['case_ids'].tolist()]
        stats = np.asarray(z['cad_statistics'], dtype=np.float64)
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError('CAD stats IDs not aligned')
    if stats.ndim != 2 or not np.isfinite(stats).all():
        raise ValueError('Invalid CAD stats')
    idx = {cid:i for i,cid in enumerate(ids)}
    return stats[[idx[cid] for cid in expected_ids]]


def make_report(archives, raw_dir, output_dir):
    """Read-only evaluation. The test targets are never used for fitting/tuning."""
    ids, labels, x, stats = {}, {}, {}, {}
    for split in SPLITS:
        ids[split], labels[split] = read_official_targets(raw_dir, split)
        x[split] = load_archive(archives[split], ids[split])
        stats[split] = raw_stats_from_archive(archives[split], ids[split])
    if len(set(ids['train']) | set(ids['validation']) | set(ids['test'])) != sum(map(len,ids.values())):
        raise ValueError('CSV splits overlap; data leakage')
    if any(s is None for s in stats.values()) and not all(s is None for s in stats.values()):
        raise ValueError('CAD statistics available only for some splits')
    methods = {'encoder_global_128': x}
    if all(s is not None for s in stats.values()):
        methods['cad_handcrafted_stats'] = stats
    result = {'protocol': {'train_fit_only': True, 'validation_alpha_selection': True,
                           'test_one_time_evaluation': True,
                           'targets_geometry': GEOMETRY_TARGETS,
                           'targets_diagnostics': DIAGNOSTIC_TARGETS,
                           'split_sizes': {s:len(ids[s]) for s in SPLITS},
                           'cad_statistics_include_brep_volume': True},
              'representations': {s:summarize_array(x[s]) for s in SPLITS}, 'probes': {}}
    for col, name in enumerate(GEOMETRY_TARGETS + DIAGNOSTIC_TARGETS):
        tr, va, te = (labels[s][:,col] for s in SPLITS)
        baseline = np.full_like(te, tr.mean(), dtype=float)
        row = {'train_mean_baseline': {'test': _score(te, baseline)},
               'encoder_pc1': fit_pc1_with_validation(x['train'], x['validation'], x['test'],tr,va,te)}
        for method, split_x in methods.items():
            row[method] = fit_ridge_with_validation(split_x['train'], split_x['validation'],
                                                     split_x['test'], tr,va,te)
        result['probes'][name] = row
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output/'probe_report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False),encoding='utf-8')
    lines = ['# CAD frozen geometry probe', '',
             'Train-only standardization + Ridge fitting; Validation-only alpha selection; untouched Test scoring.',
             'The mass and physical-load probes are diagnostics, **not** CAD geometry training labels.', '',
             '| Target | Baseline MAE | Global MAE | Global R² | PC1 MAE | CAD stats MAE |',
             '|---|---:|---:|---:|---:|---:|']
    for name, row in result['probes'].items():
        metrics = row['encoder_global_128']['test']
        geom = row.get('cad_handcrafted_stats')
        r2 = f'{metrics["r2"]:.3f}' if metrics['r2'] is not None else 'N/A (constant test target)'
        geom_mae = f'{geom["test"]["mae"]:.6g}' if geom else 'N/A'
        lines.append(f'| {name} | {row["train_mean_baseline"]["test"]["mae"]:.6g} | '
                     f'{metrics["mae"]:.6g} | {r2} | '
                     f'{row["encoder_pc1"]["test"]["mae"]:.6g} | {geom_mae} |')
    lines += ['', '## Global token diagnostics', '',
              '| Split | Effective rank | PC1 explained variance | PC1–5 |',
              '|---|---:|---:|---:|']
    for s, diag in result['representations'].items():
        lines.append(f'| {s} | {diag["effective_rank"]:.3f} | '
                     f'{diag["pc1_explained_variance"]:.2%} | {diag["pc5_explained_variance"]:.2%} |')
    lines += ['', 'Interpretation: 128 dimensions in the output contract do not imply 128 independent '
              'geometry directions. A negative test R² means worse than predicting the **test** mean; '
              'compare MAE against the train-mean baseline.',
              'This evaluation does not train the CAD encoder or alter its checkpoint.']
    (output/'probe_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='CAD frozen geometry probe with clean split protocol')
    parser.add_argument('--checkpoint', default='outputs/cad/training/best.pt')
    parser.add_argument('--cache-dir', default='data/interim/cad')
    parser.add_argument('--data-dir', default='data/raw/agentfem')
    parser.add_argument('--test-embeddings', default='outputs/cad/evaluation/embeddings.npz',
                        help='Optional existing held-out test export; used if file exists')
    parser.add_argument('--output-dir', default='outputs/cad/evaluation/probe')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    archives = {}
    for s in SPLITS:
        if s == 'test' and args.test_embeddings and Path(args.test_embeddings).is_file():
            # Re-export test when needed for fair CAD handcrafted baselines.
            # Existing exported 32x128 data is kept authoritative: if a fresh export is
            # also produced, require numerical agreement instead of silently replacing.
            check_path = Path(args.test_embeddings)
            fresh_path = export_from_checkpoint(args.checkpoint,args.cache_dir,s,out,args.batch_size)
            ids, _ = read_official_targets(args.data_dir,s)
            old = load_archive(check_path,ids)
            fresh = load_archive(fresh_path,ids)
            if not np.allclose(old,fresh,rtol=2e-4,atol=2e-5):
                raise ValueError('Re-exported Test Embeddings do not match existing test archive')
            archives[s] = fresh_path
        else:
            archives[s] = export_from_checkpoint(args.checkpoint,args.cache_dir,s,out,args.batch_size)
    result = make_report(archives,args.data_dir,out)
    print('CAD geometry probe report:',out/'probe_report.md')
    print('Global Token effective rank (test):',round(result['representations']['test']['effective_rank'],4))
    for name in GEOMETRY_TARGETS:
        print(f'{name}: R2 = {result["probes"][name]["encoder_global_128"]["test"]["r2"]:.4f}')


if __name__ == '__main__':
    main()
