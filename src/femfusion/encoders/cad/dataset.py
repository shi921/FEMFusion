"""Official case_id split -> matched STEP -> cached Face graphs -> train-only stats."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from .brep_parser import CADParseError, match_case_id, parse_source, step_sources
from .graph_builder import build_face_graph, FACE_FEATURES, EDGE_FEATURES

SPLITS = ('train', 'validation', 'test')
# Flags are categorical/binary and should never be centered.
NODE_KEEP = (7, 13)
EDGE_KEEP = (1, 2)


def read_splits(raw_dir):
    root = Path(raw_dir)
    rows, seen = {}, {}
    for split in SPLITS:
        path = root / f'{split}.csv'
        if not path.is_file():
            raise FileNotFoundError(f'Missing official {split} CSV: {path}')
        with path.open(encoding='utf-8-sig', newline='') as fh:
            reader = csv.DictReader(fh)
            if 'case_id' not in (reader.fieldnames or []):
                raise ValueError(f'{path}: missing case_id column')
            cases = []
            for row in reader:
                case_id = row['case_id'].strip()
                if not case_id or case_id in seen:
                    raise ValueError(f'Duplicate/missing case_id {case_id!r}: {seen.get(case_id)} / {split}')
                seen[case_id] = split
                cases.append(case_id)
        rows[split] = cases
    return rows


def _fingerprint(source):
    path = Path(source['path'])
    stat = path.stat()
    part = [str(path.resolve()), stat.st_size, stat.st_mtime_ns, source['member']]
    if source['kind'] == 'zip':
        import zipfile
        with zipfile.ZipFile(path) as z:
            meta = z.getinfo(source['member'])
            part.extend((meta.CRC, meta.file_size))
    return hashlib.sha256(json.dumps(part).encode()).hexdigest()[:20]


def _graphs(cache_dir):
    return Path(cache_dir) / 'graphs'


def prepare(raw_dir='data/raw/agentfem', cache_dir='data/interim/cad', strict=False):
    """Use all provided official splits; never download and never resplit.

    Missing STEP is recorded (optional CAD); ambiguous or corrupt geometry raises
    by default unless strict=False, in which case errors are written to manifest.
    """
    splits = read_splits(raw_dir)
    sources = step_sources(raw_dir)
    target = Path(cache_dir)
    _graphs(target).mkdir(parents=True, exist_ok=True)
    records = []
    for split, case_ids in splits.items():
        for case_id in case_ids:
            rec = {'case_id': case_id, 'split': split, 'status': 'missing'}
            try:
                source = match_case_id(case_id, sources)
                rec['step_source'] = source
                fingerprint = _fingerprint(source)
                # Include case_id in path only after hashing it, avoiding traversal.
                name = hashlib.sha256(f'{case_id}:{fingerprint}:cad'.encode()).hexdigest()[:28] + '.pt'
                path = _graphs(target) / name
                if not path.exists():
                    brep = parse_source(source, target)
                    graph = build_face_graph(brep, case_id)
                    torch.save(graph, path)
                else:
                    # Never use graph data from the filename alone when repairing a bad cache.
                    graph = torch.load(path, map_location='cpu', weights_only=False)
                    if graph.case_id != case_id:
                        raise CADParseError(f'Cache mismatch: {path}')
                rec.update(status='ok', graph_path=str(path.resolve()),
                           faces=graph.num_nodes, directed_adjacencies=graph.edge_index.shape[1],
                           face_mapping=[{'entity_id': eid, 'source_index': int(index), 'signature': sig}
                                         for eid, index, sig in zip(graph.face_ids,
                                             graph.face_source_indices.tolist(), graph.face_signatures)])
            except ImportError:
                # Dependency failures are fatal, not optional missing geometry.
                raise
            except (CADParseError, FileNotFoundError, ValueError, OSError) as exc:
                rec.update(status='error', error=str(exc))
                if strict:
                    raise
            records.append(rec)
    manifest = {'version': 1, 'data_dir': str(Path(raw_dir).resolve()),
                'split_counts': {s: len(cases) for s, cases in splits.items()},
                'records': records}
    (target/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    if not any(r['status'] == 'ok' and r['split'] == 'train' for r in records):
        raise ValueError(f'No train STEP found: inspect {target}/manifest.json; ZIP path matching uses case_id')
    norm = fit_normalizer([torch.load(r['graph_path'], map_location='cpu', weights_only=False)
                           for r in records if r['status'] == 'ok' and r['split'] == 'train'])
    (target/'normalizer.json').write_text(json.dumps(norm, indent=2), encoding='utf-8')
    return manifest


def _stats(values, keep, feature_names):
    if not values:
        return {'mean': [0.] * len(feature_names), 'std': [1.] * len(feature_names)}
    vals = torch.cat(values, dim=0).double()
    mean = vals.mean(dim=0)
    std = vals.std(dim=0, unbiased=False).clamp_min(1e-6)
    for i in keep:
        mean[i] = 0.
        std[i] = 1.
    return {'mean': mean.tolist(), 'std': std.tolist()}


def fit_normalizer(graphs):
    if not graphs:
        raise ValueError('Cannot fit on zero train graphs')
    v = torch.cat([g.volume_log.reshape(-1) for g in graphs]).double()
    return {'version': 1, 'source_split': 'train',
            'face': _stats([g.x for g in graphs], NODE_KEEP, FACE_FEATURES),
            'edge': _stats([g.edge_attr for g in graphs if g.edge_attr.numel()], EDGE_KEEP, EDGE_FEATURES),
            'volume': {'mean': float(v.mean()), 'std': max(float(v.std(unbiased=False)), 1e-6)}}


def _apply(values, stats):
    mean = torch.as_tensor(stats['mean'], dtype=values.dtype, device=values.device)
    std = torch.as_tensor(stats['std'], dtype=values.dtype, device=values.device)
    return (values - mean) / std


class CADDataset(torch.utils.data.Dataset):
    def __init__(self, split, cache_dir='data/interim/cad'):
        if split not in SPLITS:
            raise ValueError(f'Invalid split: {split}')
        directory = Path(cache_dir)
        manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        self.normalizer = json.loads((directory/'normalizer.json').read_text(encoding='utf-8'))
        if self.normalizer['source_split'] != 'train':
            raise ValueError('Normalizer was not fit on Train')
        self.records = [r for r in manifest['records'] if r['split'] == split and r['status'] == 'ok']
        self.missing = [r for r in manifest['records'] if r['split'] == split and r['status'] != 'ok']
        if not self.records:
            raise ValueError(f'No STEP graphs available for {split}; missing={len(self.missing)}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        g = torch.load(self.records[index]['graph_path'], map_location='cpu', weights_only=False)
        g = g.clone()
        g.x = _apply(g.x, self.normalizer['face'])
        g.edge_attr = _apply(g.edge_attr, self.normalizer['edge'])
        g.volume_log = (g.volume_log - self.normalizer['volume']['mean']) / self.normalizer['volume']['std']
        return g


def loader(dataset, batch_size=4, shuffle=False, seed=42):
    try:
        from torch_geometric.loader import DataLoader
    except ImportError as exc:
        raise ImportError('Variable-Face batching requires torch-geometric') from exc
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def main():
    p = argparse.ArgumentParser(description='Prepare native STEP B-rep graphs (no downloads)')
    p.add_argument('--data-dir', default='data/raw/agentfem')
    p.add_argument('--cache-dir', default='data/interim/cad')
    p.add_argument('--strict', action='store_true')
    a = p.parse_args()
    manifest = prepare(a.data_dir, a.cache_dir, a.strict)
    for split in SPLITS:
        found = sum(r['status'] == 'ok' and r['split'] == split for r in manifest['records'])
        print(f'{split}: {found}/{manifest["split_counts"][split]} STEP graphs')
    print(f'Manifest: {a.cache_dir}/manifest.json')


if __name__ == '__main__':
    main()
