"""Official CSV-to-ZIP joins, train-only normalization and variable-size batching."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .field_parser import read_fields, CASE_RE, validate_load
from .mesh_builder import SurfaceGraph, build_graph
from .regions import Regions, partition_regions

SPLITS = ('train', 'validation', 'test')


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    split: str
    physical_load_N: float
    archive: Path
    member: str
    crc: int
    summary_member: str


def case_records(data_root: str | Path, split: str) -> list[CaseRecord]:
    if split not in SPLITS:
        raise ValueError(f'Unknown official split {split!r}; expected {SPLITS}')
    root = Path(data_root)
    csv_file = root / f'{split}.csv'
    if not csv_file.exists():
        raise FileNotFoundError(f'Missing official {csv_file}')
    with csv_file.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not {'case_id', 'split', 'physical_load_N'}.issubset(reader.fieldnames or ()):
            raise ValueError(f'{csv_file} requires case_id,split,physical_load_N (unit N)')
        rows = list(reader)
    if not rows:
        raise ValueError(f'No rows in {csv_file}')
    seen, wanted = set(), {}
    for row in rows:
        case_id = row['case_id'].strip()
        if not CASE_RE.fullmatch(case_id):
            raise ValueError(f'Invalid case_id {case_id!r} in {csv_file}')
        if case_id in seen:
            raise ValueError(f'Duplicate case_id in CSV {csv_file}: {case_id}')
        seen.add(case_id)
        if row['split'] != split:
            raise ValueError(f'{case_id}: split={row["split"]!r}, expected {split!r}')
        wanted[case_id] = validate_load(row['physical_load_N'])
    archives = sorted(root.glob(f'fields-{split}-*.zip'))
    if not archives:
        raise FileNotFoundError(f'No fields-{split}-*.zip in {root}')
    locations = {}
    for archive in archives:
        with ZipFile(archive) as zf:
            for info in zf.infolist():
                name = info.filename
                if not name.endswith('/fields.npz'):
                    continue
                parts = name.split('/')
                if len(parts) != 3 or parts[0] != 'fem' or not CASE_RE.fullmatch(parts[1]):
                    raise ValueError(f'{archive}: unsafe/unexpected FEM ZIP path {name!r}')
                cid = parts[1]
                if cid in locations:
                    raise ValueError(f'Duplicate fields for case_id={cid}: {locations[cid][0]} and {archive}')
                if cid in wanted:
                    locations[cid] = (archive, name, info.CRC)
    missing = sorted(set(wanted) - set(locations))
    if missing:
        raise FileNotFoundError(f'Missing {split} field members by case_id ({len(missing)}): {missing[:10]}')
    # CSV order determines iteration only; ZIP members are always matched by case_id.
    return [CaseRecord(cid, split, wanted[cid], locations[cid][0], locations[cid][1],
                       locations[cid][2], f'fem/{cid}/summary.json') for cid in wanted]


@dataclass
class FieldNormalizer:
    """Column order: [ux,uy,uz] in mm, von Mises stress in MPa at reference load."""
    mean: list[float]
    std: list[float]
    reference_load_N: float = 1000.
    fitted_split: str = 'train'

    def __post_init__(self):
        if len(self.mean) != 4 or len(self.std) != 4 or min(self.std) <= 0:
            raise ValueError('FieldNormalizer expects 4 finite mean/std entries')
        if not np.isfinite(self.mean + self.std).all():
            raise ValueError('Nonfinite normalizer')
        if self.fitted_split != 'train':
            raise ValueError('Normalizer must be fitted only on train')
        validate_load(self.reference_load_N, 'reference_load_N')

    def transform(self, si: torch.Tensor) -> torch.Tensor:
        """[m,m,m,Pa] -> standardized channels; preserves sign/directions."""
        scale = si.new_tensor([1000., 1000., 1000., 1e-6])
        return (si * scale - si.new_tensor(self.mean)) / si.new_tensor(self.std)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Standardized channels -> [m,m,m,Pa] at reference load."""
        return (z * z.new_tensor(self.std) + z.new_tensor(self.mean)) / z.new_tensor([1000.,1000.,1000.,1e-6])

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**self.__dict__, 'source_units': ['m','m','m','Pa'],
                                    'learning_units': ['mm','mm','mm','MPa']}, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path: str | Path) -> 'FieldNormalizer':
        obj = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(obj['mean'], obj['std'], obj['reference_load_N'], obj['fitted_split'])


def fit_normalizer(data_root: str | Path, reference_load_N: float = 1000.,
                   records: list[CaseRecord] | None = None) -> FieldNormalizer:
    """Weighted by TRAIN nodes, never validation/test, stable streaming moments."""
    records = records if records is not None else case_records(data_root, 'train')
    if any(r.split != 'train' for r in records):
        raise ValueError('Cannot fit normalizer on non-train cases')
    count, mean, m2 = 0, np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64)
    for rec in records:
        fields = read_fields(rec.case_id, rec.archive, rec.physical_load_N,
                             reference_load_N, member=rec.member)
        a = np.column_stack((fields.displacement.astype(np.float64) * 1000.,
                             fields.stress.astype(np.float64) / 1e6))
        n = len(a)
        avg = a.mean(axis=0)
        moment = ((a - avg)**2).sum(axis=0)
        delta, total = avg - mean, count + n
        mean += delta * n / total
        m2 += moment + delta**2 * count*n / total
        count = total
    std = np.sqrt(m2 / max(count, 1))
    std = np.maximum(std, 1e-6)
    return FieldNormalizer(mean.tolist(), std.tolist(), reference_load_N)


@dataclass
class FEMExample:
    case_id: str
    graph: SurfaceGraph
    regions: Regions
    targets: torch.Tensor   # normalized [N,4]
    physical_load_N: float
    reference_load_N: float
    source: str
    original_node_count: int
    original_triangle_count: int = 0
    negative_stress_nodes: int = 0
    original_fields_si: torch.Tensor | None = None  # raw unscaled [N,4], SI; never model input


class FEMDataset(Dataset):
    def __init__(self, data_root: str | Path, split: str, normalizer: FieldNormalizer,
                 cache_dir: str | Path | None = None, max_regions: int = 32,
                 max_nodes: int | None = 4096, seed: int = 42, limit_cases: int | None = None):
        self.records = case_records(data_root, split)
        if limit_cases is not None:
            if limit_cases < 1: raise ValueError('limit_cases must be >=1')
            self.records = self.records[:limit_cases]
        self.normalizer = normalizer
        self.max_regions, self.max_nodes, self.seed = max_regions, max_nodes, seed
        self.cache_dir = Path(cache_dir) / split if cache_dir is not None else None
        self._memory: OrderedDict[int, FEMExample] = OrderedDict()

    def __len__(self):
        return len(self.records)

    def _cache_path(self, rec: CaseRecord):
        info = [str(rec.archive.resolve()), rec.archive.stat().st_size,
                rec.archive.stat().st_mtime_ns, rec.member, rec.crc,
                rec.physical_load_N, self.normalizer.reference_load_N,
                self.max_regions, self.max_nodes, self.seed, 'surface_graph_cache']
        key = hashlib.sha256(json.dumps(info).encode()).hexdigest()[:16]
        return (self.cache_dir / f'{rec.case_id}-{key}.npz') if self.cache_dir else None

    def __getitem__(self, index: int) -> FEMExample:
        if index in self._memory:
            self._memory.move_to_end(index)
            return self._memory[index]
        rec = self.records[index]
        cache = self._cache_path(rec)
        if cache and cache.exists():
            try:
                with np.load(cache, allow_pickle=False) as d:
                    g = SurfaceGraph(*(torch.from_numpy(d[k].copy()) for k in
                                       ('pos','edge_index','edge_attr','x','original_node_ids')))
                    ids = d['region_ids'].tolist()
                    region_index = torch.from_numpy(d['region_index'].copy())
                    centroids = torch.from_numpy(d['centroids'].copy())
                    sizes = d['node_sizes'].tolist()
                    packed = d['node_ids'].tolist()
                    offsets = np.cumsum([0]+sizes)
                    nodes = [packed[offsets[i]:offsets[i+1]] for i in range(len(sizes))]
                    stats = json.loads(str(d['stats_json'].item()))
                    regions = Regions(region_index, ids, centroids, nodes, stats)
                    original_count = int(d['original_count'].item())
                    triangle_count = int(d['triangle_count'].item())
                    negative_stress_nodes = int(d['negative_stress_nodes'].item())
                    raw_fields = torch.from_numpy(d['raw_fields_si'].copy())
                    source = str(d['source'].item())
            except Exception as exc:
                raise ValueError(f'Corrupt FEM cache {cache}: {exc}') from exc
        else:
            fields = read_fields(rec.case_id, rec.archive, rec.physical_load_N,
                                 self.normalizer.reference_load_N, member=rec.member)
            g = build_graph(fields, max_nodes=self.max_nodes, seed=self.seed)
            regions = partition_regions(g, max_regions=self.max_regions)
            source, original_count = fields.source, fields.node_count
            triangle_count = fields.triangle_count
            negative_stress_nodes = fields.quality['negative_stress_nodes']
            original_ids = g.original_node_ids.numpy()
            raw_fields = torch.from_numpy(np.column_stack((fields.original_displacement[original_ids],
                                                            fields.original_stress[original_ids])).astype(np.float64))
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache.with_name(cache.stem + '.tmp.npz')
                np.savez_compressed(tmp, pos=g.pos.numpy(), edge_index=g.edge_index.numpy(),
                                    edge_attr=g.edge_attr.numpy(), x=g.x.numpy(),
                                    original_node_ids=g.original_node_ids.numpy(),
                                    region_index=regions.index.numpy(), region_ids=np.asarray(regions.ids),
                                    centroids=regions.centroids.numpy(),
                                    node_ids=np.asarray([n for group in regions.node_ids for n in group], dtype=np.int64),
                                    node_sizes=np.asarray([len(group) for group in regions.node_ids]),
                                    stats_json=np.asarray(json.dumps(regions.stats)),
                                    original_count=np.asarray(original_count), source=np.asarray(source),
                                    triangle_count=np.asarray(triangle_count),
                                    negative_stress_nodes=np.asarray(negative_stress_nodes),
                                    raw_fields_si=raw_fields.numpy())
                tmp.replace(cache)
        result = FEMExample(rec.case_id, g, regions,
                            self.normalizer.transform(g.x[:, 3:7]), rec.physical_load_N,
                            self.normalizer.reference_load_N, source, original_count,
                            triangle_count, negative_stress_nodes, raw_fields)
        self._memory[index] = result
        if len(self._memory) > 2:
            self._memory.popitem(last=False)
        return result


@dataclass
class FEMBatch:
    pos: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    targets: torch.Tensor
    batch_index: torch.Tensor
    region_index: torch.Tensor
    ptr: torch.Tensor
    region_ptr: torch.Tensor
    sample_ids: list[str]
    region_ids: list[list[str]]
    node_ids: list[list[list[int]]]
    region_stats: list[list[dict]]
    region_centroids: list[torch.Tensor]
    original_node_ids: list[torch.Tensor]
    physical_load_N: list[float]
    reference_load_N: float
    sources: list[str]
    original_node_counts: list[int]
    original_triangle_counts: list[int]
    negative_stress_nodes: list[int]

    def to(self, device: torch.device | str) -> 'FEMBatch':
        for key in ('pos','edge_index','edge_attr','targets','batch_index','region_index','ptr','region_ptr'):
            setattr(self, key, getattr(self, key).to(device))
        return self


def collate_fem(examples: list[FEMExample]) -> FEMBatch:
    if not examples:
        raise ValueError('Cannot collate empty FEM batch')
    n, r = 0, 0
    pos, edges, attrs, targets, bi, ri = [], [], [], [], [], []
    ptr, rptr = [0], [0]
    for i, e in enumerate(examples):
        g = e.graph
        pos.append(g.pos);edges.append(g.edge_index + n);attrs.append(g.edge_attr)
        targets.append(e.targets)
        bi.append(torch.full((len(g.pos),), i, dtype=torch.long))
        ri.append(e.regions.index + r)
        n += len(g.pos);r += len(e.regions.ids)
        ptr.append(n);rptr.append(r)
    return FEMBatch(torch.cat(pos), torch.cat(edges, dim=1), torch.cat(attrs),
                    torch.cat(targets), torch.cat(bi), torch.cat(ri),
                    torch.tensor(ptr), torch.tensor(rptr),
                    [e.case_id for e in examples], [e.regions.ids for e in examples],
                    [e.regions.node_ids for e in examples], [e.regions.stats for e in examples],
                    [e.regions.centroids for e in examples], [e.graph.original_node_ids for e in examples],
                    [e.physical_load_N for e in examples], examples[0].reference_load_N,
                    [e.source for e in examples], [e.original_node_count for e in examples],
                    [e.original_triangle_count for e in examples],
                    [e.negative_stress_nodes for e in examples])


def make_loader(dataset: FEMDataset, batch_size: int = 2, shuffle: bool = False,
                num_workers: int = 0, seed: int = 42) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_fem, generator=generator)


def prepare_main(argv=None):
    """CLI to fit a train-only normalizer, verify joins, and build disk caches."""
    import argparse
    p = argparse.ArgumentParser(description='Prepare official AgentFEM FEM surface graphs')
    p.add_argument('--data-root', default='data/raw/agentfem')
    p.add_argument('--output-dir', default='data/processed/fem')
    p.add_argument('--splits', nargs='+', choices=SPLITS, default=list(SPLITS))
    p.add_argument('--max-regions', type=int, default=32)
    p.add_argument('--max-nodes', type=int, default=4096)
    p.add_argument('--reference-load-N', type=float, default=1000.)
    p.add_argument('--limit-cases', type=int, default=None, help='smoke testing only')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args(argv)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = fit_normalizer(a.data_root, a.reference_load_N)
    stats.save(out / 'normalizer.json')
    report = {'normalizer': str(out/'normalizer.json'), 'splits': {},
              'reference_load_N': a.reference_load_N, 'sampled_max_nodes': a.max_nodes,
              'max_regions': a.max_regions}
    for split in a.splits:
        dataset = FEMDataset(a.data_root, split, stats, cache_dir=out/'cache',
                             max_regions=a.max_regions, max_nodes=a.max_nodes,
                             seed=a.seed, limit_cases=a.limit_cases)
        nodes = edges = regions = original_nodes = original_triangles = sampled_cases = bad_stress_nodes = 0
        for example in dataset:
            nodes += len(example.graph.pos); edges += example.graph.edge_index.shape[1]
            regions += len(example.regions.ids)
            original_nodes += example.original_node_count
            original_triangles += example.original_triangle_count
            bad_stress_nodes += example.negative_stress_nodes
            sampled_cases += int(len(example.graph.pos) < example.original_node_count)
        report['splits'][split] = {'cases_prepared': len(dataset), 'nodes_retained': nodes,
                                   'directed_edges': edges, 'regions': regions,
                                   'original_surface_nodes': original_nodes,
                                   'original_surface_triangles': original_triangles,
                                   'sampled_cases': sampled_cases,
                                   'nodes_dropped_by_sampling': original_nodes-nodes,
                                   'negative_stress_nodes_in_raw_field': bad_stress_nodes}
    (out / 'prepare_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    prepare_main()
