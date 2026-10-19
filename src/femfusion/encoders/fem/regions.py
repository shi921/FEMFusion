"""Geometry-only, deterministic spatial Region partition; stress is *not* a key."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from .mesh_builder import SurfaceGraph


@dataclass
class Regions:
    index: torch.Tensor                   # [sampled N], compact 0..R-1
    ids: list[str]                        # stable geometric cell IDs
    centroids: torch.Tensor               # [R,3], metres
    node_ids: list[list[int]]             # original NPZ mesh node indices
    stats: list[dict]                     # exported analysis only; NEVER model input


def partition_regions(graph: SurfaceGraph, max_regions: int = 32) -> Regions:
    if max_regions < 1:
        raise ValueError('max_regions must be >=1')
    pos = graph.pos.detach().cpu().numpy()
    # nside^3 <= cap by construction: no target-dependent merging or truncation.
    nside = max(1, int(np.floor(np.cbrt(max_regions) + 1e-8)))
    lo, span = pos.min(axis=0), np.maximum(np.ptp(pos, axis=0), 1e-12)
    cell = np.minimum((nside * ((pos - lo) / span)).astype(np.int64), nside - 1)
    key = cell[:, 0] * nside * nside + cell[:, 1] * nside + cell[:, 2]
    unique, inv = np.unique(key, return_inverse=True)
    ids, centroids, nodes, stats = [], [], [], []
    orig = graph.original_node_ids.numpy()
    for j, c in enumerate(unique):
        selected = np.flatnonzero(inv == j)
        ix, iy, iz = int(c // (nside*nside)), int(c // nside % nside), int(c % nside)
        ids.append(f'grid_{nside}_{ix}_{iy}_{iz}')
        centroids.append(pos[selected].mean(axis=0))
        nodes.append([int(i) for i in orig[selected]])
        # Export-time statistics, not passed to the encoder in any form.
        disp = graph.x[selected, 3:6].numpy()
        stress = graph.x[selected, 6].numpy()
        stats.append({'num_nodes': len(selected), 'stress_mean_Pa': float(stress.mean()),
                      'stress_max_Pa': float(stress.max()),
                      'displacement_magnitude_mean_m': float(np.linalg.norm(disp, axis=1).mean())})
    if len(inv) != len(pos) or len(set(i for group in nodes for i in group)) != len(pos):
        raise RuntimeError('Regions failed to cover all retained mesh nodes')
    return Regions(torch.from_numpy(inv.copy()).long(), ids,
                   torch.from_numpy(np.asarray(centroids)).float(), nodes, stats)
