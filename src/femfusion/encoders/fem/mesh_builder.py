"""Bidirected, deduplicated surface-triangle graph with physical edge lengths."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from .field_parser import SurfaceFields


@dataclass
class SurfaceGraph:
    pos: torch.Tensor            # [N,3] metres
    edge_index: torch.Tensor     # [2,E] directed, contains reverse for each edge
    edge_attr: torch.Tensor      # [E,4] relative position (m) + distance (m)
    x: torch.Tensor              # [N,7] position (m), displacement (m), stress (Pa) -- NEVER input directly
    original_node_ids: torch.Tensor


def triangle_edges(points: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique undirected triangle edges, returned bidirectionally; no self loops."""
    triangles = np.asarray(triangles)
    points = np.asarray(points)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in 'iu':
        raise ValueError('triangles must be [F,3] integer indices')
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must be [N,3]')
    if not np.isfinite(points).all():
        raise ValueError('nonfinite mesh positions')
    if len(triangles) == 0 or np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise ValueError('invalid triangle node index')
    pairs = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    if (pairs[:, 0] == pairs[:, 1]).any():
        raise ValueError('degenerate triangle/self edge')
    unique = np.unique(np.sort(pairs, axis=1), axis=0)
    directed = np.concatenate((unique, unique[:, ::-1]), axis=0)
    directed = directed[np.lexsort((directed[:, 1], directed[:, 0]))]
    delta = points[directed[:, 1]] - points[directed[:, 0]]
    dist = np.linalg.norm(delta, axis=1)
    if np.any(dist <= 0):
        raise ValueError('zero-length mesh edge')
    return directed.T.astype(np.int64), np.concatenate([delta, dist[:, None]], axis=1).astype(np.float32)


def build_graph(fields: SurfaceFields, max_nodes: int | None = None, seed: int = 42) -> SurfaceGraph:
    """Build a surface graph; optional deterministic spatial sampling of nodes.

    If max_nodes is used the remaining graph is induced (some edges disappear).
    original_node_ids always permits tracing back to the unmodified NPZ mesh.
    """
    pos, tri = fields.points, fields.triangles
    original = np.arange(len(pos), dtype=np.int64)
    if max_nodes is not None and len(pos) > max_nodes:
        if max_nodes < 3:
            raise ValueError('max_nodes must be >=3')
        # Spatial coverage, deterministic even if mesh node ordering changes.
        rng = np.random.default_rng(seed)
        span = np.maximum(np.ptp(pos, axis=0), 1e-12)
        normalized = (pos - pos.min(axis=0)) / span
        # Sort by 3d voxel key; random within voxel with reproducible seed.
        bins = np.floor(np.minimum(normalized * 32, 31)).astype(np.int64)
        keys = bins[:, 0] * 1024 + bins[:, 1] * 32 + bins[:, 2]
        order = np.lexsort((rng.random(len(pos)), keys))
        chosen = np.sort(order[np.linspace(0, len(pos)-1, max_nodes).astype(int)])
        original = original[chosen]
        old_to_new = np.full(len(pos), -1, dtype=np.int64)
        old_to_new[chosen] = np.arange(len(chosen))
        mapped = old_to_new[tri]
        tri = mapped[(mapped >= 0).all(axis=1)]
        pos = pos[chosen]
        if len(tri) == 0:
            raise ValueError('node sampling removed every triangle; raise max_nodes')
    edge_index, edge_attr = triangle_edges(pos, tri)
    x = np.column_stack((pos, fields.displacement[original], fields.stress[original]))
    return SurfaceGraph(torch.from_numpy(pos.copy()).float(), torch.from_numpy(edge_index).long(),
                        torch.from_numpy(edge_attr).float(), torch.from_numpy(x.copy()).float(),
                        torch.from_numpy(original.copy()).long())
