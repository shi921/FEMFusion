"""B-rep Face-Adjacency Graph; a multigraph with one directed pair per shared edge."""
from __future__ import annotations

from itertools import combinations
import math

import torch

from .brep_parser import BRepInfo, CADParseError

# 0 is reserved for unknown, including types unseen at training time.
SURFACES = ('unknown', 'plane', 'cylinder', 'cone', 'sphere', 'torus',
            'bezier_surface', 'bspline_surface', 'surface_of_revolution',
            'surface_of_extrusion', 'offset_surface', 'other_surface')
CURVES = ('unknown', 'line', 'circle', 'ellipse', 'hyperbola', 'parabola',
          'bezier_curve', 'bspline_curve', 'offset_curve', 'other_curve')
SURFACE_VOCAB = {t: i for i, t in enumerate(SURFACES)}
CURVE_VOCAB = {t: i for i, t in enumerate(CURVES)}
FACE_FEATURES = ('log_area', 'center_x', 'center_y', 'center_z',
                 'normal_x', 'normal_y', 'normal_z', 'normal_valid',
                 'log_extent_x', 'log_extent_y', 'log_extent_z',
                 'mean_curvature_scaled', 'gaussian_curvature_scaled',
                 'curvature_valid', 'log_global_scale')
EDGE_FEATURES = ('log_edge_length', 'convexity', 'convexity_known')


def _pyg_data():
    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError('Face graph requires torch-geometric; install CAD optional dependencies. '
                          'No mesh or placeholder graph fallback.') from exc
    return Data


def _finite(value: float):
    return math.isfinite(float(value))


def build_face_graph(brep: BRepInfo, case_id: str):
    """One Face/node; one directed pair per shared non-degenerate Edge.

    Boundary/seam edges do not become self-loops. Non-manifold edges connect all
    incident face pairs. Unknown convexity is explicitly represented by mask=0.
    """
    Data = _pyg_data()
    if not case_id or not brep.faces:
        raise CADParseError('case_id and at least one B-rep face are required')
    xmin, ymin, zmin, xmax, ymax, zmax = brep.bbox
    lengths = [xmax-xmin, ymax-ymin, zmax-zmin]
    scale = max(max(lengths), 1e-9)
    midpoint = [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2]
    rows = []
    for face in brep.faces:
        extent = [face.bbox[3+i] - face.bbox[i] for i in range(3)]
        rows.append([
            math.log1p(face.area),
            *((face.center[i]-midpoint[i])/scale for i in range(3)),
            *face.normal, float(face.normal_valid),
            *(math.log1p(max(v, 0)) for v in extent),
            # Curvature has inverse-length (mean) and inverse-length² (Gaussian) units.
            face.mean_curvature*scale if face.curvature_valid else 0.,
            face.gaussian_curvature*scale*scale if face.curvature_valid else 0.,
            float(face.curvature_valid), math.log1p(scale),
        ])
    x = torch.tensor(rows, dtype=torch.float32)
    if x.shape[1] != len(FACE_FEATURES) or not bool(torch.isfinite(x).all()):
        raise CADParseError(f'{case_id}: invalid/nonfinite face attributes')
    edges, attr, kinds, edge_sources = [], [], [], []
    src_to_node = {f.source_index: idx for idx, f in enumerate(brep.faces)}
    for e in brep.edges:
        if e.degenerate:
            continue
        incident = sorted(set(src_to_node[f] for f in e.face_indices if f in src_to_node))
        for u, v in combinations(incident, 2):
            for a, b in ((u, v), (v, u)):
                edges.append((a, b))
                attr.append((math.log1p(e.length), 0.0, 0.0))
                kinds.append(CURVE_VOCAB.get(e.curve_type, 0))
                edge_sources.append(e.source_index)
    edge_index = (torch.tensor(edges, dtype=torch.long).T.contiguous() if edges
                  else torch.empty((2, 0), dtype=torch.long))
    edge_attr = (torch.tensor(attr, dtype=torch.float32) if attr
                 else torch.empty((0, len(EDGE_FEATURES)), dtype=torch.float32))
    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
             face_type=torch.tensor([SURFACE_VOCAB.get(f.surface_type, 0)
                                     for f in brep.faces], dtype=torch.long),
             edge_type=torch.tensor(kinds, dtype=torch.long),
             volume_log=torch.tensor([math.log1p(brep.volume)], dtype=torch.float32),
             case_id=str(case_id), face_ids=[f.face_id for f in brep.faces],
             face_signatures=[f.signature for f in brep.faces],
             face_source_indices=torch.tensor([f.source_index for f in brep.faces], dtype=torch.long),
             source=brep.source, edge_source_indices=torch.tensor(edge_sources, dtype=torch.long))
    return g
