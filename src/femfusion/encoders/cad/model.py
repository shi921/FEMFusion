"""Native B-rep Face MPNN with edge-aware GINE and padded 128-d tokens."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .graph_builder import FACE_FEATURES, EDGE_FEATURES, SURFACES, CURVES


def _pyg():
    try:
        from torch_geometric.nn import GINEConv, global_mean_pool
    except ImportError as exc:
        raise ImportError('CAD Encoder needs torch-geometric GINEConv; no ignored-edge fallback') from exc
    return GINEConv, global_mean_pool


def _metadata(data, counts):
    b = len(counts)
    ids = data.case_id
    if isinstance(ids, str):
        ids = [ids]
    ids = list(ids)
    if len(ids) != b:
        raise ValueError(f'case_id metadata count={len(ids)} for graphs={b}')
    face_ids = data.face_ids
    if b == 1 and len(face_ids) == counts[0] and all(isinstance(s, str) for s in face_ids):
        face_ids = [face_ids]
    elif (len(face_ids) == sum(counts) and all(isinstance(s, str) for s in face_ids)):
        # Accommodate loaders that concatenate all string lists.
        offsets = [0]
        for n in counts:
            offsets.append(offsets[-1] + n)
        face_ids = [face_ids[offsets[i]:offsets[i+1]] for i in range(b)]
    face_ids = [list(row) for row in face_ids]
    if len(face_ids) != b or any(len(row) != n for row, n in zip(face_ids, counts)):
        raise ValueError(f'Face ID alignment lost in DataLoader: counts={counts}, face_ids={face_ids}')
    return ids, face_ids


class CADEncoder(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=128, layers=3, dropout=0.1):
        super().__init__()
        if output_dim != 128:
            raise ValueError('Shared CAD EncoderOutput requires exactly 128 dimensions')
        if not 2 <= layers <= 4:
            raise ValueError('GNN depth must be 2–4 layers')
        GINEConv, _ = _pyg()
        self.face_type = nn.Embedding(len(SURFACES), hidden_dim)
        self.edge_type = nn.Embedding(len(CURVES), hidden_dim)
        self.node_mlp = nn.Sequential(nn.Linear(len(FACE_FEATURES) + 1, hidden_dim),
                                      nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.edge_mlp = nn.Linear(len(EDGE_FEATURES), hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(GINEConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                     nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
                                        edge_dim=hidden_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = dropout
        self.local_project = nn.Linear(hidden_dim, output_dim)
        self.global_project = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                            nn.Linear(hidden_dim, output_dim))

    def forward(self, data, face_mask=None):
        """Mask entire Face continuous+categorical features before message passing.

        No target attributes are appended to the masked input. `data.volume_log`
        is never read here and therefore cannot leak the global target.
        """
        _, pool = _pyg()
        x = data.x
        if x.ndim != 2 or x.shape[1] != len(FACE_FEATURES):
            raise ValueError(f'Unexpected Face feature shape: {tuple(x.shape)}')
        if face_mask is None:
            face_mask = torch.zeros(x.size(0), device=x.device, dtype=torch.bool)
        if face_mask.shape != (x.shape[0],):
            raise ValueError('face_mask must be [total_nodes]')
        safe = x.masked_fill(face_mask[:, None], 0.)
        types = data.face_type.masked_fill(face_mask, 0)
        h = self.node_mlp(torch.cat([safe, face_mask[:, None].float()], dim=-1)) + self.face_type(types)
        edge = self.edge_mlp(data.edge_attr) + self.edge_type(data.edge_type)
        for conv, norm in zip(self.convs, self.norms):
            residual = conv(h, data.edge_index, edge)
            h = norm(h + F.dropout(F.relu(residual), p=self.dropout, training=self.training))
        batch_index = getattr(data, 'batch', None)
        if batch_index is None:
            batch_index = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
        counts = torch.bincount(batch_index, minlength=int(batch_index.max().item())+1).tolist()
        sample_ids, entity_ids = _metadata(data, counts)
        graph_h = pool(h, batch_index, size=len(counts))
        global_tokens = self.global_project(graph_h)
        face_h = self.local_project(h)
        padded = face_h.new_zeros((len(counts), max(counts), face_h.shape[1]))
        valid = torch.zeros((len(counts), max(counts)), device=h.device, dtype=torch.bool)
        start = 0
        for i, n in enumerate(counts):
            padded[i, :n] = face_h[start:start+n]
            valid[i, :n] = True
            start += n
        # Exact five protocol fields; no change to common/contracts.py.
        from femfusion.common.contracts import EncoderOutput
        return EncoderOutput(global_token=global_tokens, local_tokens=padded,
                             local_mask=valid, sample_ids=sample_ids,
                             local_entity_ids=entity_ids)


class MaskedCADTask(nn.Module):
    """Masked Face numerical reconstruction + volume-log prediction from readout."""
    def __init__(self, hidden_dim=128, output_dim=128, layers=3, dropout=0.1):
        super().__init__()
        self.encoder = CADEncoder(hidden_dim, output_dim, layers, dropout)
        self.face_head = nn.Sequential(nn.Linear(output_dim, output_dim), nn.ReLU(),
                                       nn.Linear(output_dim, len(FACE_FEATURES)))
        self.volume_head = nn.Linear(output_dim, 1)

    def forward(self, data, face_mask=None):
        tokens = self.encoder(data, face_mask)
        face_tokens = tokens.local_tokens[tokens.local_mask]
        if face_mask is not None:
            face_tokens = face_tokens[face_mask]
        face_prediction = self.face_head(face_tokens)
        volume_prediction = self.volume_head(tokens.global_token).reshape(-1)
        return tokens, face_prediction, volume_prediction
