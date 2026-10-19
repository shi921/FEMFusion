"""Edge-aware Mesh MPNN + region/global pooling, masked-field decoder."""
from __future__ import annotations
import hashlib
import torch
from torch import nn
import torch.nn.functional as F
from femfusion.common.contracts import EncoderOutput
from .dataset import FEMBatch


def mask_nodes(batch: FEMBatch, ratio: float = .25, seed: int = 42,
               deterministic: bool = False) -> torch.Tensor:
    if not 0 < ratio < 1:
        raise ValueError('mask_ratio must be between 0 and 1')
    out = torch.zeros(batch.targets.size(0), dtype=torch.bool, device=batch.targets.device)
    for i, cid in enumerate(batch.sample_ids):
        a, b = int(batch.ptr[i]), int(batch.ptr[i+1])
        if b - a < 2:
            raise ValueError(f'{cid}: need >=2 nodes for masked reconstruction')
        g = torch.Generator(device='cpu')
        offset = int(hashlib.sha256(cid.encode()).hexdigest()[:12], 16) if deterministic else i
        g.manual_seed((seed + offset) % (2**63 - 1))
        k = min(b-a-1, max(1, round((b-a)*ratio)))
        idx = torch.randperm(b-a, generator=g)[:k].to(out.device)
        out[a+idx] = True
    return out


def build_model_inputs(batch: FEMBatch, hidden: torch.Tensor) -> torch.Tensor:
    """Only geometry + visible channels + visibility mask.

    Crucially NEVER consumes graph.x, per-region stress stats, nor the hidden
    targets; changing targets on hidden nodes cannot change any input entry.
    Position units are fixed 0.1 metres, retaining metric scale (not bbox unit).
    """
    if hidden.shape != (len(batch.targets),) or hidden.dtype != torch.bool:
        raise ValueError('hidden mask must be bool [total_nodes]')
    counts = torch.bincount(batch.batch_index, minlength=len(batch.sample_ids)).clamp_min(1)
    center = batch.pos.new_zeros((len(batch.sample_ids), 3))
    center.index_add_(0, batch.batch_index, batch.pos)
    center = center / counts[:, None]
    geometry = (batch.pos - center[batch.batch_index]) / 0.1
    # where (not multiply) prevents NaN and removes the original target before MPNN.
    visible = torch.where(hidden[:, None], torch.zeros_like(batch.targets), batch.targets)
    return torch.cat([geometry, visible, (~hidden).float()[:, None]], dim=-1)


def _mean_pool(features: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    summed = features.new_zeros((size, features.shape[-1]))
    summed.index_add_(0, index, features)
    counts = features.new_zeros(size)
    counts.index_add_(0, index, torch.ones(len(index), device=features.device, dtype=features.dtype))
    return summed / counts.clamp_min(1)[:, None]


class EdgeMPNNLayer(nn.Module):
    """Learned edge-conditioned messages with degree-normalized sum."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.src = nn.Linear(hidden_dim, hidden_dim)
        self.edge = nn.Linear(4, hidden_dim)
        self.update = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_attr, dropout):
        src, dst = edge_index
        # Edge rel-coordinate and distance in fixed 0.1m units, not unit-length normalization.
        m = F.relu(self.src(h[src]) + self.edge(edge_attr / .1))
        summed = h.new_zeros(h.shape)
        summed.index_add_(0, dst, m)
        degree = h.new_zeros(h.size(0))
        degree.index_add_(0, dst, torch.ones(len(dst), dtype=h.dtype, device=h.device))
        agg = summed / degree.clamp_min(1).unsqueeze(1)
        residual = self.update(torch.cat([h, agg], dim=1))
        return self.norm(h + F.dropout(residual, p=dropout, training=self.training))


class FEMEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, output_dim: int = 128,
                 num_layers: int = 3, dropout: float = .1):
        super().__init__()
        if output_dim != 128:
            raise ValueError('EncoderOutput embedding_dim must be 128')
        if not 2 <= num_layers <= 4:
            raise ValueError('FEM MPNN requires 2-4 layers')
        self.embedding = nn.Sequential(nn.Linear(8, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(EdgeMPNNLayer(hidden_dim) for _ in range(num_layers))
        self.region_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, output_dim))
        self.global_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, output_dim))
        # Projected physics tokens are used by reconstruction, not only returned.
        # Thus gradients from masked reconstruction train BOTH token heads.
        # These features are computed exclusively from target-masked node states;
        # per-region ground-truth statistics in aux are never read by the model.
        self.global_context = nn.Linear(output_dim, hidden_dim)
        self.region_context = nn.Linear(output_dim, hidden_dim)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, 4))
        self.dropout = float(dropout)

    def forward(self, batch: FEMBatch, hidden: torch.Tensor) -> tuple[EncoderOutput, torch.Tensor]:
        h = self.embedding(build_model_inputs(batch, hidden))
        for layer in self.layers:
            h = layer(h, batch.edge_index, batch.edge_attr, self.dropout)
        b = len(batch.sample_ids)
        pooled = _mean_pool(h, batch.batch_index, b)
        global_token = self.global_head(pooled)
        region_total = int(batch.region_ptr[-1].item())
        region_features = self.region_head(_mean_pool(h, batch.region_index, region_total))
        # Each node sees its graph-level and geometry-region-level context.
        # IMPORTANT: both contexts depend on masked input h, never raw hidden
        # fields or the analysis-only region statistics. The decoder therefore
        # provides a real supervised gradient path through both token heads.
        graph_context = self.global_context(global_token)[batch.batch_index]
        local_context = self.region_context(region_features)[batch.region_index]
        predicted = self.decoder(torch.cat((h, graph_context, local_context), dim=-1))
        max_regions = max(len(ids) for ids in batch.region_ids)
        local_tokens = h.new_zeros((b, max_regions, 128))
        valid = torch.zeros((b, max_regions), dtype=torch.bool, device=h.device)
        for i, ids in enumerate(batch.region_ids):
            a = int(batch.region_ptr[i]); end = int(batch.region_ptr[i+1])
            local_tokens[i, :len(ids)] = region_features[a:end]
            valid[i, :len(ids)] = True
        output = EncoderOutput(global_token=global_token, local_tokens=local_tokens,
                               local_mask=valid, sample_ids=list(batch.sample_ids),
                               local_entity_ids=[list(ids) for ids in batch.region_ids],
                               aux={
                                   'region_to_mesh_node_ids': [dict(zip(ids, nodes)) for ids, nodes in
                                                               zip(batch.region_ids, batch.node_ids)],
                                   'region_centroids_m': [v.detach().cpu().tolist() for v in batch.region_centroids],
                                   # Analysis/provenance only; NEVER used as encoder inputs.
                                   'region_stats_not_model_inputs': batch.region_stats,
                                   'source_files': batch.sources,
                                   'physical_load_N': batch.physical_load_N,
                                   'reference_load_N': batch.reference_load_N,
                                   'load_scale': [batch.reference_load_N/load for load in batch.physical_load_N],
                                   'original_node_count': batch.original_node_counts,
                                   'original_triangle_count': batch.original_triangle_counts,
                                   'negative_stress_nodes_in_raw_field': batch.negative_stress_nodes,
                               })
        return output, predicted


def reconstruction_loss(predicted: torch.Tensor, targets: torch.Tensor,
                        hidden: torch.Tensor) -> torch.Tensor:
    if hidden.dtype != torch.bool or not hidden.any():
        raise ValueError('At least one masked node required')
    # Four train-standardized targets receive equal MSE channel weight.
    return F.mse_loss(predicted[hidden], targets[hidden])


def physical_metrics(pred: torch.Tensor, target: torch.Tensor, hidden: torch.Tensor,
                     normalizer) -> dict:
    p = normalizer.inverse(pred[hidden].detach())
    t = normalizer.inverse(target[hidden].detach())
    err = (p - t).abs()
    return {'masked_nodes': int(hidden.sum()),
            'displacement_abs_sum_m': err[:, :3].sum().item(),
            'stress_abs_sum_Pa': err[:, 3].sum().item(),
            'masked_displacement_MAE_mm': err[:, :3].mean().item()*1000.,
            'masked_stress_MAE_MPa': err[:, 3].mean().item()/1e6}
