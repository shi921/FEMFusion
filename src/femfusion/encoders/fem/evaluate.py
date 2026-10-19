"""Strict official test-only evaluation + geometric region visualization export."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .dataset import FEMDataset, FieldNormalizer, make_loader
from .model import FEMEncoder, mask_nodes, reconstruction_loss, physical_metrics


def _rgb(region_id: str):
    h = hashlib.sha256(region_id.encode()).digest()
    return [int(x) for x in h[:3]]


def export_regions(batch, encoded, dest: Path, index: int):
    """PLY point cloud + JSON + NPZ tokens; maps every region to original node indices."""
    dest.mkdir(parents=True, exist_ok=True)
    cid = batch.sample_ids[index]
    a, b = int(batch.ptr[index]), int(batch.ptr[index+1])
    ra, rb = int(batch.region_ptr[index]), int(batch.region_ptr[index+1])
    regions = batch.region_index[a:b].cpu().numpy() - ra
    positions = batch.pos[a:b].cpu().numpy()
    orig = batch.original_node_ids[index].cpu().numpy()
    ply = dest/f'{cid}_regions.ply'
    with ply.open('w', encoding='utf-8') as f:
        f.write('ply\nformat ascii 1.0\ncomment surface nodes (metres), not volume tetrahedra\n')
        f.write(f'element vertex {len(positions)}\n')
        for name, kind in [('x','float'),('y','float'),('z','float'),('original_node_id','int'),
                           ('region_index','int'),('red','uchar'),('green','uchar'),('blue','uchar')]:
            f.write(f'property {kind} {name}\n')
        f.write('end_header\n')
        for xyz, oid, k in zip(positions, orig, regions):
            rgb = _rgb(batch.region_ids[index][int(k)])
            f.write(f'{xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g} {oid} {int(k)} {rgb[0]} {rgb[1]} {rgb[2]}\n')
    manifest = {'case_id': cid, 'source_file': batch.sources[index],
                'physical_load_N': batch.physical_load_N[index],
                'reference_load_N': batch.reference_load_N,
                'original_node_count': batch.original_node_counts[index],
                'original_surface_triangles': batch.original_triangle_counts[index],
                'negative_stress_nodes_in_raw_field': batch.negative_stress_nodes[index],
                'retained_surface_node_count': len(positions),
                'sampling_notice': 'Region mapping covers retained surface nodes; node IDs refer to original NPZ.',
                'regions': [{'id': rid, 'centroid_m': center.tolist(), 'mesh_node_ids': nodes,
                             'analysis_only_stats': stat}
                            for rid, center, nodes, stat in zip(batch.region_ids[index],
                                  batch.region_centroids[index], batch.node_ids[index], batch.region_stats[index])]}
    (dest/f'{cid}_region_map.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    import numpy as np
    np.savez_compressed(dest/f'{cid}_tokens.npz',
                        global_token=encoded.global_token[index].detach().cpu().numpy(),
                        local_tokens=encoded.local_tokens[index].detach().cpu().numpy(),
                        local_mask=encoded.local_mask[index].detach().cpu().numpy(),
                        region_ids=np.asarray(batch.region_ids[index]))
    return [str(ply), str(dest/f'{cid}_region_map.json'), str(dest/f'{cid}_tokens.npz')]


def evaluate_main(argv=None):
    p = argparse.ArgumentParser(description='Evaluate FEM Encoder on official held-out test split')
    p.add_argument('--data-root', default='data/raw/agentfem')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output-dir', default='outputs/fem/evaluation')
    p.add_argument('--cache-dir', default='data/processed/fem/cache')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--test-limit', type=int, default=None, help='smoke only; not full official test')
    p.add_argument('--visualize-cases', type=int, default=2)
    p.add_argument('--device', default='cpu')
    a = p.parse_args(argv)
    # Loading our own checkpoint only; weights_only=True blocks arbitrary pickle execution.
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture_version') != 'token_conditioned':
        raise ValueError('Legacy FEM checkpoint has untrained token projection heads; '
                         'retrain using token_conditioned before evaluation.')
    cfg = checkpoint['training_config']
    normalizer = FieldNormalizer(**checkpoint['normalizer'])
    model = FEMEncoder(**checkpoint['model_config']).to(a.device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    test = FEMDataset(a.data_root, 'test', normalizer, a.cache_dir,
                      max_regions=cfg['max_regions'], max_nodes=cfg['max_nodes'],
                      seed=cfg['seed'], limit_cases=a.test_limit)
    loader = make_loader(test, a.batch_size, shuffle=False, seed=cfg['seed'])
    dest = Path(a.output_dir); dest.mkdir(parents=True, exist_ok=True)
    samples, exported = [], []
    disp_sum = stress_sum = loss_sum = count = 0.
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(a.device)
            hidden = mask_nodes(batch, cfg['mask_ratio'], seed=cfg['seed'], deterministic=True)
            encoded, pred = model(batch, hidden)
            assert encoded.global_token.shape == (len(batch.sample_ids), 128)
            assert encoded.local_tokens.shape[:2] == encoded.local_mask.shape
            assert encoded.local_mask.dtype == torch.bool
            assert all(len(ids) == int(encoded.local_mask[i].sum()) for i,ids in enumerate(encoded.local_entity_ids))
            for i, cid in enumerate(batch.sample_ids):
                nmask = hidden[batch.ptr[i]:batch.ptr[i+1]]
                a0, b0 = int(batch.ptr[i]), int(batch.ptr[i+1])
                local_metrics = physical_metrics(pred[a0:b0], batch.targets[a0:b0], nmask, normalizer)
                loss = reconstruction_loss(pred[a0:b0], batch.targets[a0:b0], nmask).item()
                n = local_metrics['masked_nodes']
                count += n
                disp_sum += local_metrics['displacement_abs_sum_m']
                stress_sum += local_metrics['stress_abs_sum_Pa']
                loss_sum += loss*n
                samples.append({'case_id': cid, 'source_file': batch.sources[i],
                                'physical_load_N': batch.physical_load_N[i],
                                'reference_load_N': batch.reference_load_N,
                                'load_scale': batch.reference_load_N/batch.physical_load_N[i],
                                'retained_surface_nodes': b0-a0,
                                'original_surface_triangles': batch.original_triangle_counts[i],
                                'negative_stress_nodes_in_raw_field': batch.negative_stress_nodes[i],
                                'region_count': len(batch.region_ids[i]),
                                'normalized_masked_MSE': loss,
                                'masked_nodes': n,
                                'masked_displacement_MAE_mm': local_metrics['masked_displacement_MAE_mm'],
                                'masked_stress_MAE_MPa': local_metrics['masked_stress_MAE_MPa']})
                if len(exported)//3 < a.visualize_cases:
                    exported.extend(export_regions(batch, encoded, dest/'regions', i))
    report = {'split': 'test', 'official_full_test': a.test_limit is None,
              'benchmark_validity': ('smoke_checkpoint_NOT_a_full_train_benchmark' if
                                    cfg.get('train_limit') is not None or cfg.get('validation_limit') is not None
                                    else 'full_train_checkpoint_test_evaluation'),
              'training_case_limit': cfg.get('train_limit'),
              'validation_case_limit': cfg.get('validation_limit'),
              'cases_evaluated': len(samples), 'masked_nodes': int(count),
              'masked_displacement_MAE_mm': disp_sum / (3*count) * 1000.,
              'masked_stress_MAE_MPa': stress_sum/count/1e6,
              'normalized_masked_MSE': loss_sum/count,
              'reference_load_N': normalizer.reference_load_N,
              'mask_seed': cfg['seed'], 'mask_ratio': cfg['mask_ratio'],
              'checkpoint': str(a.checkpoint), 'visualizations': exported,
              'note': ('Masked reconstruction of previously computed surface FEM fields, not a PDE solver. '
                       'Maximum nodal Von Mises stress is mesh-dependent near singularities, and no '
                       'mesh convergence, contact, fatigue or structural-safety qualification is established.'),
              'samples': samples}
    (dest/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples','visualizations')}, indent=2))
    return report


if __name__ == '__main__':
    evaluate_main()
