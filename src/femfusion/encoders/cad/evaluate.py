"""Never used for model selection; test metrics + deterministic geometry embeddings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import CADDataset, loader
from .model import MaskedCADTask
from .train import epoch, select_face_mask, set_seed


def evaluate(checkpoint='outputs/cad/training/best.pt', output_dir='outputs/cad/evaluation', split='test'):
    if split != 'test':
        raise ValueError('Evaluation CLI is deliberately restricted to held-out test split')
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = saved['config']
    set_seed(int(cfg['seed']))
    ds = CADDataset('test', cfg['cache_dir'])
    if ds.normalizer != saved['normalizer']:
        raise ValueError('Normalizer changed after training; refuse test evaluation')
    device = torch.device(cfg.get('device', 'cpu'))
    task = MaskedCADTask(**cfg['model']).to(device)
    task.load_state_dict(saved['model'], strict=True)
    task.eval()
    stream = loader(ds, cfg['batch_size'], shuffle=False, seed=cfg['seed'])
    metrics = epoch(task, stream, device, cfg['mask_ratio'], cfg['volume_weight'],
                    int(cfg['seed']) + 100000, optimizer=None)
    # Separate unmasked pass creates deployment embeddings; masked targets never leak.
    graph_tokens, flat_faces, mapping, offsets, row_ids, volume_true, volume_pred = [], [], [], [0], [], [], []
    with torch.no_grad():
        for batch in stream:
            batch = batch.to(device)
            tokens, _, vol = task(batch, face_mask=None)
            for i, case_id in enumerate(tokens.sample_ids):
                mask = tokens.local_mask[i]
                face = tokens.local_tokens[i][mask].detach().cpu().numpy()
                face_ids = list(tokens.local_entity_ids[i])
                if len(face) != len(face_ids):
                    raise ValueError(f'Face ID/token mismatch for {case_id}')
                graph_tokens.append(tokens.global_token[i].detach().cpu().numpy())
                flat_faces.append(face)
                offsets.append(offsets[-1] + len(face_ids))
                row_ids.append(case_id)
                volume_true.append(float(batch.volume_log[i].cpu()))
                volume_pred.append(float(vol[i].cpu()))
                source_indices = batch.face_source_indices[batch.batch == i].cpu().tolist()
                mapping.append({'case_id': case_id, 'source': batch.source[i],
                                'face_entities': [{'entity_id': eid,
                                                   'source_face_index': int(index),
                                                   'flat_embedding_row': offsets[-2] + j}
                                                  for j,(eid,index) in enumerate(zip(face_ids, source_indices))]})
    vol_norm = ds.normalizer['volume']
    truth = np.expm1(np.asarray(volume_true) * vol_norm['std'] + vol_norm['mean'])
    pred = np.expm1(np.asarray(volume_pred) * vol_norm['std'] + vol_norm['mean'])
    metrics['volume_mae_original_units'] = float(np.abs(truth-pred).mean())
    metrics['volume_rmse_original_units'] = float(np.sqrt(np.mean((truth-pred)**2)))
    metrics['n_graphs'] = len(row_ids)
    metrics['n_faces'] = offsets[-1]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'embeddings.npz',
                        case_ids=np.asarray(row_ids, dtype=str),
                        global_token=np.stack(graph_tokens).astype(np.float32),
                        local_tokens=np.concatenate(flat_faces, axis=0).astype(np.float32),
                        local_offsets=np.asarray(offsets, dtype=np.int64))
    (output/'face_mapping.json').write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f'Embedding files in {output}')
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='outputs/cad/training/best.pt')
    parser.add_argument('--output-dir', default='outputs/cad/evaluation')
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output_dir)


if __name__ == '__main__':
    main()
