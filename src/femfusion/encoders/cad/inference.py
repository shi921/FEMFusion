"""Predict unmasked CAD tokens for a single native STEP file (no CSV needed)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .brep_parser import parse_step
from .dataset import _apply
from .graph_builder import build_face_graph
from .model import MaskedCADTask


def infer(step_path: str, case_id: str, checkpoint: str, output: str):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    norm = saved['normalizer']
    model = MaskedCADTask(**saved['config']['model']).eval()
    model.load_state_dict(saved['model'], strict=True)
    parsed = parse_step(step_path)
    graph = build_face_graph(parsed, case_id)
    graph.x = _apply(graph.x, norm['face'])
    graph.edge_attr = _apply(graph.edge_attr, norm['edge'])
    # B-rep-derived volume is stored as a label and never used by CADEncoder.
    with torch.no_grad():
        tokens = model.encoder(graph)
    result = Path(output)
    result.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(result, global_token=tokens.global_token.cpu().numpy(),
                        local_tokens=tokens.local_tokens[0][tokens.local_mask[0]].cpu().numpy(),
                        face_ids=np.array(tokens.local_entity_ids[0], dtype=str))
    result.with_suffix('.json').write_text(json.dumps({
        'case_id': case_id, 'source': str(Path(step_path).resolve()),
        'faces': [{'entity_id': f.face_id, 'source_index': f.source_index, 'signature': f.signature}
                  for f in parsed.faces]
    }, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--step', required=True)
    p.add_argument('--case-id', required=True)
    p.add_argument('--checkpoint', default='outputs/cad/training/best.pt')
    p.add_argument('--output', default='outputs/cad/evaluation/inference.npz')
    a = p.parse_args()
    print(f'Saved native B-rep embeddings: {infer(a.step, a.case_id, a.checkpoint, a.output)}')


if __name__ == '__main__':
    main()
