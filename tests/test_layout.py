"""Lightweight source checks: no network, Torch, or training data required."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run  # noqa: E402


class SourceLayoutTests(unittest.TestCase):
    def test_project_name_and_source_package(self):
        self.assertTrue((ROOT / 'src/femfusion/common/contracts.py').is_file())
        self.assertFalse((ROOT / 'src/agentfem_mm').exists())
        self.assertIn('name = "femfusion"', (ROOT / 'pyproject.toml').read_text())

    def test_all_output_paths_use_outputs(self):
        paths = [
            'outputs/cad/training', 'outputs/cad/evaluation',
            'outputs/fem/training', 'outputs/fem/evaluation',
            'outputs/text/training', 'outputs/text/evaluation',
        ]
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue((ROOT / path / '.gitkeep').exists())
        self.assertFalse((ROOT / 'experiments').exists())
        self.assertFalse((ROOT / 'checkpoints').exists())

    def test_fem_training_uses_configured_path(self):
        config = {
            'data': {'root': 'data/raw/agentfem', 'cache_dir': 'data/processed/fem/cache',
                     'max_regions': 32, 'max_nodes': 4096, 'reference_load_N': 1000.0},
            'encoder': {'hidden_dim': 128, 'num_layers': 3, 'dropout': .1},
            'training': {'run_dir': 'outputs/fem/training', 'epochs': 100, 'batch_size': 2,
                         'lr': .001, 'weight_decay': .0001, 'seed': 42, 'mask_ratio': .25},
        }
        with patch.object(run, 'load_config', return_value=config), patch.object(run, 'verify_data'), \
             patch.object(run, 'run') as execute:
            run.train('fem', device='cpu')
        cmd = execute.call_args.args[0]
        self.assertEqual(cmd[cmd.index('--run-dir') + 1], 'outputs/fem/training')
        self.assertEqual(cmd[cmd.index('--device') + 1], 'cpu')

    def test_fem_resume_is_configurable(self):
        config = {
            "data": {"root": "data/raw/agentfem", "cache_dir": "data/processed/fem/cache",
                     "max_regions": 32, "max_nodes": 4096, "reference_load_N": 1000},
            "encoder": {"hidden_dim": 128, "num_layers": 3, "dropout": .1},
            "training": {"run_dir": "outputs/fem/training", "epochs": 101, "batch_size": 2,
                         "lr": .001, "weight_decay": .0001, "seed": 42, "mask_ratio": .25,
                         "resume_checkpoint": "outputs/fem/training/last.pt"},
        }
        with patch.object(run, "load_config", return_value=config), \
             patch.object(run, "verify_data"), patch.object(run, "run") as execute:
            run.train("fem", device="cpu")
        cmd = execute.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--resume") + 1], "outputs/fem/training/last.pt")

    def test_text_evaluation_output_is_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'outputs/text/training/best.pt'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'placeholder')
            with patch.object(run, 'ROOT', Path(tmp)), \
                 patch.object(run, 'load_config', return_value={
                     'training': {'checkpoint_dir': 'outputs/text/training'}}), \
                 patch.object(run, 'verify_data'), patch.object(run, 'run') as execute:
                run.evaluate('text', device='cpu')
            cmd = execute.call_args.args[0]
            self.assertEqual(cmd[cmd.index('--output-dir') + 1], 'outputs/text/evaluation')

    def test_eval_looks_for_configured_cad_checkpoint(self):
        with patch.object(run, 'load_config', return_value={'output_dir': 'outputs/cad/training'}), \
             patch.object(run, 'verify_data'):
            with self.assertRaisesRegex(FileNotFoundError, 'outputs/cad/training/best.pt'):
                run.evaluate('cad', device='cpu')

    def test_eval_looks_for_configured_fem_checkpoint(self):
        config = {'training': {'run_dir': 'outputs/fem/training'},
                  'data': {'root': 'data/raw/agentfem', 'cache_dir': 'data/processed/fem/cache'}}
        with patch.object(run, 'load_config', return_value=config), patch.object(run, 'verify_data'):
            with self.assertRaisesRegex(FileNotFoundError, 'outputs/fem/training/best.pt'):
                run.evaluate('fem', device='cpu')


class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('migrate_fem_run', ROOT / 'scripts/migrate_fem_run.py')
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_copy_run_directory_with_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'old'; dst = Path(d) / 'new'
            src.mkdir(); (src / 'best.pt').write_bytes(b'checkpoint')
            (src / 'other.txt').write_text('not included')
            self.module.migrate(src, dst)
            self.assertEqual((dst / 'best.pt').read_bytes(), b'checkpoint')
            self.assertTrue((src / 'best.pt').is_file())
            self.assertFalse((dst / 'other.txt').exists())
            with self.assertRaises(FileExistsError):
                self.module.migrate(src, dst)

    def test_zip_archive_and_dry_run(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'experiments.zip'; dst = Path(d) / 'new'
            with ZipFile(src, 'w') as z:
                z.writestr('experiments/runs/fem_full/best.pt', b'model')
                z.writestr('experiments/runs/fem_full/history.csv', b'epoch,loss')
                z.writestr('__MACOSX/test', b'metadata')
            result = self.module.migrate(src, dst, dry_run=True)
            self.assertEqual(len(result), 2)
            self.assertFalse(dst.exists())
            self.module.migrate(src, dst)
            self.assertEqual((dst / 'best.pt').read_bytes(), b'model')
            self.assertFalse((dst / '__MACOSX').exists())


if __name__ == '__main__':
    unittest.main()
