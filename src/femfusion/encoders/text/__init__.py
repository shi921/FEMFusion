"""Single-modality text encoder. Exports are lazy to keep `python -m` clean."""
from importlib import import_module

_EXPORTS = {
    "EngineeringMetadata": "metadata", "INPUT_FIELDS": "metadata", "UNITS": "metadata",
    "TEMPLATE_IDS": "templates", "render": "templates",
    "EngineeringTextDataset": "dataset", "collate_views": "dataset", "collate_pairs": "dataset",
    "EngineeringTextEncoder": "model", "load_pretrained": "model",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
