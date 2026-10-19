"""Controlled, semantically equivalent AgentFEM engineering *proxy* descriptions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .metadata import CaseRecord, INPUT_FIELDS, EngineeringMetadata

TEMPLATE_IDS = ("concise", "requirements", "narrative")


@dataclass(frozen=True)
class Description:
    case_id: str
    template_id: str
    text: str
    values: dict[str, float]
    units: dict[str, str]
    masked_field: str | None = None


def _fmt(value: float) -> str:
    """Nine significant digits, deterministic; numeric ground truth remains unrounded."""
    return format(value, ".9g")


def render(record: CaseRecord, metadata: EngineeringMetadata, template_id: str,
           masked_field: str | None = None) -> Description:
    if template_id not in TEMPLATE_IDS:
        raise ValueError(f"unknown template_id {template_id}")
    if masked_field is not None and masked_field not in INPUT_FIELDS:
        raise ValueError(f"unknown masked field {masked_field}")
    v = record.values
    ctx = metadata.context

    def pair(field: str) -> str:
        return "[unspecified]" if masked_field == field else f"{_fmt(v[field])} {metadata_unit(field)}"

    def metadata_unit(field: str) -> str:
        from .metadata import UNITS
        return UNITS[field] if UNITS[field] != "1" else "(dimensionless)"

    condition = (f"linear isotropic elasticity (Young's modulus {_fmt(ctx['young_modulus_Pa'] / 1e9)} GPa, "
                 f"Poisson ratio {_fmt(ctx['poisson_ratio'])}, density {_fmt(ctx['density_kg_m3'])} kg/m^3); "
                 f"overall height {_fmt(ctx['height_m'])} m; {ctx['fixed_boundary']}; "
                 f"{ctx['loading']}")
    if template_id == "concise":
        text = (f"{ctx['component'].capitalize()}. Named design parameters: depth {pair('depth')}, "
                f"radius {pair('radius')}, waist {pair('waist')}, bow {pair('bow')}. "
                f"Physical load: {pair('physical_load_N')}. Conditions: {condition}.")
    elif template_id == "requirements":
        text = (f"Engineering specification for a {ctx['component']}: set the named parameter depth to "
                f"{pair('depth')}, radius to {pair('radius')}, waist to {pair('waist')}, "
                f"and bow to {pair('bow')}. Apply {pair('physical_load_N')} of physical load. "
                f"Use {condition}.")
    else:
        text = (f"This {ctx['component']} is represented by the named design values "
                f"bow={pair('bow')}, waist={pair('waist')}, radius={pair('radius')}, "
                f"and depth={pair('depth')}. The applied physical load is {pair('physical_load_N')}. "
                f"Its assumed setup is {condition}.")
    from .metadata import UNITS
    return Description(record.case_id, template_id, text, dict(v), dict(UNITS), masked_field)
