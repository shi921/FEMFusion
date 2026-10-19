"""Strict AgentFEM surface-field reader: case_id join, safe ZIP reads, SI units."""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile
import numpy as np

CASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MAX_ENTRY_BYTES = 256 * 1024 * 1024


@dataclass
class SurfaceFields:
    case_id: str
    points: np.ndarray
    triangles: np.ndarray
    displacement: np.ndarray  # SI: m, possibly rescaled to reference load
    stress: np.ndarray  # SI: Pa, possibly rescaled to reference load
    original_displacement: np.ndarray
    original_stress: np.ndarray
    physical_load_N: float
    reference_load_N: float
    load_scale: float
    source: str
    summary: dict
    quality: dict = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return len(self.points)

    @property
    def triangle_count(self) -> int:
        return len(self.triangles)


def validate_load(load: float, field: str = 'physical_load_N') -> float:
    try:
        load = float(load)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field} must be specified in N, got {load!r}') from exc
    if not np.isfinite(load) or load <= 0:
        raise ValueError(f'{field} must be positive finite Newtons (N), got {load}')
    # The official experiments use 1,000 N or 10,000 N; detect common kN-as-N errors.
    if load < 100 or load > 1e7:
        raise ValueError(f'{field}={load} N is outside supported [100, 1e7] N; check N versus kN units')
    return load


def _valid_member(name: str) -> bool:
    p = PurePosixPath(name)
    return (bool(name) and not name.startswith('/') and '\\' not in name
            and all(part not in ('', '.', '..') for part in name.split('/'))
            and p.as_posix() == name)


def _bytes_from_zip(zf: ZipFile, name: str) -> bytes:
    if not _valid_member(name):
        raise ValueError(f'Unsafe ZIP member path: {name!r}')
    try:
        info = zf.getinfo(name)
    except KeyError as exc:
        raise FileNotFoundError(f'Missing ZIP member {name!r} in {zf.filename}') from exc
    if info.is_dir() or info.file_size > MAX_ENTRY_BYTES:
        raise ValueError(f'Unsafe/oversized ZIP entry {name!r}: {info.file_size} bytes')
    # Disallow symbolic links in archives (unix type 0120000).
    if (info.external_attr >> 16) & 0o170000 == 0o120000:
        raise ValueError(f'ZIP symbolic link forbidden: {name!r}')
    with zf.open(info) as stream:
        payload = stream.read(MAX_ENTRY_BYTES + 1)
    if len(payload) > MAX_ENTRY_BYTES:
        raise ValueError(f'ZIP member too large: {name!r}')
    return payload


def _validate_arrays(case_id: str, fields: dict) -> tuple[np.ndarray, ...]:
    required = ('points', 'triangles', 'displacement', 'stress')
    absent = set(required) - set(fields)
    if absent:
        raise ValueError(f'{case_id}: missing NPZ arrays {sorted(absent)}')
    pts, tri, disp, stress = (np.asarray(fields[k]) for k in required)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        raise ValueError(f'{case_id}: points must be [N>=3,3] (m), got {pts.shape}')
    if tri.ndim != 2 or tri.shape[1] != 3 or len(tri) < 1:
        raise ValueError(f'{case_id}: triangles must be [F>=1,3], got {tri.shape}')
    if disp.shape != pts.shape or stress.shape != (len(pts),):
        raise ValueError(f'{case_id}: displacement must be {pts.shape}, stress [{len(pts)}]')
    if any(a.dtype.kind not in 'fi' for a in (pts, disp, stress)):
        raise ValueError(f'{case_id}: physical arrays must be numeric')
    if tri.dtype.kind not in 'iu':
        raise ValueError(f'{case_id}: triangle node indices must be integers')
    for name, a in [('points', pts), ('displacement', disp), ('stress', stress)]:
        if not np.isfinite(a).all():
            raise ValueError(f'{case_id}: non-finite values in {name}')
    if np.any(tri < 0) or np.any(tri >= len(pts)):
        raise ValueError(f'{case_id}: triangle node index out of bounds for {len(pts)} nodes')
    if np.any((tri[:, 0] == tri[:, 1]) | (tri[:, 1] == tri[:, 2]) | (tri[:, 0] == tri[:, 2])):
        raise ValueError(f'{case_id}: degenerate triangle with duplicate indices')
    # Some official surface-sampled stress arrays contain negative interpolated values.
    # Preserve these as supplied (no silent clipping), and record their count for QC.
    return (pts.astype(np.float32), tri.astype(np.int64), disp.astype(np.float64),
            stress.astype(np.float64))


def read_fields(case_id: str, source: str | Path, physical_load_N: float,
                reference_load_N: float = 1000., *, member: str | None = None,
                summary_member: str | None = None) -> SurfaceFields:
    """Read an unpacked fields.npz or a ZIP member (never positional ZIP matching)."""
    if not CASE_RE.fullmatch(case_id):
        raise ValueError(f'Unsafe case_id: {case_id!r}')
    load = validate_load(physical_load_N)
    ref = validate_load(reference_load_N, 'reference_load_N')
    source = Path(source)
    summary: dict = {}
    where = str(source)
    try:
        if source.suffix.lower() == '.zip':
            member = member or f'fem/{case_id}/fields.npz'
            summary_member = summary_member or f'fem/{case_id}/summary.json'
            if member != f'fem/{case_id}/fields.npz':
                raise ValueError(f'{case_id}: ZIP field member does not match case_id: {member}')
            with ZipFile(source) as zf:
                payload = _bytes_from_zip(zf, member)
                if summary_member in zf.namelist():
                    if summary_member != f'fem/{case_id}/summary.json':
                        raise ValueError(f'{case_id}: mismatched summary member')
                    summary = json.loads(_bytes_from_zip(zf, summary_member))
            where += f'::{member}'
            with np.load(BytesIO(payload), allow_pickle=False) as npz:
                arrays = {name: npz[name] for name in npz.files}
        else:
            if source.name != 'fields.npz':
                raise ValueError('Unpacked source must be named fields.npz')
            with np.load(source, allow_pickle=False) as npz:
                arrays = {name: npz[name] for name in npz.files}
            summary_path = source.with_name('summary.json')
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
        reported = summary.get('physical_load_N', summary.get('load_N'))
        if reported is not None and not np.isclose(validate_load(reported, 'summary.load_N'), load, rtol=1e-6):
            raise ValueError(f'{case_id}: CSV physical_load_N={load} disagrees with summary load_N={reported}')
        if summary.get('design', {}).get('name') not in (None, case_id):
            raise ValueError(f'{case_id}: summary design.name mismatch: {summary["design"]["name"]}')
        pts, tri, disp, stress = _validate_arrays(case_id, arrays)
        scale = ref / load
        # Preserve the ORIGINAL source arrays, including float64 values, unchanged.
        # Apply physical load scaling in float64 SI units before float32 model conversion.
        raw_disp = np.asarray(arrays['displacement']).copy()
        raw_stress = np.asarray(arrays['stress']).copy()
        return SurfaceFields(case_id, pts, tri,
                             (disp * scale).astype(np.float32),
                             (stress * scale).astype(np.float32),
                             raw_disp, raw_stress, load, ref, scale, where, summary,
                             {'negative_stress_nodes': int((stress < 0).sum()),
                              'negative_stress_min_Pa': float(stress.min()) if (stress < 0).any() else None})
    except Exception as exc:
        raise type(exc)(f'FEM case={case_id} source={where}: {exc}') from exc
