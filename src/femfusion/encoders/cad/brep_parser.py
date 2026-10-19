"""Native OpenCascade STEP -> explicit topological and geometric measurements.

Never tessellates geometry. Face/edge indices are local to *one* shape only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
import hashlib
import json
import math
import re
import tempfile
import zipfile


class CADParseError(ValueError):
    """A STEP document or a required geometric invariant is invalid."""


def _occ():
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepTools import BRepTools
        from OCP.Bnd import Bnd_Box
        from OCP.GeomLProp import GeomLProp_SLProps
        from OCP.GProp import GProp_GProps
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX, TopAbs_REVERSED
        from OCP.TopExp import TopExp, TopExp_Explorer
        from OCP.TopoDS import TopoDS
        # Some OCP builds do not expose this specialized container binding.
        # TopExp_Explorer remains available and supports the same topology walk.
        try:
            from OCP.TopTools import TopTools_IndexedMapOfShape
        except ImportError:
            TopTools_IndexedMapOfShape = None
    except ImportError as exc:
        raise ImportError("Native STEP parsing requires OCP (OpenCascade); no mesh fallback. See docs/SETUP.md") from exc
    return locals()


class _ExplorerIndexMap:
    """OCP-version-independent indexed topology map with OCC's 1-based API.

    TopoDS_Shape.IsSame compares underlying geometry and placement while
    ignoring orientation, matching the semantics needed for edge incidence.
    """

    def __init__(self, shape, kind, explorer_cls):
        self._shapes = []
        explorer = explorer_cls(shape, kind)
        while explorer.More():
            candidate = explorer.Current()
            if self.FindIndex(candidate) == 0:
                self._shapes.append(candidate)
            explorer.Next()

    def Extent(self):
        return len(self._shapes)

    def FindKey(self, index):
        if not 1 <= index <= len(self._shapes):
            raise IndexError(f'OCC shape index out of range: {index}')
        return self._shapes[index - 1]

    def FindIndex(self, shape):
        for i, candidate in enumerate(self._shapes, 1):
            if candidate.IsSame(shape):
                return i
        return 0


def _maps(shape, kind, occ):
    indexed_map = occ.get('TopTools_IndexedMapOfShape')
    if indexed_map is not None:
        result = indexed_map()
        occ['TopExp'].MapShapes_s(shape, kind, result)
        return result
    return _ExplorerIndexMap(shape, kind, occ['TopExp_Explorer'])


def _topods_cast(shape, kind: str, occ):
    """Cast TopoDS_Shape across OCP bindings with/without ``*_s`` methods.

    OCP releases expose these casts as either TopoDS.Face_s/Edge_s or
    TopoDS.Face/Edge. This does not change topology or stable face indices.
    """
    topods = occ['TopoDS']
    cast = getattr(topods, f'{kind}_s', None)
    if cast is None:
        cast = getattr(topods, kind, None)
    if cast is None:
        raise ImportError(f'OCP.TopoDS lacks both {kind}_s and {kind} casts')
    return cast(shape)


def _box(shape, occ):
    bbox = occ['Bnd_Box']()
    occ['BRepBndLib'].Add_s(shape, bbox)
    if bbox.IsVoid():
        raise CADParseError('OpenCascade returned a void bounding box')
    # OCP wheel bindings differ: some expose Bnd_Box.Get() as six floats,
    # while others return an unregistered C++ Bnd_Box::Limits type and raise
    # TypeError before Python receives a result. CornerMin/CornerMax return
    # ordinary gp_Pnt objects in both supported binding families.
    try:
        result = tuple(float(t) for t in bbox.Get())
    except TypeError:
        minimum = bbox.CornerMin().Coord()
        maximum = bbox.CornerMax().Coord()
        result = tuple(float(t) for t in (*minimum, *maximum))
    if len(result) != 6:
        raise CADParseError(f'Expected six B-rep bounding box coordinates, got: {result}')
    if not all(math.isfinite(t) for t in result):
        raise CADParseError(f'Non-finite B-rep bounding box: {result}')
    return result


def _measure(shape, mode, occ):
    props = occ['GProp_GProps']()
    if mode == 'surface':
        occ['BRepGProp'].SurfaceProperties_s(shape, props)
    elif mode == 'linear':
        occ['BRepGProp'].LinearProperties_s(shape, props)
    else:
        occ['BRepGProp'].VolumeProperties_s(shape, props)
    amount = float(props.Mass())
    center = tuple(float(t) for t in props.CentreOfMass().Coord())
    if not math.isfinite(amount) or not all(math.isfinite(t) for t in center):
        raise CADParseError(f'Invalid {mode} properties')
    return amount, center


def _enum_name(value):
    raw = str(value).split('.')[-1].removeprefix('GeomAbs_')
    # OpenCascade names e.g. BSplineSurface/BezierCurve (not snake_case).
    return re.sub(r'(?<=[a-z])(?=[A-Z])', '_', raw).lower()


@dataclass
class FaceInfo:
    face_id: str
    source_index: int
    surface_type: str
    area: float
    center: tuple[float, float, float]
    bbox: tuple[float, float, float, float, float, float]
    normal: tuple[float, float, float]
    normal_valid: bool
    mean_curvature: float
    gaussian_curvature: float
    curvature_valid: bool
    edge_indices: list[int]
    signature: str
    missing: list[str]


@dataclass
class EdgeInfo:
    source_index: int
    curve_type: str
    length: float
    face_indices: list[int]
    degenerate: bool
    missing: list[str]


@dataclass
class BRepInfo:
    source: str
    bbox: tuple[float, float, float, float, float, float]
    volume: float
    solid_count: int
    edge_count: int
    vertex_count: int
    faces: list[FaceInfo]
    edges: list[EdgeInfo]

    def asdict(self):
        return asdict(self)


def parse_shape(shape, source: str = '<in-memory>') -> BRepInfo:
    """Inspect an OpenCascade TopoDS_Shape without meshing.

    Fails closed for invalid/empty B-reps, zero or negative volume, missing face
    areas, or invalid boxes. Non-essential normals/curvatures have validity flags.
    """
    o = _occ()
    if shape.IsNull():
        raise CADParseError(f'{source}: null STEP shape')
    if not o['BRepCheck_Analyzer'](shape).IsValid():
        raise CADParseError(f'{source}: invalid B-rep topology (BRepCheck_Analyzer)')
    face_map = _maps(shape, o['TopAbs_FACE'], o)
    edge_map = _maps(shape, o['TopAbs_EDGE'], o)
    solid_map = _maps(shape, o['TopAbs_SOLID'], o)
    vertices = _maps(shape, o['TopAbs_VERTEX'], o)
    if face_map.Extent() == 0 or solid_map.Extent() == 0:
        raise CADParseError(f'{source}: expected at least one Face and Solid')
    global_bbox = _box(shape, o)
    volume, _ = _measure(shape, 'volume', o)
    if volume <= 1e-12:
        raise CADParseError(f'{source}: non-positive/degenerate solid volume: {volume}')
    face_edges = {}
    incidence = {idx: [] for idx in range(1, edge_map.Extent() + 1)}
    faces = []
    for i in range(1, face_map.Extent() + 1):
        face = _topods_cast(face_map.FindKey(i), 'Face', o)
        try:
            area, center = _measure(face, 'surface', o)
            bbox = _box(face, o)
        except Exception as exc:
            raise CADParseError(f'{source}: Face {i} geometry failed: {exc}') from exc
        if area <= 1e-12:
            raise CADParseError(f'{source}: degenerate Face {i}: area={area}')
        adaptor = o['BRepAdaptor_Surface'](face)
        surface_type = _enum_name(adaptor.GetType())
        normal = (0.0, 0.0, 0.0)
        mean, gaussian, normal_valid, curvature_valid = 0.0, 0.0, False, False
        missing = []
        try:
            # BRepTools UV bounds are face-specific (can differ from untrimmed surface).
            umin, umax, vmin, vmax = o['BRepTools'].UVBounds_s(face)
            if not all(math.isfinite(v) for v in (umin, umax, vmin, vmax)):
                raise ValueError('Unbounded face UV range')
            uv = ((umin + umax) / 2, (vmin + vmax) / 2)
            props = o['GeomLProp_SLProps'](o['BRep_Tool'].Surface_s(face), *uv, 2, 1e-7)
            if props.IsNormalDefined():
                normal = tuple(float(t) for t in props.Normal().Coord())
                if face.Orientation() == o['TopAbs_REVERSED']:
                    normal = tuple(-t for t in normal)
                normal_valid = all(math.isfinite(t) for t in normal)
            if props.IsCurvatureDefined():
                mean, gaussian = float(props.MeanCurvature()), float(props.GaussianCurvature())
                curvature_valid = math.isfinite(mean) and math.isfinite(gaussian)
        except Exception as exc:
            missing.append(f'normal_or_curvature: {type(exc).__name__}: {exc}')
        if not normal_valid:
            missing.append('normal')
            normal = (0.0, 0.0, 0.0)
        if not curvature_valid:
            missing.append('curvature')
            mean, gaussian = 0.0, 0.0
        local_edges = _maps(face, o['TopAbs_EDGE'], o)
        ids = []
        for k in range(1, local_edges.Extent() + 1):
            eid = edge_map.FindIndex(local_edges.FindKey(k))
            if eid < 1:
                raise CADParseError(f'{source}: Face {i} references an unindexed Edge')
            ids.append(eid)
            incidence[eid].append(i)
        face_edges[i] = ids
        # Diagnostic only; signatures/indices not stable across changed STEP exports.
        quantized = [round(v, 6) for v in (*center, area, *bbox)]
        signature = hashlib.sha256(json.dumps([surface_type, quantized]).encode()).hexdigest()[:16]
        faces.append(FaceInfo(f'F{i}', i, surface_type, area, center, bbox, normal,
                              normal_valid, mean, gaussian, curvature_valid, ids,
                              signature, missing))
    edges = []
    for i in range(1, edge_map.Extent() + 1):
        edge = _topods_cast(edge_map.FindKey(i), 'Edge', o)
        degenerate = bool(o['BRep_Tool'].Degenerated_s(edge))
        missing = []
        if degenerate:
            length, curve_type = 0.0, 'unknown'
            missing.append('degenerate_edge')
        else:
            try:
                length, _ = _measure(edge, 'linear', o)
                curve_type = _enum_name(o['BRepAdaptor_Curve'](edge).GetType())
                if length <= 1e-12:
                    degenerate = True
                    missing.append('zero_length')
            except Exception as exc:
                length, curve_type = 0.0, 'unknown'
                degenerate = True
                missing.append(f'curve: {type(exc).__name__}: {exc}')
        edges.append(EdgeInfo(i, curve_type, length, sorted(set(incidence[i])), degenerate, missing))
    return BRepInfo(source, global_bbox, volume, solid_map.Extent(), edge_map.Extent(),
                    vertices.Extent(), faces, edges)


def parse_step(path: str | Path) -> BRepInfo:
    path = Path(path)
    if not path.is_file() or path.suffix.lower() not in {'.step', '.stp'}:
        raise CADParseError(f'Expected an existing .step/.stp file: {path}')
    o = _occ()
    reader = o['STEPControl_Reader']()
    status = reader.ReadFile(str(path.resolve()))
    if status != o['IFSelect_RetDone']:
        raise CADParseError(f'{path}: STEPControl_Reader.ReadFile status={status}')
    if reader.TransferRoots() <= 0:
        raise CADParseError(f'{path}: STEP reader transferred no roots')
    return parse_shape(reader.OneShape(), str(path))


def _safe_zip_member(name: str) -> bool:
    p = PurePosixPath(name.replace('\\', '/'))
    return bool(name) and not p.is_absolute() and '..' not in p.parts and p.suffix.lower() in {'.step', '.stp'}


def step_sources(raw_dir: str | Path) -> list[dict]:
    """Recursively index valid STEP members from geometry ZIPs and loose STEPs.

    ZIPs never get extracted during indexing. Unsafe member paths are ignored.
    """
    root = Path(raw_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f'Expected raw AgentFEM data in {root}')
    sources = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or not path.resolve().is_relative_to(root):
            continue
        if path.suffix.lower() in {'.step', '.stp'}:
            sources.append({'kind': 'file', 'path': str(path), 'member': None})
        elif path.suffix.lower() == '.zip' and 'geometry' in path.name.lower():
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if not member.is_dir() and _safe_zip_member(member.filename):
                        sources.append({'kind': 'zip', 'path': str(path), 'member': member.filename})
    return sources


def match_case_id(case_id: str, sources: list[dict]) -> dict:
    """Enforce unique file match, not CSV-row order or approximate numerical match."""
    cid = str(case_id).strip()
    if not cid or '/' in cid or '\\' in cid:
        raise CADParseError(f'Invalid case_id: {cid!r}')
    def matches(src):
        source_path = PurePosixPath(src['member'] or src['path'])
        # Official AgentFEM geometry archives store each sample at
        # fem/<case_id>/geometry.step. The STEP basename is identical for
        # every case, so the enclosing directory is the authoritative ID.
        if source_path.stem.casefold() == cid.casefold():
            return True
        if (source_path.stem.casefold() == 'geometry'
                and source_path.parent.name.casefold() == cid.casefold()):
            return True
        # Also support loose files such as geometry-case_A.step, without
        # confusing IDs such as case_1 and case_10.
        return bool(re.search(r'(?<![a-zA-Z0-9])' + re.escape(cid) + r'(?![a-zA-Z0-9])', source_path.stem, re.I))
    matched = [src for src in sources if matches(src)]
    if len(matched) != 1:
        raise CADParseError(f'case_id={cid!r}: expected 1 STEP, found {len(matched)}; '
                            f'candidates={[s["member"] or s["path"] for s in matched]}')
    return matched[0]


def parse_source(source: dict, cache_dir: str | Path, max_member_bytes=256 * 1024 * 1024) -> BRepInfo:
    """Read a matching STEP member via isolated temp file; never extract ZIP wholesale."""
    if source['kind'] == 'file':
        return parse_step(source['path'])
    if source['kind'] != 'zip' or not _safe_zip_member(source['member']):
        raise CADParseError(f'Unsafe or unsupported CAD source: {source}')
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source['path']) as archive:
        info = archive.getinfo(source['member'])
        if info.file_size > max_member_bytes:
            raise CADParseError(f'ZIP member exceeds {max_member_bytes} bytes: {info.filename}')
        with tempfile.TemporaryDirectory(prefix='step-', dir=cache_dir) as temporary:
            path = Path(temporary) / ('source' + PurePosixPath(info.filename).suffix.lower())
            with archive.open(info) as inp, path.open('wb') as out:
                import shutil
                shutil.copyfileobj(inp, out)
            if path.stat().st_size != info.file_size:
                raise CADParseError(f'Truncated ZIP member: {info.filename}')
            parsed = parse_step(path)
            parsed.source = f'{source["path"]}::{source["member"]}'
            return parsed
