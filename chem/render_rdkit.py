"""可选的结构式（键线式/球棍式）绘图，基于 RDKit。

RDKit 是重依赖（Windows/Linux 轮子约 100 MB 起，且部分平台/新版本
Python 还没有预编译包），因此本模块**全程懒加载**：

* 没装 RDKit 时 :func:`rdkit_available` 返回 ``False``，调用绘图会抛
  :class:`RenderError`，由上层降级为文本输出；
* 装了 RDKit 才真正 ``import rdkit``。

这样插件在没装 RDKit 的环境里也能完整工作，只是少了结构式图片。
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache

_DRAW_COLORS: dict[str, tuple[float, float, float]] = {
    "C": (0.12, 0.12, 0.14),
    "N": (0.15, 0.35, 0.85),
    "O": (0.80, 0.15, 0.15),
    "S": (0.80, 0.55, 0.10),
    "P": (0.85, 0.45, 0.10),
    "F": (0.20, 0.65, 0.35),
    "Cl": (0.20, 0.65, 0.35),
    "Br": (0.55, 0.25, 0.15),
    "I": (0.45, 0.20, 0.65),
    "H": (0.35, 0.35, 0.38),
}


class RenderError(RuntimeError):
    """结构式渲染失败。"""


@dataclass
class StructureOptions:
    """结构式绘图参数。"""

    width: int = 900
    height: int = 700
    background: tuple[float, float, float] = (1.0, 1.0, 1.0)
    bond_line_width: float = 2.4
    fixed_bond_length: float = 32.0
    add_stereo_annotations: bool = True
    rotate: float = 0.0
    """二维坐标的旋转角度；0 表示用 RDKit 的默认取向。"""


@lru_cache(maxsize=1)
def _load_rdkit():
    """尝试导入 RDKit，返回 ``(Chem, Draw, None)`` 或 ``(None, None, 原因)``。"""
    try:
        from rdkit import Chem  # noqa: PLC0415 - 懒加载
        from rdkit.Chem import Draw  # noqa: PLC0415 - 懒加载
        from rdkit.Chem.Draw import rdMolDraw2D  # noqa: PLC0415 - 懒加载
    except Exception as error:  # pragma: no cover - 依赖缺失路径
        return None, None, f"{type(error).__name__}: {error}"
    return (Chem, Draw, rdMolDraw2D), None, None


def rdkit_available() -> bool:
    """RDKit 是否可用。"""
    package, _, _ = _load_rdkit()
    return package is not None


def rdkit_unavailable_reason() -> str:
    """RDKit 不可用的原因，可用时返回空串。"""
    _, _, reason = _load_rdkit()
    return reason or ""


def rdkit_version() -> str:
    """RDKit 版本号，不可用时返回空串。"""
    package, _, _ = _load_rdkit()
    if package is None:
        return ""
    try:
        import rdkit  # noqa: PLC0415 - 懒加载

        return str(rdkit.__version__)
    except Exception:  # pragma: no cover - 版本读取失败
        return "unknown"


def render_structure_png(
    smiles: str,
    options: StructureOptions | None = None,
) -> bytes:
    """把 SMILES 渲染成结构式 PNG。

    Args:
        smiles: 合法的 SMILES。
        options: 绘图参数。

    Returns:
        PNG 文件内容。

    Raises:
        RenderError: RDKit 不可用、SMILES 无法解析或绘图失败时抛出。
    """
    package, _, reason = _load_rdkit()
    if package is None:
        raise RenderError(
            "未安装 RDKit，无法绘制结构式。可执行 pip install rdkit 后重启，"
            "或继续使用结构简式图片。原因：" + reason,
        )
    Chem, Draw, rdMolDraw2D = package
    opts = options or StructureOptions()

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RenderError(f"RDKit 无法解析该 SMILES：{smiles}")

    try:
        Chem.rdDepictor.Compute2DCoords(molecule)
    except Exception:  # pragma: no cover - 个别分子没有可用的二维坐标
        pass

    drawer = rdMolDraw2D.MolDraw2DCairo(opts.width, opts.height)
    draw_options = drawer.drawOptions()
    draw_options.bondLineWidth = opts.bond_line_width
    draw_options.fixedBondLength = opts.fixed_bond_length
    draw_options.addStereoAnnotation = opts.add_stereo_annotations
    draw_options.clearBackground = True
    draw_options.setBackgroundColour(opts.background)
    for symbol, color in _DRAW_COLORS.items():
        try:
            draw_options.updateAtomPalette({symbol: color})
        except Exception:  # pragma: no cover - 不同版本 API 略有差异
            break

    try:
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
        drawer.FinishDrawing()
    except Exception as error:  # pragma: no cover - 绘图内部异常
        raise RenderError(f"RDKit 绘图失败：{error}") from error

    data = drawer.GetDrawingText()
    if not data:
        raise RenderError("RDKit 返回了空的绘图结果")
    return bytes(data)


def render_structure_raster_png(
    smiles: str,
    options: StructureOptions | None = None,
) -> bytes:
    """备选实现：用 RDKit 的 ``Draw.MolToImage`` 出图。

    当 Cairo 后端不可用（少数精简安装）时作为退路。
    """
    package, _, reason = _load_rdkit()
    if package is None:
        raise RenderError("未安装 RDKit，无法绘制结构式。原因：" + reason)
    Chem, Draw, _ = package
    opts = options or StructureOptions()

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RenderError(f"RDKit 无法解析该 SMILES：{smiles}")
    try:
        Chem.rdDepictor.Compute2DCoords(molecule)
        image = Draw.MolToImage(molecule, size=(opts.width, opts.height))
    except Exception as error:  # pragma: no cover - 退路实现
        raise RenderError(f"RDKit 绘图失败：{error}") from error

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_structure_safe(
    smiles: str,
    options: StructureOptions | None = None,
) -> tuple[bytes | None, str]:
    """带降级的结构式绘图，返回 ``(PNG 字节或 None, 失败原因)``。"""
    try:
        return render_structure_png(smiles, options), ""
    except RenderError as error:
        primary = str(error)
    try:
        return render_structure_raster_png(smiles, options), ""
    except RenderError:
        return None, primary


def describe_rdkit() -> str:
    """返回 RDKit 环境说明，供 ``/chem 状态`` 使用。"""
    if rdkit_available():
        return f"RDKit 可用，版本 {rdkit_version()}，可绘制结构式"
    return (
        "RDKit 不可用（" + (rdkit_unavailable_reason() or "未知原因")
        + "），仅输出结构简式"
    )
