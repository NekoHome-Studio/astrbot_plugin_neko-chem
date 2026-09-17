"""用 Pillow 画键线式（骨架式/骨骼式）结构图。

键线式的画法约定（与教材一致）：

* **碳原子不写出来**：碳在顶点上，两端也默认是碳；
* **杂原子标注符号**：O、N、S、Cl 等直接写元素符号，并带上氢原子
  （``OH``、``NH₂``、``SH``），电荷写成上标；
* **双键画两条平行线**：环内的双键第二条线画在环内侧并稍短，
  环外的双键（例如羰基 ``C=O``）两条线等长；
* **三键画三条线**；
* **芳香环用凯库勒式**：交替双键，而不是画一个圆圈 —— 教材主要教的
  是凯库勒式，而且这种画法对吡啶、萘同样适用。

二维坐标由 :mod:`chem.layout` 生成，本模块只负责把坐标变成像素。
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

from .fonts import describe_font_setup, load_font, text_width
from .layout import Layout, bond_orders_for_drawing
from .smiles import Molecule

#: 杂原子按元素着色（CPK 风格），碳骨架统一用主色。
ATOM_COLORS: dict[str, tuple[int, int, int]] = {
    "O": (200, 38, 38),
    "N": (37, 99, 235),
    "S": (176, 128, 17),
    "P": (198, 106, 20),
    "F": (22, 163, 74),
    "Cl": (22, 163, 74),
    "Br": (146, 64, 14),
    "I": (109, 40, 217),
    "B": (194, 120, 60),
    "Si": (120, 113, 108),
}

#: 需要标注元素符号的原子：除了碳以外的重原子。
LABELED_SYMBOLS = frozenset(
    {
        "B", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I",
        "Se", "Te", "As", "Na", "K", "Ca", "Mg", "Fe", "Cu", "Zn",
        "Mn", "Ag", "Ba", "Al", "Li",
    },
)


class SkeletalRenderError(RuntimeError):
    """键线式渲染失败。"""


@dataclass
class SkeletalOptions:
    """键线式渲染参数。"""

    bond_length: float = 54.0
    """键长（像素）。整张图按它缩放。"""

    line_width: float = 2.6
    color: tuple[int, int, int] = (17, 24, 39)
    background: tuple[int, int, int] | None = (255, 255, 255)
    padding: int = 34
    supersample: int = 3

    font_scale: float = 0.62
    """杂原子标签字号相对键长的比例。"""

    subscript_scale: float = 0.66
    """下标（氢原子数）字号相对标签字号的比例。"""

    double_bond_gap: float = 0.17
    """双键两条平行线的间距（以键长为单位）。"""

    show_hetero_hydrogens: bool = True
    """是否在杂原子上写出氢原子（``OH``、``NH₂``）。"""

    show_charges: bool = True
    """是否写出电荷。"""

    title: str | None = None
    subtitle: str | None = None
    caption_size_ratio: float = 0.46
    accents: tuple[tuple[int, int, int], ...] = field(
        default_factory=lambda: ((37, 99, 235), (220, 38, 38)),
    )


# ---------------------------------------------------------------- 标签内容


def atom_label(
    molecule: Molecule,
    index: int,
    *,
    show_hydrogens: bool = True,
    show_charges: bool = True,
) -> tuple[str, int, str]:
    """给出某原子的标签内容，返回 ``(元素符号, 氢原子数, 电荷文字)``。

    碳原子通常返回空符号（键线式里不写碳）；只有当分子只有它一个原子，
    或者它带电荷时才会写出 ``C``。
    """
    atom = molecule.atoms[index]
    symbol = atom.symbol

    is_lone = len(molecule.atoms) == 1
    needs_label = symbol in LABELED_SYMBOLS or is_lone or (
        show_charges and atom.charge != 0
    )
    if not needs_label:
        return "", 0, ""

    hydrogens = 0
    if show_hydrogens and atom.hydrogens:
        # 碳上的一般不写氢（键线式约定），杂原子要写
        if symbol != "C":
            hydrogens = atom.hydrogens
        elif is_lone:
            hydrogens = atom.hydrogens

    charge_text = ""
    if show_charges and atom.charge:
        sign = "+" if atom.charge > 0 else "-"
        magnitude = abs(atom.charge)
        charge_text = sign if magnitude == 1 else f"{magnitude}{sign}"
    return symbol, hydrogens, charge_text


def _label_width(font, sub_font, symbol: str, hydrogens: int, charge: str) -> float:
    """估算标签占用的宽度（用于避让计算）。"""
    width = text_width(font, symbol)
    if hydrogens:
        width += text_width(sub_font, str(hydrogens))
    if charge:
        width += text_width(sub_font, charge)
    return width


# ---------------------------------------------------------------- 绘制实现


def render_skeletal_png(
    molecule: Molecule,
    layout: Layout,
    options: SkeletalOptions | None = None,
) -> bytes:
    """把分子图与排版结果画成键线式 PNG。

    Args:
        molecule: 分子图。
        layout: :func:`chem.layout.compute_layout` 的结果。
        options: 渲染参数。

    Returns:
        PNG 文件内容。

    Raises:
        SkeletalRenderError: Pillow 不可用、排版不可用或几何异常时抛出。
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:  # pragma: no cover - 依赖缺失路径
        raise SkeletalRenderError(
            "缺少 pillow 依赖，无法渲染图片。请执行 pip install pillow。",
        ) from error

    if not layout.drawable:
        raise SkeletalRenderError(
            "该分子的环系无法在平面上排成规整图形（"
            + "；".join(layout.issues)
            + "），为避免画出错误结构已放弃绘制",
        )
    if not molecule.atoms or not layout.coords:
        raise SkeletalRenderError("分子为空，无法绘制")

    opts = options or SkeletalOptions()
    scale = max(1, opts.supersample)
    bond = max(8.0, opts.bond_length) * scale
    line_width = max(1.0, opts.line_width) * scale
    gap = max(0.02, opts.double_bond_gap) * bond

    label_size = max(8, int(bond * opts.font_scale))
    sub_size = max(6, int(label_size * opts.subscript_scale))
    caption_size = max(6, int(label_size * opts.caption_size_ratio))
    label_font = load_font(label_size)
    sub_font = load_font(sub_size)
    caption_font = load_font(caption_size)

    # 归一化坐标 → 像素坐标（数学的 y 轴向上，图像 y 轴向下，要翻转）
    min_x, min_y, max_x, max_y = layout.bounds()
    points = {
        index: (
            (point[0] - min_x) * bond,
            (max_y - point[1]) * bond,
        )
        for index, point in layout.coords.items()
    }

    pad = max(0, opts.padding) * scale
    content_w = (max_x - min_x) * bond
    content_h = (max_y - min_y) * bond
    # 把标签可能超出几何范围的部分也算进来
    margin = label_size * 0.9
    width = int(content_w + pad * 2 + margin * 2)
    height = int(content_h + pad * 2 + margin * 2)
    points = {
        index: (x + pad + margin, y + pad + margin)
        for index, (x, y) in points.items()
    }

    caption_gap = int(label_size * 0.5) if opts.subtitle else 0
    title_gap = int(label_size * 0.42) if opts.title else 0
    title_h = (
        sum(caption_font.getmetrics()) if opts.title else 0
    )
    subtitle_h = (
        sum(caption_font.getmetrics()) if opts.subtitle else 0
    )
    height += title_h + title_gap + caption_gap + subtitle_h

    background = opts.background
    if background is None:
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        halo = (255, 255, 255, 255)
    else:
        canvas = Image.new("RGB", (width, height), background)
        halo = background
    draw = ImageDraw.Draw(canvas)

    offset_y = title_h + title_gap
    if offset_y:
        points = {
            index: (x, y + offset_y) for index, (x, y) in points.items()
        }

    orders = bond_orders_for_drawing(molecule, layout)
    ring_centers = _ring_centers(molecule, layout, points)

    # 1) 先画所有键
    for key, order in orders.items():
        left, right = tuple(key)
        _draw_bond(
            draw,
            points[left],
            points[right],
            order,
            color=opts.color,
            line_width=line_width,
            gap=gap,
            bond_length=bond,
            ring_center=ring_centers.get(key),
        )

    # 2) 再画原子标签（带描边，压住底下的键线）
    for atom in molecule.atoms:
        point = points.get(atom.index)
        if point is None:
            continue
        symbol, hydrogens, charge = atom_label(
            molecule,
            atom.index,
            show_hydrogens=opts.show_hetero_hydrogens,
            show_charges=opts.show_charges,
        )
        if not symbol:
            continue
        color = ATOM_COLORS.get(symbol, opts.color)
        _draw_atom_label(
            draw,
            point,
            symbol,
            hydrogens,
            charge,
            label_font=label_font,
            sub_font=sub_font,
            color=color,
            halo=halo,
            line_width=line_width,
            bond=bond,
        )

    # 3) 标题与小字
    cursor_y = pad // 2
    if opts.title:
        title_color = opts.accents[0] if opts.accents else opts.color
        draw.text(
            ((width - text_width(caption_font, opts.title)) // 2, cursor_y),
            opts.title,
            font=caption_font,
            fill=title_color,
        )
    if opts.subtitle:
        subtitle_color = opts.accents[1] if len(opts.accents) > 1 else opts.color
        draw.text(
            (
                (width - text_width(caption_font, opts.subtitle)) // 2,
                height - subtitle_h - pad // 3,
            ),
            opts.subtitle,
            font=caption_font,
            fill=subtitle_color,
        )

    if scale > 1:
        canvas = canvas.resize(
            (max(1, width // scale), max(1, height // scale)),
            Image.LANCZOS,
        )

    buffer = io.BytesIO()
    if background is None:
        canvas.save(buffer, format="PNG", optimize=True)
    else:
        canvas.convert("RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _ring_centers(
    molecule: Molecule,
    layout: Layout,
    points: dict[int, tuple[float, float]],
) -> dict[frozenset[int], tuple[float, float]]:
    """算出每根环上键所对应的环心，供双键选内侧方向。"""
    centers: dict[frozenset[int], tuple[float, float]] = {}
    for ring in layout.rings:
        present = [atom for atom in ring if atom in points]
        if len(present) < 3:
            continue
        center = (
            sum(points[atom][0] for atom in present) / len(present),
            sum(points[atom][1] for atom in present) / len(present),
        )
        for index, atom in enumerate(ring):
            key = frozenset({atom, ring[(index + 1) % len(ring)]})
            centers.setdefault(key, center)
    return centers


def _draw_bond(
    draw,
    left: tuple[float, float],
    right: tuple[float, float],
    order: float,
    *,
    color: tuple[int, int, int],
    line_width: float,
    gap: float,
    bond_length: float,
    ring_center: tuple[float, float] | None,
) -> None:
    """画一根键：单线、双线或三线。"""
    dx, dy = right[0] - left[0], right[1] - left[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux

    if order < 1.5:
        draw.line([left, right], fill=color, width=int(round(line_width)))
        return

    if order < 2.5:
        _draw_double_bond(
            draw, left, right, (nx, ny), ring_center=ring_center,
            color=color, line_width=line_width, gap=gap, bond_length=bond_length,
        )
        return

    # 三键：中间一条 + 两侧各一条
    draw.line([left, right], fill=color, width=int(round(line_width)))
    for sign in (1.0, -1.0):
        shift = (nx * gap * sign, ny * gap * sign)
        draw.line(
            [
                (left[0] + shift[0], left[1] + shift[1]),
                (right[0] + shift[0], right[1] + shift[1]),
            ],
            fill=color,
            width=int(round(line_width * 0.85)),
        )


def _draw_double_bond(
    draw,
    left: tuple[float, float],
    right: tuple[float, float],
    normal: tuple[float, float],
    *,
    ring_center: tuple[float, float] | None,
    color: tuple[int, int, int],
    line_width: float,
    gap: float,
    bond_length: float,
) -> None:
    """画双键。环内的第二条线画在环内侧并稍短，环外两条线等长。"""
    nx, ny = normal
    if ring_center is None:
        # 环外双键（羰基、烯烃）：两条等长平行线
        for sign in (1.0, -1.0):
            shift = (nx * gap * 0.5 * sign, ny * gap * 0.5 * sign)
            draw.line(
                [
                    (left[0] + shift[0], left[1] + shift[1]),
                    (right[0] + shift[0], right[1] + shift[1]),
                ],
                fill=color,
                width=int(round(line_width)),
            )
        return

    # 环内双键：主线就是键本身，第二条线偏向环心并两端缩短
    draw.line([left, right], fill=color, width=int(round(line_width)))
    midpoint = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
    toward_center = (ring_center[0] - midpoint[0], ring_center[1] - midpoint[1])
    magnitude = math.hypot(*toward_center)
    if magnitude < 1e-6:
        return
    cx, cy = toward_center[0] / magnitude, toward_center[1] / magnitude
    shift = (cx * gap, cy * gap)
    shrink = bond_length * 0.16
    dx, dy = right[0] - left[0], right[1] - left[1]
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    draw.line(
        [
            (left[0] + shift[0] + ux * shrink, left[1] + shift[1] + uy * shrink),
            (right[0] + shift[0] - ux * shrink, right[1] + shift[1] - uy * shrink),
        ],
        fill=color,
        width=int(round(line_width * 0.9)),
    )


def _draw_atom_label(
    draw,
    point: tuple[float, float],
    symbol: str,
    hydrogens: int,
    charge: str,
    *,
    label_font,
    sub_font,
    color: tuple[int, int, int],
    halo,
    line_width: float,
    bond: float,
) -> None:
    """把元素符号居中画在原子上，氢数与电荷接在右边（带描边压线）。"""
    x, y = point
    stroke = max(1, int(round(line_width * 0.85)))
    draw.text(
        (x, y),
        symbol,
        font=label_font,
        fill=color,
        anchor="mm",
        stroke_width=stroke,
        stroke_fill=halo,
    )
    cursor = x + text_width(label_font, symbol) / 2
    drop = bond * 0.16
    rise = bond * 0.30

    if hydrogens:
        text = str(hydrogens)
        draw.text(
            (cursor, y + drop),
            text,
            font=sub_font,
            fill=color,
            anchor="lm",
            stroke_width=stroke,
            stroke_fill=halo,
        )
        cursor += text_width(sub_font, text)

    if charge:
        draw.text(
            (cursor, y - rise),
            charge,
            font=sub_font,
            fill=color,
            anchor="lm",
            stroke_width=stroke,
            stroke_fill=halo,
        )


def render_smiles_skeletal_png(
    smiles: str,
    options: SkeletalOptions | None = None,
) -> tuple[bytes | None, str]:
    """一站式入口：SMILES → 键线式 PNG。

    Returns:
        ``(PNG 字节或 None, 失败原因)``。
    """
    from .layout import compute_layout
    from .smiles import SmilesError, parse_smiles

    try:
        molecule = parse_smiles(smiles)
    except SmilesError as error:
        return None, f"SMILES 无法解析：{error.reason}"

    layout = compute_layout(molecule)
    try:
        return render_skeletal_png(molecule, layout, options), ""
    except SkeletalRenderError as error:
        return None, str(error)


__all__ = [
    "ATOM_COLORS",
    "SkeletalOptions",
    "SkeletalRenderError",
    "atom_label",
    "describe_font_setup",
    "render_skeletal_png",
    "render_smiles_skeletal_png",
]
