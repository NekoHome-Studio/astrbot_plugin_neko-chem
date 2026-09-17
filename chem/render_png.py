"""用 Pillow 把结构简式渲染成 PNG 图片。

之所以自己排版而不是直接把字符串交给字体：简式里的下标必须是**真下标**
（``CH₃``），而 Unicode 下标字符在很多中文字体里缺字形，会变成方框。
这里的做法是把下标当成"更小的字号 + 基线偏移"来画，因此只依赖字体
是否含数字与字母，几乎不会缺字形。

不依赖 AstrBot，也不依赖 RDKit；只有 Pillow（``pillow``）一个第三方依赖。
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

from .fonts import (
    available_font,
    describe_font_setup,
    load_font as _load_font,
)
from .fonts import (
    text_width as _text_width,
)
from .model import Atom, Bond, Group, Node, render_ascii


class RenderError(RuntimeError):
    """渲染失败。"""


@dataclass
class RenderOptions:
    """渲染参数。"""

    font_size: int = 56
    """主字号（像素）。下标会按 :attr:`subscript_scale` 缩小。"""

    color: tuple[int, int, int] = (17, 24, 39)
    """公式颜色。"""

    background: tuple[int, int, int] | None = (255, 255, 255)
    """底色；``None`` 表示透明背景。"""

    padding: int = 28
    """四周留白。"""

    supersample: int = 3
    """超采样倍数，先放大绘制再缩回，用于抗锯齿。"""

    subscript_scale: float = 0.62
    """下标字号相对主字号的比例。"""

    subscript_drop: float = 0.18
    """下标基线相对主字号下移的比例。"""

    superscript_rise: float = 0.42
    """上标（电荷）基线相对主字号上移的比例。"""

    title: str | None = None
    """顶部小标题，通常放物质名称。"""

    subtitle: str | None = None
    """底部小字，通常放分子式与相对分子质量。"""

    caption_size_ratio: float = 0.42
    """标题/小字相对主字号的比例。"""

    accents: tuple[tuple[int, int, int], ...] = field(
        default_factory=lambda: ((37, 99, 235), (220, 38, 38)),
    )
    """标题与小字的颜色。"""


# ---------------------------------------------------------------- 排版片段


@dataclass(frozen=True)
class Run:
    """一段文字及其相对基线位置。

    Attributes:
        text: 要绘制的文字。
        level: ``0`` 基线、``-1`` 下标、``+1`` 上标。
    """

    text: str
    level: int = 0


def _count_runs(count: int | str) -> list[Run]:
    """把下标渲染成数字/字母片段。"""
    if count == 1:
        return []
    return [Run(str(count), -1)]


def build_runs(nodes: tuple[Node, ...]) -> list[Run]:
    """把简式 AST 展开成一串排版片段。"""
    runs: list[Run] = []
    for node in nodes:
        if isinstance(node, Atom):
            runs.append(Run(node.symbol, 0))
            runs.extend(_count_runs(node.count))
            if node.charge:
                sign = "+" if node.charge > 0 else "-"
                magnitude = abs(node.charge)
                text = sign if magnitude == 1 else f"{magnitude}{sign}"
                runs.append(Run(text, 1))
        elif isinstance(node, Group):
            if node.prefix and isinstance(node.count, int) and node.count > 1:
                runs.append(Run(str(node.count), 0))
                runs.extend(build_runs(node.items))
                continue
            runs.append(Run("(", 0))
            runs.extend(build_runs(node.items))
            runs.append(Run(")", 0))
            runs.extend(_count_runs(node.count))
        else:
            runs.append(Run(node.symbol, 0))
    return runs


# ---------------------------------------------------------------- 绘制实现


def render_condensed_png(
    nodes: tuple[Node, ...],
    options: RenderOptions | None = None,
) -> bytes:
    """把结构简式渲染成 PNG 字节流。

    Args:
        nodes: 简式 AST。
        options: 渲染参数。

    Returns:
        PNG 文件内容。

    Raises:
        RenderError: Pillow 不可用或排版失败时抛出。
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:  # pragma: no cover - 依赖缺失路径
        raise RenderError(
            "缺少 pillow 依赖，无法渲染图片。请执行 pip install pillow，"
            "或把 image.enabled 关掉改用文本输出。",
        ) from error

    if not nodes:
        raise RenderError("没有可渲染的简式内容")

    opts = options or RenderOptions()
    scale = max(1, opts.supersample)
    base_size = max(8, opts.font_size) * scale
    sub_size = max(6, int(base_size * opts.subscript_scale))
    caption_size = max(6, int(base_size * opts.caption_size_ratio))

    base_font = _load_font(base_size)
    sub_font = _load_font(sub_size)
    caption_font = _load_font(caption_size)

    runs = build_runs(nodes)
    if not runs:
        raise RenderError("没有可渲染的简式内容")

    ascent, descent = base_font.getmetrics()
    sub_ascent = sub_font.getmetrics()[0]
    line_height = ascent + descent

    # Pillow 的 draw.text 以"上沿"为锚点，所以要按字体实际 ascent 反推偏移，
    # 才能让下标真正落在主基线下方、上标落在主基线上方。
    drop = int(base_size * opts.subscript_drop)
    rise = int(base_size * opts.superscript_rise)
    level_offset = {
        0: 0,
        -1: ascent - sub_ascent + drop,
        1: ascent - sub_ascent - rise,
    }

    # 量出每一段的宽度与偏移
    measured: list[tuple[Run, int, int]] = []
    total_width = 0
    min_top = 0
    max_bottom = line_height
    for run in runs:
        font = base_font if run.level == 0 else sub_font
        offset = level_offset[run.level]

        width = _text_width(font, run.text)
        # 用字体的 bbox 估算这一段实际占用的上下范围
        bbox = font.getbbox(run.text)
        min_top = min(min_top, offset + bbox[1])
        max_bottom = max(max_bottom, offset + bbox[3])
        measured.append((run, total_width, offset))
        total_width += width

    caption_gap = int(base_size * 0.35) if opts.subtitle else 0
    title_gap = int(base_size * 0.30) if opts.title else 0
    title_height = (caption_font.getmetrics()[0] + caption_font.getmetrics()[1]
                    if opts.title else 0)
    subtitle_height = (
        caption_font.getmetrics()[0] + caption_font.getmetrics()[1]
        if opts.subtitle else 0
    )

    formula_height = max_bottom - min_top
    content_width = max(
        total_width,
        _text_width(caption_font, opts.title) if opts.title else 0,
        _text_width(caption_font, opts.subtitle) if opts.subtitle else 0,
    )

    pad = max(0, opts.padding) * scale
    width = content_width + pad * 2
    height = formula_height + pad * 2 + title_height + title_gap \
        + caption_gap + subtitle_height

    background = opts.background
    if background is None:
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    else:
        canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)

    cursor_y = pad

    # 顶部标题
    if opts.title:
        title_color = opts.accents[0] if opts.accents else opts.color
        draw.text(
            ((width - _text_width(caption_font, opts.title)) // 2, cursor_y),
            opts.title,
            font=caption_font,
            fill=title_color,
        )
        cursor_y += title_height + title_gap

    # 主体公式
    baseline = cursor_y - min_top
    cursor_x = (width - total_width) // 2
    for run, x, offset in measured:
        font = base_font if run.level == 0 else sub_font
        draw.text((cursor_x + x, baseline + offset), run.text, font=font,
                  fill=opts.color)
    cursor_y += formula_height + caption_gap

    # 底部小字
    if opts.subtitle:
        subtitle_color = opts.accents[1] if len(opts.accents) > 1 else opts.color
        draw.text(
            ((width - _text_width(caption_font, opts.subtitle)) // 2, cursor_y),
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


def render_condensed_text_png(
    text: str,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    options: RenderOptions | None = None,
) -> bytes:
    """便捷入口：直接把纯文本式子（如 ``CH3CH2OH``）渲染成 PNG。"""
    from .text_formula import parse_condensed

    parsed = parse_condensed(text)
    opts = options or RenderOptions()
    if title is not None:
        opts.title = title
    if subtitle is not None:
        opts.subtitle = subtitle
    return render_condensed_png(parsed.nodes, opts)


#: 兼容旧调用点：字体相关的实现已搬到 :mod:`chem.fonts`。
__all__ = [
    "RenderError",
    "RenderOptions",
    "Run",
    "ascii_summary",
    "available_font",
    "build_runs",
    "describe_font_setup",
    "render_condensed_png",
    "render_condensed_text_png",
]


def ascii_summary(nodes: tuple[Node, ...]) -> str:
    """返回纯 ASCII 式子，便于日志。"""
    return render_ascii(nodes)
