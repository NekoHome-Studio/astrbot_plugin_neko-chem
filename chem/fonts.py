"""跨平台字体探测与文字测量。

结构简式渲染（:mod:`chem.render_png`）与键线式渲染
(:mod:`chem.render_skeletal`）共用这里的逻辑：按平台常见路径依次找可用的
TTF，找不到就退回 Pillow 内置字体，并对外说明用的是哪一个。
"""

from __future__ import annotations

import os
from functools import lru_cache

#: 字体搜索顺序：按平台常见路径依次尝试，找到第一个可用的就用。
FONT_CANDIDATES: tuple[str, ...] = (
    # Windows
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    # 随插件分发的字体（可选）
    os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf"),
)

BUILTIN_FONT_LABEL = "<Pillow 内置字体>"


@lru_cache(maxsize=32)
def load_font(size: int):
    """按字号加载字体；找不到任何 TTF 时回退到 Pillow 内置位图字体。"""
    from PIL import ImageFont

    for path in FONT_CANDIDATES:
        if not path or not os.path.isfile(path):
            continue
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 不支持 size 参数
        return ImageFont.load_default()


def text_width(font, text: str | None) -> int:
    """测量文字宽度。"""
    if not text:
        return 0
    try:
        return int(font.getlength(text))
    except AttributeError:  # pragma: no cover - 老版本 Pillow 兜底
        return font.getsize(text)[0]


def available_font() -> str:
    """返回实际使用的字体路径，供日志与自检使用。"""
    for path in FONT_CANDIDATES:
        if path and os.path.isfile(path):
            return path
    return BUILTIN_FONT_LABEL


def describe_font_setup() -> str:
    """返回字体环境说明，供 ``/化学 状态`` 使用。"""
    path = available_font()
    if path == BUILTIN_FONT_LABEL:
        return "未找到常见 TTF 字体，将使用 Pillow 内置字体（上下标效果一般）"
    return f"使用字体：{path}"


def pillow_version() -> str:
    """返回 Pillow 版本号，不可用时返回 ``未安装``。"""
    try:
        import PIL  # noqa: PLC0415 - 可选依赖

        return str(PIL.__version__)
    except Exception:  # pragma: no cover - 依赖缺失路径
        return "未安装"
