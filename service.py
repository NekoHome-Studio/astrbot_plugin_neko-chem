"""插件侧服务层：读取配置、渲染图片、管理图片缓存。

这一层是"化学核心库"与"AstrBot 插件层"之间的适配层：

* 把 AstrBot 的 ``AstrBotConfig`` 转成带默认值的强类型配置对象；
* 调用 :mod:`chem.render_png` / :mod:`chem.render_rdkit` 出图；
* 把图片落到插件数据目录下的缓存目录，并按数量/时间淘汰旧文件。

配置读取刻意写成"逐层取值 + 兜底默认值"，这样即使 WebUI 保存的配置
缺字段、类型不对，插件也不会因为 ``KeyError`` 崩掉。
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .chem import Compound
from .chem.layout import Layout, compute_layout
from .chem.render_png import RenderError as PngRenderError
from .chem.render_png import RenderOptions, render_condensed_png
from .chem.render_rdkit import (
    RenderError as StructureRenderError,
)
from .chem.render_rdkit import (
    StructureOptions,
    rdkit_available,
    render_structure_safe,
)
from .chem.render_skeletal import (
    SkeletalOptions,
    SkeletalRenderError,
    render_skeletal_png,
)

PLUGIN_NAME = "astrbot_plugin_chem_structure"

DEFAULT_CACHE_DIR_NAME = "cache"


# ---------------------------------------------------------------- 配置模型


def _lookup(config: Mapping[str, Any] | None, path: str, default: Any) -> Any:
    """按 ``"a.b.c"`` 逐层取值，任何一层缺失或类型不符都返回默认值。"""
    if not isinstance(config, Mapping):
        return default
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current if current is not None else default


def _as_bool(config: Mapping[str, Any] | None, path: str, default: bool) -> bool:
    """读取布尔配置。"""
    value = _lookup(config, path, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "是", "开启"}
    return bool(value)


def _as_int(
    config: Mapping[str, Any] | None,
    path: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """读取整数配置并夹到合理区间。"""
    value = _lookup(config, path, default)
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _as_color(
    config: Mapping[str, Any] | None,
    path: str,
    default: tuple[int, int, int],
) -> tuple[int, int, int]:
    """读取 ``#RRGGBB`` 形式的颜色。"""
    value = _lookup(config, path, None)
    if isinstance(value, str):
        text = value.strip().lstrip("#")
        if len(text) == 6:
            try:
                return (
                    int(text[0:2], 16),
                    int(text[2:4], 16),
                    int(text[4:6], 16),
                )
            except ValueError:
                pass
    return default


@dataclass
class ImageSettings:
    """结构简式图片的渲染设置。"""

    enabled: bool = True
    font_size: int = 56
    padding: int = 28
    supersample: int = 3
    transparent: bool = False
    color: tuple[int, int, int] = (17, 24, 39)
    show_title: bool = True
    show_caption: bool = True


@dataclass
class StructureSettings:
    """结构式绘图的设置。"""

    enabled: bool = True
    width: int = 900
    height: int = 700


@dataclass
class SkeletalSettings:
    """自研键线式绘图的设置。"""

    enabled: bool = True
    bond_length: int = 54
    line_width: float = 2.6
    transparent: bool = False
    show_hydrogens: bool = True
    use_rdkit: bool = True
    """RDKit 可用时是否优先用 RDKit 画 /化学 结构式。"""
    prefer_for_cyclic: bool = True
    """环状物质是否优先出键线式（而不是信息量很低的连写式简式）。"""


@dataclass
class OutputSettings:
    """文本输出的设置。"""

    with_text: bool = True
    show_smiles: bool = True
    show_percent: bool = True
    max_notes: int = 3


@dataclass
class InputSettings:
    """输入解析的设置。"""

    enable_abbrev: bool = True
    aggressive_abbrev: bool = False


@dataclass
class CacheSettings:
    """图片缓存的设置。"""

    enabled: bool = True
    max_files: int = 300
    ttl_hours: int = 24


@dataclass
class ChemSettings:
    """插件全部设置的聚合。"""

    image: ImageSettings = field(default_factory=ImageSettings)
    structure: StructureSettings = field(default_factory=StructureSettings)
    skeletal: SkeletalSettings = field(default_factory=SkeletalSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    input: InputSettings = field(default_factory=InputSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)


def load_settings(config: Mapping[str, Any] | None) -> ChemSettings:
    """从 AstrBot 配置对象构造强类型设置。"""
    return ChemSettings(
        image=ImageSettings(
            enabled=_as_bool(config, "image.enabled", True),
            font_size=_as_int(config, "image.font_size", 56, 16, 160),
            padding=_as_int(config, "image.padding", 28, 0, 200),
            supersample=_as_int(config, "image.supersample", 3, 1, 4),
            transparent=_as_bool(config, "image.transparent", False),
            color=_as_color(config, "image.color", (17, 24, 39)),
            show_title=_as_bool(config, "image.show_title", True),
            show_caption=_as_bool(config, "image.show_caption", True),
        ),
        structure=StructureSettings(
            enabled=_as_bool(config, "structure.enabled", True),
            width=_as_int(config, "structure.width", 900, 200, 3000),
            height=_as_int(config, "structure.height", 700, 200, 3000),
        ),
        skeletal=SkeletalSettings(
            enabled=_as_bool(config, "skeletal.enabled", True),
            bond_length=_as_int(config, "skeletal.bond_length", 54, 20, 160),
            line_width=max(
                0.5, min(8.0, float(_lookup(config, "skeletal.line_width", 2.6) or 2.6)),
            ),
            transparent=_as_bool(config, "skeletal.transparent", False),
            show_hydrogens=_as_bool(config, "skeletal.show_hydrogens", True),
            use_rdkit=_as_bool(config, "skeletal.use_rdkit", True),
            prefer_for_cyclic=_as_bool(config, "skeletal.prefer_for_cyclic", True),
        ),
        output=OutputSettings(
            with_text=_as_bool(config, "output.with_text", True),
            show_smiles=_as_bool(config, "output.show_smiles", True),
            show_percent=_as_bool(config, "output.show_percent", True),
            max_notes=_as_int(config, "output.max_notes", 3, 0, 10),
        ),
        input=InputSettings(
            enable_abbrev=_as_bool(config, "input.enable_abbrev", True),
            aggressive_abbrev=_as_bool(config, "input.aggressive_abbrev", False),
        ),
        cache=CacheSettings(
            enabled=_as_bool(config, "cache.enabled", True),
            max_files=_as_int(config, "cache.max_files", 300, 0, 5000),
            ttl_hours=_as_int(config, "cache.ttl_hours", 24, 0, 24 * 30),
        ),
    )


# ---------------------------------------------------------------- 缓存目录


def resolve_cache_dir() -> Path:
    """确定图片缓存目录。

    优先放在 AstrBot 的 ``data/plugin_data/<插件名>`` 下（符合"持久化
    数据放 data 目录"的规范）；拿不到 AstrBot 路径时退回系统临时目录。
    """
    try:
        from astrbot.core.utils.astrbot_path import (  # noqa: PLC0415 - 可选依赖
            get_astrbot_plugin_data_path,
        )

        base = Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME
    except Exception:
        base = Path(tempfile.gettempdir()) / PLUGIN_NAME
    directory = base / DEFAULT_CACHE_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------------------------------------------------------- 渲染服务


class RenderService:
    """负责出图与缓存。"""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.settings = load_settings(config)
        self.cache_dir = resolve_cache_dir()
        self._last_cleanup = 0.0
        self._layout_cache: dict[str, Layout] = {}
        self.render_count = 0
        self.structure_count = 0
        self.skeletal_count = 0
        self.error_count = 0

    # -------------------------------------------------------- 设置与统计

    def refresh(self, config: Mapping[str, Any] | None) -> None:
        """配置热更新后重新读取设置。"""
        self.settings = load_settings(config)

    def cache_stats(self) -> dict[str, Any]:
        """当前缓存状态。"""
        if not self.cache_dir.is_dir():
            return {"dir": str(self.cache_dir), "files": 0, "bytes": 0}
        files = list(self.cache_dir.glob("*.png"))
        total = 0
        for path in files:
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return {"dir": str(self.cache_dir), "files": len(files), "bytes": total}

    # -------------------------------------------------------- 缓存维护

    def cleanup_cache(self, *, force: bool = False) -> int:
        """按数量和有效期清理缓存，返回删除的文件数。"""
        cache = self.settings.cache
        if not cache.enabled and not force:
            return 0
        now = time.time()
        # 最多每 5 分钟清理一次，避免每次渲染都扫目录
        if not force and now - self._last_cleanup < 300:
            return 0
        self._last_cleanup = now

        if not self.cache_dir.is_dir():
            return 0
        entries: list[tuple[float, int, Path]] = []
        for path in self.cache_dir.glob("*.png"):
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((stat.st_mtime, stat.st_size, path))

        removed = 0
        ttl_seconds = cache.ttl_hours * 3600
        if ttl_seconds > 0:
            for mtime, _, path in entries:
                if now - mtime > ttl_seconds:
                    try:
                        path.unlink()
                        removed += 1
                    except OSError:
                        pass
            entries = [item for item in entries if item[2].exists()]

        if cache.max_files > 0 and len(entries) > cache.max_files:
            entries.sort(key=lambda item: item[0])
            for _, _, path in entries[: len(entries) - cache.max_files]:
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.png"

    def _write_cached(self, key: str, data: bytes) -> Path:
        """写入缓存文件；写失败时退回临时文件，保证功能可用。"""
        cache = self.settings.cache
        if cache.enabled:
            try:
                path = self._cache_path(key)
                path.write_bytes(data)
                self.cleanup_cache()
                return path
            except OSError:
                pass
        handle = tempfile.NamedTemporaryFile(
            prefix="chem_", suffix=".png", delete=False,
        )
        with handle:
            handle.write(data)
        return Path(handle.name)

    @staticmethod
    def _digest(*parts: Any) -> str:
        """生成稳定的内容摘要，用作缓存文件名。"""
        hasher = hashlib.sha1()
        for part in parts:
            hasher.update(repr(part).encode("utf-8"))
            hasher.update(b"\x1f")
        return hasher.hexdigest()[:32]

    # -------------------------------------------------------- 结构简式图

    def condensed_options(
        self,
        *,
        title: str | None = None,
        caption: str | None = None,
    ) -> RenderOptions:
        """根据设置构造简式渲染参数。"""
        image = self.settings.image
        return RenderOptions(
            font_size=image.font_size,
            color=image.color,
            background=None if image.transparent else (255, 255, 255),
            padding=image.padding,
            supersample=image.supersample,
            title=title if image.show_title else None,
            subtitle=caption if image.show_caption else None,
        )

    def render_condensed(
        self,
        compound: Compound,
        *,
        title: str | None = None,
        caption: str | None = None,
    ) -> Path:
        """渲染结构简式图片并返回文件路径。

        Raises:
            PngRenderError: 简式为空或 Pillow 不可用时抛出。
        """
        nodes = compound.condensed_nodes
        if not nodes:
            raise PngRenderError("该物质没有可绘制的结构简式")
        options = self.condensed_options(title=title, caption=caption)
        key = self._digest(
            "condensed",
            compound.condensed_text,
            options.font_size,
            options.color,
            options.background,
            options.padding,
            options.supersample,
            options.title,
            options.subtitle,
        )
        cached = self._cache_path(key)
        if self.settings.cache.enabled and cached.is_file():
            try:
                if cached.stat().st_size > 0:
                    self.render_count += 1
                    return cached
            except OSError:
                pass
        data = render_condensed_png(nodes, options)
        self.render_count += 1
        return self._write_cached(key, data)

    def render_condensed_png_bytes(
        self,
        compound: Compound,
        options: RenderOptions,
    ) -> bytes:
        """直接返回简式图片字节流（给 Web API 用，不落盘缓存）。

        Raises:
            PngRenderError: 简式为空或 Pillow 不可用时抛出。
        """
        nodes = compound.condensed_nodes
        if not nodes:
            raise PngRenderError("该物质没有可绘制的结构简式")
        data = render_condensed_png(nodes, options)
        self.render_count += 1
        return data

    # -------------------------------------------------------- 结构式图

    def structure_available(self) -> bool:
        """当前是否真的能用 RDKit 画结构式。"""
        return self.settings.structure.enabled and rdkit_available()

    # -------------------------------------------------------- 键线式（自研）

    def layout_of(self, compound: Compound) -> Layout | None:
        """取分子图的二维排版结果，带一层缓存。

        排版要跑松弛迭代，对复杂分子不算便宜；同一分子反复请求时应复用。
        """
        if compound.molecule is None:
            return None
        key = compound.smiles or compound.condensed_text or compound.input_text
        cached = self._layout_cache.get(key)
        if cached is not None:
            return cached
        layout = compute_layout(compound.molecule)
        if len(self._layout_cache) >= 64:
            self._layout_cache.clear()
        self._layout_cache[key] = layout
        return layout

    def skeletal_options(
        self,
        *,
        title: str | None = None,
        caption: str | None = None,
    ) -> SkeletalOptions:
        """根据设置构造键线式渲染参数。"""
        skeletal = self.settings.skeletal
        return SkeletalOptions(
            bond_length=skeletal.bond_length,
            line_width=skeletal.line_width,
            background=None if skeletal.transparent else (255, 255, 255),
            show_hetero_hydrogens=skeletal.show_hydrogens,
            title=title,
            subtitle=caption,
        )

    def render_skeletal(
        self,
        compound: Compound,
        *,
        title: str | None = None,
        caption: str | None = None,
    ) -> tuple[Path | None, str]:
        """画键线式，返回 ``(路径或 None, 失败原因)``。"""
        if not self.settings.skeletal.enabled:
            return None, "键线式绘图已在配置中关闭（skeletal.enabled）"
        if compound.molecule is None:
            return None, (
                "该输入只有元素组成（简式或分子式），没有键连信息，"
                "画不出键线式；请给出来源名称或 SMILES"
            )

        layout = self.layout_of(compound)
        if layout is None:
            return None, "无法为该分子生成二维坐标"
        if not layout.drawable:
            return None, (
                "该分子的环系无法在平面上排成规整图形（"
                + "；".join(layout.issues)
                + "），为避免画出错误结构已放弃绘制"
            )

        options = self.skeletal_options(title=title, caption=caption)
        key = self._digest(
            "skeletal",
            compound.molecule.smiles,
            options.bond_length,
            options.line_width,
            options.background,
            options.show_hetero_hydrogens,
            options.title,
            options.subtitle,
        )
        cached = self._cache_path(key)
        if self.settings.cache.enabled and cached.is_file():
            try:
                if cached.stat().st_size > 0:
                    self.skeletal_count += 1
                    return cached, ""
            except OSError:
                pass

        try:
            data = render_skeletal_png(compound.molecule, layout, options)
        except SkeletalRenderError as error:
            self.error_count += 1
            return None, str(error)
        except Exception as error:  # pragma: no cover - 兜底
            self.error_count += 1
            return None, f"键线式渲染异常：{error}"

        self.skeletal_count += 1
        return self._write_cached(key, data), ""

    def render_skeletal_png_bytes(
        self,
        compound: Compound,
        options: SkeletalOptions,
    ) -> bytes:
        """直接返回键线式 PNG 字节流（给 Web API 用）。

        Raises:
            SkeletalRenderError: 排版不可用或渲染失败时抛出。
        """
        layout = self.layout_of(compound)
        if layout is None or compound.molecule is None:
            raise SkeletalRenderError("该输入没有可用于绘制键线式的结构信息")
        data = render_skeletal_png(compound.molecule, layout, options)
        self.skeletal_count += 1
        return data

    def render_structure(self, smiles: str) -> tuple[Path | None, str]:
        """渲染结构式图片，返回 ``(路径或 None, 失败原因)``。"""
        if not self.settings.structure.enabled:
            return None, "结构式绘图已在配置中关闭（structure.enabled）"
        if not rdkit_available():
            from .chem.render_rdkit import rdkit_unavailable_reason

            return None, (
                "未安装 RDKit，无法绘制结构式（pip install rdkit 后重启即可）"
                + (
                    f"；当前原因：{rdkit_unavailable_reason()}"
                    if rdkit_unavailable_reason()
                    else ""
                )
            )

        options = StructureOptions(
            width=self.settings.structure.width,
            height=self.settings.structure.height,
        )
        key = self._digest("structure", smiles, options.width, options.height)
        cached = self._cache_path(key)
        if self.settings.cache.enabled and cached.is_file():
            try:
                if cached.stat().st_size > 0:
                    self.structure_count += 1
                    return cached, ""
            except OSError:
                pass

        data, reason = render_structure_safe(smiles, options)
        if data is None:
            return None, reason
        self.structure_count += 1
        return self._write_cached(key, data), ""

    # -------------------------------------------------------- 自检

    def environment_report(self) -> dict[str, Any]:
        """收集环境自检信息。"""
        from .chem.fonts import available_font, describe_font_setup, pillow_version
        from .chem.render_rdkit import rdkit_unavailable_reason, rdkit_version

        return {
            "pillow": pillow_version(),
            "font": available_font(),
            "font_note": describe_font_setup(),
            "rdkit_available": rdkit_available(),
            "rdkit_version": rdkit_version(),
            "rdkit_reason": rdkit_unavailable_reason(),
            "skeletal_enabled": self.settings.skeletal.enabled,
            "cache": self.cache_stats(),
            "render_count": self.render_count,
            "structure_count": self.structure_count,
            "skeletal_count": self.skeletal_count,
            "error_count": self.error_count,
        }


__all__ = [
    "CacheSettings",
    "ChemSettings",
    "ImageSettings",
    "InputSettings",
    "OutputSettings",
    "PLUGIN_NAME",
    "RenderService",
    "SkeletalSettings",
    "StructureRenderError",
    "StructureSettings",
    "load_settings",
    "resolve_cache_dir",
]
