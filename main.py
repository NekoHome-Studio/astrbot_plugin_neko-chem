"""AstrBot 插件入口：画化学结构简式。

支持把中文名/俗名、SMILES、结构简式、分子式四类输入画成图片并给出
分子式、相对分子质量、元素质量分数等信息。

指令一览::

    /化学 结构   乙醇          综合输出（结构简式图片 + 文字信息）
    /化学 简式   CH3CH2OH      只输出结构简式
    /化学 分子式 C2H6O         分子式、相对分子质量、元素质量分数
    /化学 结构式 乙醇          用 RDKit 画结构式（装了 RDKit 才可用）
    /化学 列表   [关键词]      浏览内置物质词表
    /化学 状态                 环境自检（字体、RDKit、缓存）
    /化学 帮助                 使用说明

    结构 乙醇                  顶层快捷方式，等价于 "/化学 结构 乙醇"

注意
----
本模块**不能**加 ``from __future__ import annotations``：那会把类型注解
变成字符串，而 AstrBot 判断贪婪参数靠的是 ``annotation is GreedyStr``，
一旦字符串化，``GreedyStr`` 就会失效，带空格的输入会被截断。
"""

import asyncio
import time
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Plain
from astrbot.api.star import Context, Star
from astrbot.api.web import error_response, json_response, request
from astrbot.core.star.filter.command import GreedyStr

from .chem import (
    Compound,
    ResolveError,
    categories,
    decorate_formula,
    entry_count,
    resolve,
    search_names,
)
from .chem.render_png import RenderError as PngRenderError
from .chem.render_png import RenderOptions  # noqa: F401 - 供类型提示
from .chem.render_rdkit import rdkit_available
from .service import PLUGIN_NAME, ChemSettings, RenderService

PLUGIN_VERSION = "1.0.0"
LOG_PREFIX = "[chem]"

#: 单条回复的文本长度上限，超过就截断，避免刷屏。
MAX_TEXT_LENGTH = 1500


class ChemStructurePlugin(Star):
    """把化学结构简式画成图片的 AstrBot 插件。"""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = config
        self.service = RenderService(config)
        self._lock = asyncio.Semaphore(2)
        self._started_at = time.time()
        self._register_web_apis()
        logger.info(
            "%s 插件已加载，内置物质 %d 条，缓存目录 %s",
            LOG_PREFIX,
            entry_count(),
            self.service.cache_dir,
        )

    # ------------------------------------------------------------ 生命周期

    async def initialize(self) -> None:
        """插件激活后做一次自检与环境清理。"""
        report = self.service.environment_report()
        logger.info(
            "%s 环境自检：pillow=%s，rdkit=%s，字体=%s",
            LOG_PREFIX,
            report["pillow"],
            report["rdkit_version"] or "不可用",
            report["font"],
        )
        if not report["rdkit_available"]:
            logger.info(
                "%s 未安装 RDKit，将只输出结构简式图片（如需结构式请 "
                "pip install rdkit）",
                LOG_PREFIX,
            )
        removed = self.service.cleanup_cache(force=True)
        if removed:
            logger.info("%s 已清理 %d 个过期图片缓存", LOG_PREFIX, removed)

    async def terminate(self) -> None:
        """插件卸载时清理缓存，释放磁盘。"""
        removed = self.service.cleanup_cache(force=True)
        logger.info("%s 插件已卸载，清理缓存文件 %d 个", LOG_PREFIX, removed)

    @filter.on_astrbot_loaded()
    async def on_loaded(self) -> None:
        """AstrBot 初始化完成后输出一次能力摘要。"""
        logger.info(
            "%s 就绪：/化学 结构 乙醇 可以开始使用", LOG_PREFIX,
        )

    # ------------------------------------------------------------ 配置工具

    @property
    def settings(self) -> ChemSettings:
        """当前生效的设置（每次读取时与配置对象同步，支持热更新）。"""
        self.service.refresh(self.config)
        return self.service.settings

    # ------------------------------------------------------------ 文本组装

    def _format_compound(
        self,
        compound: Compound,
        *,
        detailed: bool = True,
        prefer_skeletal: bool = False,
    ) -> str:
        """把物质信息整理成聊天文本。

        ``prefer_skeletal`` 为真时说明这张图会用键线式表示，此时若"结构简式"
        只是自动推导出来的分子式（苯 → C6H6），就不必再列一行误导性的
        "结构简式"，改成一句说明。
        """
        settings = self.settings
        lines: list[str] = []

        title = compound.display_name
        if compound.category:
            title = f"{title} · {compound.category}"
        lines.append(title)

        shows_real_condensed = (
            compound.condensed_source == "name-db"
            or not prefer_skeletal
        )
        if compound.condensed_nodes:
            if shows_real_condensed:
                lines.append(f"结构简式：{compound.condensed_unicode}")
                if compound.condensed_source == "name-db":
                    lines.append("（教材通用写法）")
            else:
                lines.append("环状结构：用键线式表示（连写式简式无法表达成环）")

        if compound.formula:
            formula_text = decorate_formula(compound.formula)
            mass = compound.molar_mass
            if mass is not None:
                lines.append(f"分子式：{formula_text}　相对分子质量：{mass:.2f}")
            else:
                lines.append(f"最简式：{formula_text}")

        if detailed and settings.output.show_percent and compound.breakdown:
            pieces = [
                f"{symbol} {fraction * 100:.1f}%"
                for symbol, _, fraction in compound.breakdown
            ]
            lines.append("元素质量分数：" + " · ".join(pieces))

        if detailed and settings.output.show_smiles and compound.smiles:
            lines.append(f"SMILES：{compound.smiles}")

        if detailed:
            lines.append(f"输入识别为：{compound.source_label}")

        if compound.candidates:
            lines.append("同分异构体：" + "、".join(compound.candidates))

        for note in compound.notes[: settings.output.max_notes]:
            lines.append(f"提示：{note}")

        text = "\n".join(lines)
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH] + "\n…（内容过长已截断）"
        return text

    def _format_error(self, error: ResolveError) -> str:
        """把解析失败整理成友好的提示。"""
        lines = [f"没能识别「{error.text}」。", error.reason]
        if error.suggestions:
            lines.append("你是不是想找：" + "、".join(error.suggestions[:8]))
        if error.details:
            for detail in error.details[:3]:
                lines.append("· " + detail.splitlines()[0])
        lines.append(
            "支持：中文名/俗名（乙醇、甘油）、SMILES（CCO）、"
            "结构简式（CH3CH2OH）、分子式（C2H6O）。",
        )
        return "\n".join(lines)

    # ------------------------------------------------------------ 出图

    def _caption(self, compound: Compound) -> str:
        """图片底部小字：分子式与相对分子质量。"""
        pieces: list[str] = []
        if compound.formula:
            pieces.append(decorate_formula(compound.formula))
        if compound.name:
            pieces.append(compound.name)
        mass = compound.molar_mass
        if mass is not None:
            pieces.append(f"M = {mass:.2f}")
        return "　".join(pieces)

    def _track(self, event: AstrMessageEvent, path: Path | None) -> None:
        """把临时图片登记给 AstrBot，发送完成后自动删除。

        只对不在缓存目录里的文件登记——缓存文件要复用，不能删。
        """
        if path is None:
            return
        try:
            if path.parent.resolve() == self.service.cache_dir.resolve():
                return
            event.track_temporary_local_file(str(path))
        except OSError:
            event.track_temporary_local_file(str(path))

    def _condensed_image(
        self,
        event: AstrMessageEvent,
        compound: Compound,
    ) -> Path | None:
        """渲染简式图片，失败时记录日志并返回 ``None``。"""
        settings = self.settings
        if not settings.image.enabled:
            return None
        if not compound.condensed_nodes:
            return None
        title = compound.display_name if compound.name else None
        try:
            path = self.service.render_condensed(
                compound, title=title, caption=self._caption(compound),
            )
        except PngRenderError as error:
            logger.warning("%s 简式渲染失败：%s", LOG_PREFIX, error)
            self.service.error_count += 1
            return None
        except Exception as error:  # pragma: no cover - 兜底，避免拖垮插件
            logger.exception("%s 简式渲染出现异常：%s", LOG_PREFIX, error)
            self.service.error_count += 1
            return None
        self._track(event, path)
        return path

    def _skeletal_image(
        self,
        event: AstrMessageEvent,
        compound: Compound,
    ) -> tuple[Path | None, str]:
        """渲染自研键线式图片，返回 ``(路径或 None, 失败原因)``。"""
        title = compound.display_name if compound.name else None
        path, reason = self.service.render_skeletal(
            compound, title=title, caption=self._caption(compound),
        )
        self._track(event, path)
        return path, reason

    def _structure_image(
        self,
        event: AstrMessageEvent,
        compound: Compound,
    ) -> tuple[Path | None, str, str]:
        """画"结构式"，返回 ``(路径或 None, 失败原因, 实际用的引擎)``。

        优先用 RDKit（画质与通用性最好），没装或失败时退回自研的纯 Python
        键线式引擎 —— 这样在没装 RDKit 的环境里 `结构式` 也依然能用。
        """
        settings = self.settings
        if compound.molecule is None:
            return None, (
                "该输入只有元素组成（简式或分子式），没有键连信息，"
                "画不出结构式；请给出来源名称或 SMILES"
            ), ""
        if not settings.structure.enabled:
            return None, "结构式绘图已在配置中关闭（structure.enabled）", ""

        reasons: list[str] = []
        if settings.skeletal.use_rdkit and rdkit_available() and compound.smiles:
            path, reason = self.service.render_structure(compound.smiles)
            if path is not None:
                self._track(event, path)
                return path, "", "RDKit"
            reasons.append(f"RDKit：{reason}")

        path, reason = self._skeletal_image(event, compound)
        if path is not None:
            return path, "", "内置键线式引擎"
        reasons.append(f"内置键线式引擎：{reason}")
        return None, "；".join(reasons), ""

    # ------------------------------------------------------------ 指令实现

    def _query_of(self, query: GreedyStr) -> str:
        """把 GreedyStr 参数（可能是哨兵类本身）转成普通字符串。"""
        if query is GreedyStr or query is None:
            return ""
        return str(query).strip()

    async def _resolve(self, event: AstrMessageEvent, query: str):
        """解析用户输入；失败时直接返回错误文本。"""
        settings = self.settings
        return resolve(
            query,
            enable_abbrev=settings.input.enable_abbrev,
            aggressive_abbrev=settings.input.aggressive_abbrev,
        )

    def _usage_text(self) -> str:
        """没有给出物质时的用法提示。"""
        return (
            "请告诉我要画什么物质，例如：\n"
            "· /化学 结构 乙醇\n"
            "· /化学 结构 葡萄糖\n"
            "· /化学 结构 CCO\n"
            "· /化学 结构 CH3CH(CH3)COOH\n"
            "· /化学 结构 C2H6O\n"
            "也可以直接输入 /结构 乙醇，或发送 /化学 帮助 查看全部指令。"
        )

    @filter.command_group("化学", alias={"chem"})
    def chem_group(self):
        """化学结构简式助手。"""

    @chem_group.command("结构", alias={"画"})
    async def cmd_structure(
        self, event: AstrMessageEvent, query: GreedyStr = GreedyStr,
    ):
        """绘制结构简式。参数: 物质名称/SMILES/结构简式/分子式"""
        event.stop_event()
        text = self._query_of(query)
        if not text:
            yield event.plain_result(self._usage_text())
            return
        async with self._lock:
            try:
                compound = await self._resolve(event, text)
            except ResolveError as error:
                yield event.plain_result(self._format_error(error))
                return

            chain = []
            settings = self.settings

            # 环状物质的"连写式简式"往往就是分子式本身（苯 → C6H6），
            # 信息量很低；这时候键线式才是教材里的画法，优先出键线式。
            prefer_skeletal = (
                settings.skeletal.enabled
                and settings.skeletal.prefer_for_cyclic
                and compound.is_cyclic
                and compound.condensed_source != "name-db"
                and compound.molecule is not None
            )

            if settings.output.with_text:
                chain.append(
                    Plain(
                        self._format_compound(
                            compound, prefer_skeletal=prefer_skeletal,
                        ),
                    ),
                )

            image_path: Path | None = None
            skeletal_path: Path | None = None
            skeletal_reason = ""
            if prefer_skeletal:
                skeletal_path, skeletal_reason = self._skeletal_image(event, compound)
            if skeletal_path is None:
                image_path = self._condensed_image(event, compound)

            if image_path is not None:
                chain.append(Image.fromFileSystem(str(image_path)))
            elif skeletal_path is not None:
                if prefer_skeletal:
                    chain.append(Plain("该物质是环状结构，下面是键线式："))
                else:
                    chain.append(Plain("该物质没有连写式简式，下面是键线式："))
                chain.append(Image.fromFileSystem(str(skeletal_path)))
            elif compound.condensed_nodes:
                chain.append(
                    Plain(
                        "（图片渲染不可用，以下是文本形式）\n"
                        f"{compound.condensed_unicode}",
                    ),
                )
            elif skeletal_reason:
                chain.append(Plain(f"（无法出图：{skeletal_reason}）"))

            if not chain:
                # 文本与图片都被关掉或都不可用时，至少给点反馈
                chain.append(Plain(self._format_compound(compound)))
            yield event.chain_result(chain)

    @chem_group.command("简式", alias={"结构简式"})
    async def cmd_condensed(
        self, event: AstrMessageEvent, query: GreedyStr = GreedyStr,
    ):
        """只输出结构简式文本。参数: 物质名称/SMILES/结构简式/分子式"""
        event.stop_event()
        text = self._query_of(query)
        if not text:
            yield event.plain_result("用法：/化学 简式 乙醇")
            return
        async with self._lock:
            try:
                compound = await self._resolve(event, text)
            except ResolveError as error:
                yield event.plain_result(self._format_error(error))
                return

            if not compound.condensed_nodes:
                reason = (
                    compound.notes[0] if compound.notes
                    else "该物质无法用连写式简式表达"
                )
                yield event.plain_result(
                    f"{compound.display_name} 没有连写式结构简式。\n{reason}",
                )
                return

            lines = [f"{compound.display_name} 的结构简式："]
            lines.append(compound.condensed_unicode)
            lines.append(f"（纯文本形式：{compound.condensed_text}）")
            if compound.formula:
                lines.append(f"分子式：{decorate_formula(compound.formula)}")
            for note in compound.notes[: self.settings.output.max_notes]:
                lines.append(f"提示：{note}")
            yield event.plain_result("\n".join(lines))

    @chem_group.command("分子式", alias={"式量", "相对分子质量"})
    async def cmd_formula(
        self, event: AstrMessageEvent, query: GreedyStr = GreedyStr,
    ):
        """计算分子式、相对分子质量与元素质量分数。参数: 物质名称/SMILES/简式/分子式"""
        event.stop_event()
        text = self._query_of(query)
        if not text:
            yield event.plain_result("用法：/化学 分子式 C2H6O")
            return
        async with self._lock:
            try:
                compound = await self._resolve(event, text)
            except ResolveError as error:
                yield event.plain_result(self._format_error(error))
                return

            if compound.indeterminate or not compound.counts:
                lines = [f"{compound.display_name} 的组成不能唯一确定。"]
                if compound.formula:
                    lines.append(f"式子：{decorate_formula(compound.formula)}")
                lines.extend(f"提示：{n}" for n in compound.notes[:3])
                yield event.plain_result("\n".join(lines))
                return

            lines = [f"{compound.display_name}"]
            lines.append(f"分子式：{decorate_formula(compound.formula)}")
            mass = compound.molar_mass
            if mass is not None:
                lines.append(f"相对分子质量：{mass:.2f}")
            if compound.breakdown:
                lines.append("元素质量分数：")
                for symbol, count, fraction in compound.breakdown:
                    lines.append(
                        f"· {symbol}　{count} 个　{fraction * 100:.2f}%",
                    )
            if compound.candidates:
                lines.append("同分异构体：" + "、".join(compound.candidates))
            yield event.plain_result("\n".join(lines))

    @chem_group.command("结构式", alias={"骨架式", "骨骼式"})
    async def cmd_structure_drawing(
        self, event: AstrMessageEvent, query: GreedyStr = GreedyStr,
    ):
        """绘制结构式（优先 RDKit，未安装时用内置键线式引擎）。参数: 物质名称/SMILES"""
        event.stop_event()
        text = self._query_of(query)
        if not text:
            yield event.plain_result("用法：/化学 结构式 环己烷")
            return
        async with self._lock:
            try:
                compound = await self._resolve(event, text)
            except ResolveError as error:
                yield event.plain_result(self._format_error(error))
                return

            path, reason, engine = self._structure_image(event, compound)
            if path is None:
                lines = [f"画不出 {compound.display_name} 的结构式。", reason]
                if compound.condensed_nodes:
                    lines.append("结构简式：" + compound.condensed_unicode)
                yield event.plain_result("\n".join(lines))
                return

            lines = [f"{compound.display_name} 的结构式（{engine}）"]
            if compound.formula:
                lines.append(f"分子式：{decorate_formula(compound.formula)}")
            if compound.smiles:
                lines.append(f"SMILES：{compound.smiles}")
            yield event.chain_result(
                [Plain("\n".join(lines)), Image.fromFileSystem(str(path))],
            )

    @chem_group.command("键线式")
    async def cmd_skeletal(
        self, event: AstrMessageEvent, query: GreedyStr = GreedyStr,
    ):
        """用内置引擎绘制键线式（不依赖 RDKit）。参数: 物质名称/SMILES"""
        event.stop_event()
        text = self._query_of(query)
        if not text:
            yield event.plain_result("用法：/化学 键线式 环己烷")
            return
        async with self._lock:
            try:
                compound = await self._resolve(event, text)
            except ResolveError as error:
                yield event.plain_result(self._format_error(error))
                return

            path, reason = self._skeletal_image(event, compound)
            if path is None:
                lines = [f"画不出 {compound.display_name} 的键线式。", reason]
                if compound.condensed_nodes:
                    lines.append("结构简式：" + compound.condensed_unicode)
                yield event.plain_result("\n".join(lines))
                return

            lines = [f"{compound.display_name} 的键线式"]
            if compound.formula:
                lines.append(f"分子式：{decorate_formula(compound.formula)}")
            if compound.smiles:
                lines.append(f"SMILES：{compound.smiles}")
            yield event.chain_result(
                [Plain("\n".join(lines)), Image.fromFileSystem(str(path))],
            )

    @chem_group.command("列表", alias={"图鉴", "词表"})
    async def cmd_list(
        self, event: AstrMessageEvent, keyword: GreedyStr = GreedyStr,
    ):
        """浏览内置物质词表。参数: [类别或关键词]"""
        event.stop_event()
        text = self._query_of(keyword)
        if not text:
            lines = [f"内置物质共 {entry_count()} 条，按类别："]
            for name, entries in categories().items():
                lines.append(f"· {name}　{len(entries)} 条")
            lines.append("查看某一类：/化学 列表 醇")
            lines.append("也可以直接搜名字：/化学 列表 苯")
            yield event.plain_result("\n".join(lines))
            return

        all_categories = categories()
        if text in all_categories:
            entries = all_categories[text]
            lines = [f"{text}（{len(entries)} 条）："]
            for entry in entries:
                lines.append(f"· {entry.name}")
            yield event.plain_result("\n".join(lines))
            return

        hits = search_names(text)
        if not hits:
            lines = [
                f"没有找到与「{text}」相关的物质。",
                "可用类别：" + "、".join(all_categories),
            ]
            yield event.plain_result("\n".join(lines))
            return

        lines = [f"与「{text}」相关的物质（{len(hits)} 条）："]
        for entry in hits:
            formula = ""
            try:
                match = resolve(entry.name)
                formula = decorate_formula(match.formula)
            except ResolveError:
                formula = ""
            lines.append(f"· {entry.name}　{formula}　{entry.category}")
        yield event.plain_result("\n".join(lines))

    @chem_group.command("状态", alias={"自检"})
    async def cmd_status(self, event: AstrMessageEvent):
        """查看插件运行状态与环境自检结果。"""
        event.stop_event()
        report = self.service.environment_report()
        cache = report["cache"]
        uptime = time.time() - self._started_at
        lines = [
            f"化学结构简式插件 v{PLUGIN_VERSION}",
            f"· 内置物质：{entry_count()} 条",
            f"· Pillow：{report['pillow']}",
            f"· 字体：{report['font']}",
        ]
        if report["rdkit_available"]:
            lines.append(f"· RDKit：可用（{report['rdkit_version']}），优先用 RDKit 画结构式")
        else:
            lines.append("· RDKit：不可用，结构式改由内置键线式引擎绘制")
            lines.append(f"　原因：{report['rdkit_reason'] or '未知'}")
        if report["skeletal_enabled"]:
            lines.append("· 内置键线式引擎：已启用（纯 Python，无需额外依赖）")
        else:
            lines.append("· 内置键线式引擎：已关闭（skeletal.enabled）")
        lines.append(
            f"· 图片缓存：{cache['files']} 个文件 / {cache['bytes'] / 1024:.0f} KiB",
        )
        lines.append(f"· 缓存目录：{cache['dir']}")
        lines.append(
            f"· 本次运行：简式图 {report['render_count']} 次（含缓存命中），"
            f"键线式 {report['skeletal_count']} 次，"
            f"RDKit 结构式 {report['structure_count']} 次，"
            f"失败 {report['error_count']} 次",
        )
        lines.append(f"· 已运行：{uptime / 60:.1f} 分钟")
        yield event.plain_result("\n".join(lines))

    @chem_group.command("帮助", alias={"用法"})
    async def cmd_help(self, event: AstrMessageEvent):
        """查看使用说明。"""
        event.stop_event()
        yield event.plain_result(HELP_TEXT)

    @filter.command("结构", alias={"画结构"})
    async def cmd_structure_shortcut(
        self, event: AstrMessageEvent, query: GreedyStr = GreedyStr,
    ):
        """绘制结构简式（/化学 结构的快捷方式）。参数: 物质名称/SMILES/简式/分子式"""
        async for result in self.cmd_structure(event, query):
            yield result

    # ------------------------------------------------------------ Web API

    def _register_web_apis(self) -> None:
        """注册 Pages 使用的后端接口（路由带插件名前缀）。"""
        routes = (
            ("info", self.page_info, ["GET"], "插件与环境信息"),
            ("catalog", self.page_catalog, ["GET"], "内置物质词表"),
            ("render", self.page_render, ["POST"], "解析并渲染物质"),
        )
        for suffix, handler, methods, description in routes:
            try:
                self.context.register_web_api(
                    f"/{PLUGIN_NAME}/{suffix}", handler, methods, description,
                )
            except Exception as error:  # pragma: no cover - 重复注册时降级
                logger.warning(
                    "%s 注册 Web API %s 失败：%s", LOG_PREFIX, suffix, error,
                )

    async def page_info(self):
        """返回插件与环境信息。"""
        report = self.service.environment_report()
        return json_response(
            {
                "status": "ok",
                "data": {
                    "plugin": PLUGIN_NAME,
                    "version": PLUGIN_VERSION,
                    "entries": entry_count(),
                    "categories": {
                        name: len(items) for name, items in categories().items()
                    },
                    "environment": report,
                },
            },
        )

    async def page_catalog(self):
        """返回内置物质词表。"""
        payload = {
            name: [
                {
                    "name": entry.name,
                    "aliases": list(entry.aliases),
                    "smiles": entry.smiles,
                    "condensed": entry.condensed,
                }
                for entry in items
            ]
            for name, items in categories().items()
        }
        return json_response(
            {"status": "ok", "data": {"catalog": payload}},
        )

    async def page_render(self):
        """解析输入并返回简式文本、图片（base64）与结构式状态。"""
        try:
            payload = await request.json(default={})
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            return error_response("请求体需要是 JSON 对象")

        raw = str(payload.get("input") or "").strip()
        if not raw:
            return error_response("请提供 input 字段，例如 {\"input\": \"乙醇\"}")
        if len(raw) > 200:
            return error_response("输入过长，请控制在 200 字符以内")

        with_image = bool(payload.get("image", True))
        with_structure = bool(payload.get("structure", False))
        with_skeletal = bool(payload.get("skeletal", False))

        settings = self.settings
        try:
            compound = resolve(
                raw,
                enable_abbrev=settings.input.enable_abbrev,
                aggressive_abbrev=settings.input.aggressive_abbrev,
            )
        except ResolveError as error:
            return error_response(
                error.reason,
                data={"suggestions": error.suggestions, "details": error.details},
            )

        import base64

        data: dict[str, object] = {
            "name": compound.display_name,
            "category": compound.category,
            "source": compound.source_label,
            "formula": compound.formula,
            "formula_html": decorate_formula(compound.formula),
            "molar_mass": compound.molar_mass,
            "condensed": compound.condensed_text,
            "condensed_html": compound.condensed_html,
            "smiles": compound.smiles,
            "notes": compound.notes,
            "candidates": compound.candidates,
            "text": self._format_compound(compound),
            "breakdown": [
                {"symbol": symbol, "count": count, "fraction": fraction}
                for symbol, count, fraction in compound.breakdown
            ],
        }

        if with_image and compound.condensed_nodes and settings.image.enabled:
            try:
                options: RenderOptions = self.service.condensed_options(
                    title=compound.display_name if compound.name else None,
                    caption=self._caption(compound),
                )
                png = self.service.render_condensed_png_bytes(compound, options)
                data["image"] = "data:image/png;base64," + base64.b64encode(
                    png,
                ).decode("ascii")
            except Exception as error:
                data["image_error"] = str(error)

        if with_skeletal:
            if compound.molecule is None:
                data["skeletal_error"] = (
                    "该输入只有元素组成（简式或分子式），没有键连信息，画不出键线式"
                )
            else:
                try:
                    skeletal_options = self.service.skeletal_options(
                        title=compound.display_name if compound.name else None,
                        caption=self._caption(compound),
                    )
                    png = self.service.render_skeletal_png_bytes(
                        compound, skeletal_options,
                    )
                    data["skeletal"] = (
                        "data:image/png;base64,"
                        + base64.b64encode(png).decode("ascii")
                    )
                except Exception as error:
                    data["skeletal_error"] = str(error)

        if with_structure:
            if compound.molecule is None:
                data["structure_error"] = "该输入没有键连信息，画不出结构式"
            else:
                path, reason = self.service.render_structure(compound.smiles or "")
                engine = "RDKit"
                if path is None and self.settings.skeletal.enabled:
                    path, reason = self.service.render_skeletal(compound)
                    engine = "内置键线式引擎"
                if path is None:
                    data["structure_error"] = reason
                else:
                    try:
                        data["structure"] = (
                            "data:image/png;base64,"
                            + base64.b64encode(path.read_bytes()).decode("ascii")
                        )
                        data["structure_engine"] = engine
                    except OSError as error:
                        data["structure_error"] = f"读取图片失败：{error}"

        return json_response({"status": "ok", "data": data})


HELP_TEXT = """化学结构简式插件 使用说明

支持四类输入，会自动识别：
· 中文名/俗名：乙醇、酒精、甘油、草酸、TNT、胆矾
· SMILES：CCO、c1ccccc1、CC(=O)O
· 结构简式：CH3CH2OH、CH2=CH2、CH3(CH2)4CH3、CuSO4·5H2O
· 分子式：C2H6O、C6H6、NaCl（同分异构体会一并列出）

指令：
· /化学 结构 <物质>    画结构简式图片；若是环状物质则改画键线式
· /化学 简式 <物质>    只用文字给出结构简式
· /化学 分子式 <物质>  计算分子式、相对分子质量、元素质量分数
· /化学 键线式 <物质>  用内置引擎画键线式（不需要 RDKit）
· /化学 结构式 <物质>  画结构式：有 RDKit 就用 RDKit，否则用内置引擎
· /化学 列表 [类别]    浏览内置物质词表
· /化学 状态           查看字体、RDKit、图片缓存等自检信息
· /化学 帮助           显示本说明
· /结构 <物质>         等价于 /化学 结构 <物质>

关于两种画法：
· 结构简式（连写式）：CH3CH2OH 这种把原子连起来写的式子。链状有机物与
  苯系物都能自动生成；环烷烃、杂环、稠环没法用连写式无歧义表达。
· 键线式（骨架式）：碳在顶点上不写出来，杂原子标符号，双键画平行线的
  那种图。环己烷、苯、萘、葡萄糖环都能画，由插件自带的纯 Python 引擎
  生成，不需要安装任何额外依赖。

关于分子式：
· 分子式无法唯一确定结构（C2H6O 既可能是乙醇也可能是甲醚），
  这类输入会列出词表里的全部候选。
· 只给出简式或分子式时没有键连信息，画不出键线式；
  想要图请给中文名或 SMILES。"""
