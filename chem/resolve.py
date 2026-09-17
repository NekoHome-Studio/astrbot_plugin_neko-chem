"""统一输入入口：把用户输入解析成一份完整的物质信息。

支持四类输入，按下列优先级自动识别：

1. **中文名 / 俗名**（``乙醇``、``酒精``、``甘油``、``TNT``）——查内置名称表；
2. **SMILES**（``CCO``、``c1ccccc1``、``CC(=O)O``）——解析成分子图；
3. **结构简式/结构式**（``CH3CH2OH``、``CH2=CH2``、``CH3(CH2)4CH3``）
   ——解析成 AST 后规范化；
4. **分子式**（``C2H6O``、``C6H6``）——按元素组成反查名称表，命中多个
   同分异构体时全部列出。

识别顺序经过专门设计，避免歧义被误判：

* ``CCO`` 既像 SMILES 又像"只有重原子的式子"，由于它不含氢、不含括号、
  也不含键符号，会被判为 SMILES（乙醇）；
* ``C2H6O`` 的写法与其 Hill 分子式完全相同，会被判为分子式，并提示
  "既可能是乙醇也可能是甲醚"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .condense import CondenseResult, molecule_to_condensed
from .model import (
    Atom,
    Bond,
    Group,
    Node,
    element_breakdown,
    hill_formula,
    molar_mass,
    render_ascii,
    render_html,
    render_unicode,
)
from .names import NameEntry, lookup_by_name, search_names
from .smiles import Molecule, SmilesError, parse_smiles
from .text_formula import FormulaSyntaxError, ParsedCondensed, parse_condensed

#: 输入中含这些字符时一定是 SMILES（简式里不会出现）。
_SMILES_ONLY_CHARS = ("[", "]", "@", "%", "\\", "/")

_CJK_PATTERN = re.compile(r"[\u3400-\u9fff\u3040-\u30ff]")


class ResolveError(ValueError):
    """输入无法解析。

    Attributes:
        text: 原始输入。
        suggestions: 可能的名称提示。
        details: 各条解析路径的具体报错，便于排查。
    """

    def __init__(
        self,
        text: str,
        reason: str,
        *,
        suggestions: list[str] | None = None,
        details: list[str] | None = None,
    ) -> None:
        self.text = text
        self.reason = reason
        self.suggestions = suggestions or []
        self.details = details or []
        super().__init__(reason)


@dataclass
class Compound:
    """一份完整的物质信息，供渲染层与命令层使用。"""

    input_text: str
    source_kind: str
    """``name`` / ``smiles`` / ``condensed`` / ``formula`` 之一。"""
    source_label: str
    """输入类型的中文说明。"""
    name: str | None = None
    category: str = ""
    smiles: str | None = None
    molecule: Molecule | None = None
    condensed_nodes: tuple[Node, ...] | None = None
    condensed_source: str = ""
    """``name-db``（教材写法）/ ``generated``（自动推导）/ ``user``（用户输入）。"""
    formula: str = ""
    counts: dict[str, int] = field(default_factory=dict)
    charge: int = 0
    indeterminate: bool = False
    """是否含聚合度这类无法确定分子式的下标。"""
    notes: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    """分子式反查出的其他同分异构体名称。"""

    # ------------------------------------------------------------ 派生属性

    @property
    def condensed_text(self) -> str:
        """纯 ASCII 简式。"""
        if self.condensed_nodes is None:
            return ""
        return render_ascii(self.condensed_nodes)

    @property
    def condensed_unicode(self) -> str:
        """带真下标的简式。"""
        if self.condensed_nodes is None:
            return ""
        return render_unicode(self.condensed_nodes)

    @property
    def condensed_html(self) -> str:
        """HTML 片段形式的简式。"""
        if self.condensed_nodes is None:
            return ""
        return render_html(self.condensed_nodes)

    @property
    def molar_mass(self) -> float | None:
        """相对分子质量；下标不确定时为 ``None``。"""
        if self.indeterminate or not self.counts:
            return None
        return molar_mass(self.counts)

    @property
    def breakdown(self) -> list[tuple[str, int, float]]:
        """元素质量分数明细。"""
        if self.indeterminate or not self.counts:
            return []
        return element_breakdown(self.counts)

    @property
    def display_name(self) -> str:
        """用于标题的显示名。

        优先用词表里的中文名；没有名字时回显用户输入的原文，
        而不是只给一个分子式，这样 ``CCO``、``CH3CH2OH`` 这类输入
        的输出标题与用户输入一致，更容易对照。
        """
        return self.name or self.input_text or self.formula

    @property
    def is_cyclic(self) -> bool:
        """是否含环。"""
        return bool(self.molecule and self.molecule.has_ring)


# ---------------------------------------------------------------- 识别辅助


def _has_cjk(text: str) -> bool:
    """是否含中日文字符。"""
    return bool(_CJK_PATTERN.search(text))


def _is_flat_atom_sequence(nodes: tuple[Node, ...]) -> bool:
    """是否只由元素原子组成（没有括号、没有键符号）。"""
    return all(isinstance(node, Atom) for node in nodes)


def _contains_explicit_hydrogen(nodes: tuple[Node, ...]) -> bool:
    """输入里是否写出了氢原子（含括号内部）。

    这是区分"简式"与"骨架式/SMILES"的关键判据：简式一定会把氢写出来
    （``CH3CH2OH``、``(C6H10O5)n``），而 ``CCO``、``CC(=O)O`` 这类只写
    重原子的写法实际上是 SMILES。括号和键符号不能作为判据，因为两者
    在两种写法里都会出现。
    """
    for node in nodes:
        if isinstance(node, Atom) and node.symbol == "H":
            return True
        if isinstance(node, Group) and _contains_explicit_hydrogen(node.items):
            return True
    return False


def _contains_adduct_bond(nodes: tuple[Node, ...]) -> bool:
    """是否含 ``·``（结晶水/加合物分隔符，如 ``CuSO4·5H2O``）。

    带 ``·`` 的写法只表达元素组成，不携带键连信息，因此应当按分子式
    处理（可以反查到"胆矾"），而不是当成结构简式。
    """
    for node in nodes:
        if isinstance(node, Bond) and node.symbol == "·":
            return True
        if isinstance(node, Group) and _contains_adduct_bond(node.items):
            return True
    return False


def _formula_key(counts: dict[str, int]) -> str:
    """把元素组成压成可比较的规范键（与书写顺序无关）。"""
    return "".join(
        f"{symbol}{count}" for symbol, count in sorted(counts.items())
    )


def _lookup_by_counts(counts: dict[str, int]) -> list[NameEntry]:
    """按元素组成反查名称表（与元素书写顺序无关）。"""
    from .names import NAME_ENTRIES

    key = _formula_key(counts)
    hits: list[NameEntry] = []
    for entry in NAME_ENTRIES:
        entry_counts = _entry_counts(entry)
        if entry_counts and _formula_key(entry_counts) == key:
            hits.append(entry)
    hits.sort(key=lambda item: item.name)
    return hits


def _entry_counts(entry: NameEntry) -> dict[str, int]:
    """取条目的元素组成。"""
    if not entry.smiles:
        return {}
    try:
        return parse_smiles(entry.smiles).formula_counts()
    except SmilesError:
        return {}


# ---------------------------------------------------------------- 各路构造


def _from_entry(entry: NameEntry, input_text: str) -> Compound:
    """由名称表条目构造物质信息。"""
    compound = Compound(
        input_text=input_text,
        source_kind="name",
        source_label="中文名/俗名",
        name=entry.name,
        category=entry.category,
    )
    notes = compound.notes

    if entry.smiles:
        compound.smiles = entry.smiles
        try:
            molecule = parse_smiles(entry.smiles)
            compound.molecule = molecule
            compound.counts = molecule.formula_counts()
            compound.charge = molecule.total_charge()
            compound.formula = hill_formula(compound.counts, compound.charge)
        except SmilesError as error:  # pragma: no cover - 词表已在测试中校验
            notes.append(f"内置 SMILES 解析失败：{error.reason}")

    # 优先使用教材约定写法
    if entry.condensed:
        try:
            parsed = parse_condensed(entry.condensed)
            compound.condensed_nodes = parsed.nodes
            compound.condensed_source = "name-db"
            if parsed.indeterminate:
                compound.indeterminate = True
                if not compound.formula:
                    compound.formula = parsed.ascii_text
                notes.append("该物质是高分子，聚合度 n 不定，只能给出最简式。")
        except FormulaSyntaxError as error:
            notes.append(f"内置简式解析失败：{error.reason}")

    # 没有约定写法就按分子图自动推导
    if compound.condensed_nodes is None and compound.molecule is not None:
        result = molecule_to_condensed(compound.molecule)
        if result.ok:
            compound.condensed_nodes = result.nodes
            compound.condensed_source = "generated"
            notes.extend(result.notes)
        else:
            notes.append(f"未给出连写式简式：{result.reason}")

    return compound


def _from_molecule(
    molecule: Molecule,
    input_text: str,
    *,
    source_kind: str,
    source_label: str,
    name: str | None = None,
    category: str = "",
) -> Compound:
    """由分子图构造物质信息。"""
    compound = Compound(
        input_text=input_text,
        source_kind=source_kind,
        source_label=source_label,
        name=name,
        category=category,
        smiles=input_text if source_kind == "smiles" else None,
        molecule=molecule,
        counts=molecule.formula_counts(),
        charge=molecule.total_charge(),
    )
    compound.formula = hill_formula(compound.counts, compound.charge)

    result: CondenseResult = molecule_to_condensed(molecule)
    if result.ok:
        compound.condensed_nodes = result.nodes
        compound.condensed_source = "generated"
        compound.notes.extend(result.notes)
    else:
        compound.notes.append(f"未给出连写式简式：{result.reason}")
        if molecule.has_ring:
            compound.notes.append("含环物质建议用键线式（多边形）表示，可尝试结构式绘图。")
    return compound


def _from_condensed(parsed: ParsedCondensed, input_text: str) -> Compound:
    """由用户输入的简式构造物质信息。"""
    compound = Compound(
        input_text=input_text,
        source_kind="condensed",
        source_label="结构简式",
        condensed_nodes=parsed.nodes,
        condensed_source="user",
        counts=parsed.counts,
        charge=parsed.charge,
        indeterminate=parsed.indeterminate,
    )
    if not parsed.indeterminate:
        compound.formula = parsed.formula
    else:
        compound.formula = parsed.ascii_text
    compound.notes.extend(parsed.warnings)
    if parsed.abbreviations:
        compound.notes.append(
            "已展开基团缩写：" + "、".join(dict.fromkeys(parsed.abbreviations)),
        )
    compound.notes.append(
        "输入的是简式，只有原子组成信息，无法反推键连方式，"
        "因此不能绘制结构式；如需结构式请给出来源名称或 SMILES。",
    )
    return compound


def _from_formula(
    counts: dict[str, int],
    input_text: str,
    hits: list[NameEntry],
) -> Compound:
    """由分子式构造物质信息。"""
    charge = 0
    formula = hill_formula(counts, charge)
    if hits:
        compound = _from_entry(hits[0], input_text)
        compound.source_kind = "formula"
        compound.source_label = "分子式"
        compound.candidates = [entry.name for entry in hits[1:]]
        if compound.candidates:
            compound.notes.append(
                "同一分子式对应多种结构，以下均符合："
                + "、".join([hits[0].name, *compound.candidates]),
            )
        return compound

    compound = Compound(
        input_text=input_text,
        source_kind="formula",
        source_label="分子式",
        formula=formula,
        counts=counts,
        charge=charge,
    )
    compound.notes.append(
        "仅凭分子式无法确定结构简式，内置词表中也没有该组成的常见物质。",
    )
    return compound


# ---------------------------------------------------------------- 主入口


def resolve(
    text: str,
    *,
    enable_abbrev: bool = True,
    aggressive_abbrev: bool = False,
) -> Compound:
    """把用户输入解析成 :class:`Compound`。

    Args:
        text: 用户输入，可以是中文名、SMILES、结构简式或分子式。
        enable_abbrev: 是否识别 ``Me``/``Et``/``Ph`` 等基团缩写。
        aggressive_abbrev: 是否连 ``Ac``/``Pr`` 这类与元素冲突的缩写也识别。

    Returns:
        解析结果。

    Raises:
        ResolveError: 四条解析路径都无法识别时抛出。
    """
    raw = (text or "").strip()
    if not raw:
        raise ResolveError(raw, "输入为空")

    details: list[str] = []

    # ---- 1) 中文名 / 俗名
    if _has_cjk(raw):
        entry = lookup_by_name(raw)
        if entry is not None:
            return _from_entry(entry, raw)
        suggestions = [item.name for item in search_names(raw)]
        raise ResolveError(
            raw,
            f"名称表里没有找到「{raw}」",
            suggestions=suggestions,
        )

    # 纯拉丁字母的别名（如 TNT、PHBV）也先试名称表
    entry = lookup_by_name(raw)
    if entry is not None:
        return _from_entry(entry, raw)

    # ---- 2) 含 SMILES 专有语法，直接按 SMILES 处理
    if any(char in raw for char in _SMILES_ONLY_CHARS):
        try:
            molecule = parse_smiles(raw)
        except SmilesError as error:
            raise ResolveError(
                raw,
                "看起来是 SMILES，但无法解析",
                details=[error.reason],
            ) from error
        return _from_molecule(
            molecule, raw, source_kind="smiles", source_label="SMILES",
        )

    # ---- 3) 尝试按结构简式解析
    parsed: ParsedCondensed | None = None
    try:
        parsed = parse_condensed(
            raw,
            enable_abbrev=enable_abbrev,
            aggressive_abbrev=aggressive_abbrev,
        )
    except FormulaSyntaxError as error:
        details.append(f"按简式解析失败：{error.reason}")

    if parsed is not None:
        if parsed.indeterminate:
            # 含聚合度变量（如淀粉 (C6H10O5)n），只能作为简式展示
            return _from_condensed(parsed, raw)

        if _contains_adduct_bond(parsed.nodes):
            # 结晶水/加合物只表达组成，按分子式反查（CuSO4·5H2O → 胆矾）
            return _from_formula(
                parsed.counts, raw, _lookup_by_counts(parsed.counts),
            )

        if parsed.is_bare_formula and not parsed.indeterminate:
            hits = _lookup_by_counts(parsed.counts)
            return _from_formula(parsed.counts, raw, hits)

        if _contains_explicit_hydrogen(parsed.nodes):
            return _from_condensed(parsed, raw)

        # 纯重原子序列：可能是分子式（NaCl、CCl4）也可能其实是 SMILES
        hits = _lookup_by_counts(parsed.counts)
        if hits:
            return _from_formula(parsed.counts, raw, hits)
        try:
            molecule = parse_smiles(raw)
        except SmilesError as error:
            details.append(f"按 SMILES 解析失败：{error.reason}")
        else:
            return _from_molecule(
                molecule, raw, source_kind="smiles", source_label="SMILES",
            )
        if _is_flat_atom_sequence(parsed.nodes):
            # 既不是已知物质、也不是 SMILES，但元素符号全合法，按分子式处理
            return _from_formula(parsed.counts, raw, [])

    # ---- 4) 兜底：再试一次 SMILES
    if parsed is None:
        try:
            molecule = parse_smiles(raw)
        except SmilesError as error:
            details.append(f"按 SMILES 解析失败：{error.reason}")
        else:
            return _from_molecule(
                molecule, raw, source_kind="smiles", source_label="SMILES",
            )

    raise ResolveError(
        raw,
        f"无法识别「{raw}」。可以输入中文名（乙醇）、SMILES（CCO）、"
        "结构简式（CH3CH2OH）或分子式（C2H6O）。",
        details=details,
    )


def suggest_examples() -> list[str]:
    """给出一组示例输入，用于帮助信息。"""
    return ["乙醇", "葡萄糖", "CCO", "c1ccccc1", "CH3CH(CH3)COOH", "C2H6O", "CuSO4·5H2O"]
