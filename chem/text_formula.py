"""结构简式 / 结构式文本的解析器。

支持把下列写法统一解析成一棵 :mod:`chem.model` AST：

* 结构简式：``CH3CH2OH``、``CH3COOH``、``CH3(CH2)4CH3``、``CH2OH(CHOH)4CHO``
* 带键线的结构式描述：``CH3-CH2-OH``、``CH2=CH2``、``HC≡CH``
* 带真上下标的粘贴文本：``CH₃CH₂OH``、``NH₄⁺``
* 结晶水等系数前缀：``CuSO4·5H2O``
* 常见基团缩写：``PhCH3``、``CH3CH2Me``

解析器只做词法/语法层面的检查（元素符号、括号配对、下标位置），
不做价键合理性校验——简式本身不携带完整结构信息。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .elements import ELEMENT_SYMBOLS
from .model import (
    BOND_ALIASES,
    VARIABLE_SUBSCRIPT_LETTERS,
    Atom,
    Bond,
    Group,
    IndeterminateFormula,
    Node,
    collect_counts,
    hill_formula,
    normalize_unicode_formula,
    render_ascii,
    total_charge,
)


class FormulaSyntaxError(ValueError):
    """结构简式语法错误。

    Attributes:
        text: 原始输入。
        position: 出错位置（归一化后字符串的下标）。
        reason: 中文错误说明。
    """

    def __init__(self, text: str, position: int, reason: str) -> None:
        self.text = text
        self.position = position
        self.reason = reason
        pointer = " " * max(position, 0) + "^"
        super().__init__(f"{reason}\n{text}\n{pointer}")


#: 默认启用的基团缩写。这些缩写都不会与元素符号冲突。
SAFE_ABBREVIATIONS: dict[str, str] = {
    "Allyl": "CH2CH=CH2",
    "Me": "CH3",
    "Et": "CH2CH3",
    "nPr": "CH2CH2CH3",
    "iPr": "CH(CH3)2",
    "nBu": "CH2CH2CH2CH3",
    "iBu": "CH2CH(CH3)2",
    "sBu": "CH(CH3)CH2CH3",
    "tBu": "C(CH3)3",
    "Ph": "C6H5",
    "Bn": "CH2C6H5",
    "Vi": "CH=CH2",
}

#: 与元素符号冲突或含义有歧义的缩写，需显式开启 ``aggressive_abbrev``。
RISKY_ABBREVIATIONS: dict[str, str] = {
    "Ac": "CH3CO",  # Ac 同时是锕(Actinium)
    "Pr": "CH2CH2CH3",  # Pr 同时是镨(Praseodymium)
    "Bz": "C6H5CO",
    "Tos": "C7H7SO2",
    "Ms": "CH3SO2",
}

_ABBREV_CACHE: dict[tuple[str, bool], tuple[Node, ...]] = {}


@dataclass
class ParsedCondensed:
    """结构简式解析结果。

    Attributes:
        source: 归一化后的输入文本。
        raw: 用户原始输入。
        nodes: 解析出的 AST。
        counts: 元素原子个数统计。
        charge: 总电荷。
        is_bare_formula: 输入文本与自身 Hill 分子式完全一致，因此无法与
            分子式区分（如 ``C2H6O``）。
        indeterminate: 含聚合度这类变量下标，分子式不确定。
        abbreviations: 本次解析实际使用的缩写。
        warnings: 非致命提示。
    """

    source: str
    raw: str
    nodes: tuple[Node, ...]
    counts: dict[str, int]
    charge: int
    is_bare_formula: bool
    indeterminate: bool = False
    abbreviations: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)

    @property
    def ascii_text(self) -> str:
        """纯 ASCII 形式的简式。"""
        return render_ascii(self.nodes)

    @property
    def formula(self) -> str:
        """Hill 分子式；下标不确定时返回空串。"""
        if self.indeterminate:
            return ""
        return hill_formula(self.counts, self.charge)


def abbreviations_for(aggressive: bool) -> dict[str, str]:
    """返回启用的缩写表。"""
    if aggressive:
        merged = dict(SAFE_ABBREVIATIONS)
        merged.update(RISKY_ABBREVIATIONS)
        return merged
    return dict(SAFE_ABBREVIATIONS)


def _abbrev_ast(expansion: str) -> tuple[Node, ...]:
    """把缩写的展开式编译成 AST（内部再解析一次，禁用缩写避免递归）。"""
    cache_key = (expansion, False)
    cached = _ABBREV_CACHE.get(cache_key)
    if cached is not None:
        return cached
    parsed = parse_condensed(expansion, enable_abbrev=False)
    _ABBREV_CACHE[cache_key] = parsed.nodes
    return parsed.nodes


def _match_abbreviation(
    text: str,
    index: int,
    table: dict[str, str],
) -> tuple[str, str] | None:
    """在 ``index`` 处做最长缩写的匹配。

    缩写分为两类：``Me``/``Et``/``Ph`` 这类大写开头，以及 ``iPr``/``nBu``
    这类小写前缀开头。小写开头的缩写不会与元素符号冲突，可以直接匹配；
    大写开头的缩写则要求下一个字符不是小写字母，避免把 ``Mercury`` 之类
    的普通文本误吞成 ``Me``。
    """
    best: tuple[str, str] | None = None
    for abbrev, expansion in table.items():
        if not text.startswith(abbrev, index):
            continue
        end = index + len(abbrev)
        if abbrev[0].isupper() and end < len(text) and text[end].islower():
            continue
        if best is None or len(abbrev) > len(best[0]):
            best = (abbrev, expansion)
    return best


def _read_number(text: str, index: int) -> tuple[int | str | None, int]:
    """读取可选的下标，返回 ``(数值或聚合度变量, 新下标)``。

    下标可以是数字，也可以是 ``n``/``m``/``x`` 这类聚合度变量
    （如淀粉 ``(C6H10O5)n``）。字母下标只在后面不是另一个单元的
    开头时才成立，避免把 ``nPr`` 这类缩写误读成下标。
    """
    start = index
    while index < len(text) and text[index].isdigit():
        index += 1
    if index > start:
        return int(text[start:index]), index

    if index < len(text) and text[index] in VARIABLE_SUBSCRIPT_LETTERS:
        following = text[index + 1] if index + 1 < len(text) else ""
        # 后面还跟着单元开头（大写字母、左括号）就说明是缩写/新单元
        if following == "" or following in ")]" or following in BOND_ALIASES:
            return text[index], index + 1
    return None, start


def _apply_count(nodes: tuple[Node, ...], count: int | str | None) -> list[Node]:
    """给刚解析出的单元套上下标个数。"""
    if count is None or count == 1:
        return list(nodes)
    if len(nodes) == 1 and isinstance(nodes[0], Atom):
        atom = nodes[0]
        if isinstance(count, str) or isinstance(atom.count, str):
            return [Atom(atom.symbol, count, atom.charge)]
        return [Atom(atom.symbol, atom.count * count, atom.charge)]
    return [Group(nodes, count)]


def _apply_charge(nodes: list[Node], position: int, charge: int, raw: str) -> None:
    """把电荷挂到最近的一个原子节点上。"""
    for offset in range(len(nodes) - 1, -1, -1):
        node = nodes[offset]
        if isinstance(node, Atom):
            nodes[offset] = Atom(node.symbol, node.count, node.charge + charge)
            return
    raise FormulaSyntaxError(raw, position, "电荷符号前面缺少原子")


def parse_condensed(
    text: str,
    *,
    enable_abbrev: bool = True,
    aggressive_abbrev: bool = False,
) -> ParsedCondensed:
    """解析结构简式 / 结构式描述文本。

    Args:
        text: 用户输入。
        enable_abbrev: 是否识别 ``Me``、``Et``、``Ph`` 等基团缩写。
        aggressive_abbrev: 是否连 ``Ac``、``Pr`` 这类与元素符号冲突的
            缩写也一并识别。

    Returns:
        解析结果。

    Raises:
        FormulaSyntaxError: 文本不是合法的简式时抛出。
    """
    raw = text
    source = normalize_unicode_formula(text).strip()
    # 去掉空白与常见的分隔逗号，保留化学符号本身
    source = source.replace(" ", "").replace("\t", "").replace(",", "").replace("，", "")
    if not source:
        raise FormulaSyntaxError(raw, 0, "输入为空，请给出结构简式或物质名称")

    table = abbreviations_for(aggressive_abbrev) if enable_abbrev else {}
    used_abbrev: list[str] = []
    warnings: list[str] = []

    nodes, index = _parse_sequence(source, 0, raw, table, used_abbrev, top_level=True)
    if index != len(source):
        raise FormulaSyntaxError(raw, index, f"无法解析的字符 {source[index]!r}")

    counts: dict[str, int] = {}
    charge = 0
    indeterminate = False
    try:
        counts = collect_counts(nodes)
        charge = total_charge(nodes)
    except IndeterminateFormula:
        indeterminate = True
        warnings.append("式中含聚合度这类变量下标（如 n），分子式不唯一。")

    ascii_text = render_ascii(nodes)
    is_bare_formula = (
        not indeterminate
        and ascii_text == hill_formula(counts, charge)
    )

    if is_bare_formula and len(counts) > 1:
        warnings.append(
            "该写法与分子式完全相同，无法据此确定唯一的结构简式（例如 C2H6O "
            "既可能是乙醇也可能是甲醚）。",
        )

    return ParsedCondensed(
        source=source,
        raw=raw,
        nodes=tuple(nodes),
        counts=counts,
        charge=charge,
        is_bare_formula=is_bare_formula,
        indeterminate=indeterminate,
        abbreviations=tuple(used_abbrev),
        warnings=warnings,
    )


def _parse_sequence(
    text: str,
    index: int,
    raw: str,
    table: dict[str, str],
    used_abbrev: list[str],
    *,
    top_level: bool,
) -> tuple[list[Node], int]:
    """解析一段同级序列，返回 ``(节点列表, 新下标)``。"""
    nodes: list[Node] = []
    length = len(text)

    while index < length:
        char = text[index]

        if char == ")":
            if top_level:
                raise FormulaSyntaxError(raw, index, "多余的右括号")
            return nodes, index

        if char == "(":
            inner, index = _parse_sequence(
                text, index + 1, raw, table, used_abbrev, top_level=False,
            )
            if index >= length or text[index] != ")":
                raise FormulaSyntaxError(raw, index, "左括号没有对应的右括号")
            index += 1  # 跳过 ')'
            if not inner:
                raise FormulaSyntaxError(raw, index - 1, "空括号")
            count, index = _read_number(text, index)
            # 用户显式写的括号一律保留，即便个数为 1，以便原样回显
            nodes.append(Group(tuple(inner), count if count is not None else 1))
            continue

        if char.isdigit():
            # 位于单元开头，说明是"系数前缀"写法，如结晶水 5H2O
            count, index = _read_number(text, index)
            assert count is not None
            rest, index = _parse_sequence(
                text, index, raw, table, used_abbrev, top_level=top_level,
            )
            if not rest:
                raise FormulaSyntaxError(raw, index, "数字后面缺少化学式")
            nodes.append(Group(tuple(rest), count, prefix=True))
            continue

        if char in "+-":
            if char == "+" or _is_charge_position(text, index, nodes):
                _apply_charge(nodes, index, 1 if char == "+" else -1, raw)
                index += 1
                continue
            # 作为键线符号处理
            nodes.append(Bond("-"))
            index += 1
            continue

        if char in BOND_ALIASES:
            nodes.append(Bond(BOND_ALIASES[char]))
            index += 1
            continue

        if char == "[" or char == "]":
            raise FormulaSyntaxError(
                raw, index, "暂不支持方括号写法，请改用圆括号或直接写出原子",
            )

        # 先尝试基团缩写，再尝试元素符号
        abbreviation = _match_abbreviation(text, index, table)
        if abbreviation is not None:
            abbrev, expansion = abbreviation
            used_abbrev.append(abbrev)
            index += len(abbrev)
            count, index = _read_number(text, index)
            nodes.extend(_apply_count(_abbrev_ast(expansion), count))
            continue

        symbol, index = _read_element(text, index, raw)
        count, index = _read_number(text, index)
        nodes.extend(_apply_count((Atom(symbol),), count))

    return nodes, index


def _is_charge_position(text: str, index: int, nodes: list[Node]) -> bool:
    """判断 ``-`` 是电荷还是键线符号。"""
    if not nodes:
        return True
    following = text[index + 1] if index + 1 < len(text) else ""
    if following == "":
        return True
    # 后面紧跟另一个电荷符号时，把当前当作电荷
    if following in "+-)":
        return True
    # 后面紧跟键线符号时，当前只能是电荷
    if following in BOND_ALIASES:
        return True
    # 后面是原子/括号/缩写开头，则视为键线
    if following == "(" or following.isupper():
        return False
    return True


def _read_element(text: str, index: int, raw: str) -> tuple[str, int]:
    """读取一个元素符号，返回 ``(符号, 新下标)``。"""
    char = text[index]
    if not char.isupper():
        raise FormulaSyntaxError(
            raw, index, f"无法识别的符号 {char!r}（元素符号需以大写字母开头）",
        )
    two = text[index:index + 2]
    if len(two) == 2 and two in ELEMENT_SYMBOLS:
        return two, index + 2
    if char in ELEMENT_SYMBOLS:
        return char, index + 1
    raise FormulaSyntaxError(raw, index, f"未知元素符号 {char!r}")
