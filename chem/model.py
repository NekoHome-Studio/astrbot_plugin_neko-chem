"""结构简式的抽象表示（AST）与通用渲染。

一个结构简式被表示成一串节点：

* :class:`Atom`  —— 元素原子，带下标个数与电荷，例如 ``H3``、``NH4+``；
* :class:`Group` —— 分组，带个数；``(CH2)4`` 是括号下标写法，
  ``5H2O`` 是系数前缀写法；
* :class:`Bond`  —— 键符号，例如 ``=``、``≡``、``-``。

这样既能表达 ``CH3CH2OH``，也能表达 ``CH2OH(CHOH)4CHO`` 这类带重复基团
的写法；分子式、摩尔质量、文本/图片/HTML 渲染全部由同一棵 AST 派生。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Union

from .elements import ATOMIC_MASS, element_zh

# ---------------------------------------------------------------- 节点定义


@dataclass(frozen=True)
class Atom:
    """元素原子节点。

    Attributes:
        symbol: 元素符号，如 ``"C"``。
        count: 下标原子个数，1 时不显示；也可以是 ``"n"`` 这类聚合度变量。
        charge: 电荷数，0 时不显示。
    """

    symbol: str
    count: Count = 1
    charge: int = 0


@dataclass(frozen=True)
class Group:
    """分组节点。

    Attributes:
        items: 组内节点。
        count: 分组个数，1 时不显示；也可以是 ``"n"`` 这类聚合度变量。
        prefix: 为真时使用"系数前缀"写法（结晶水 ``5H₂O``）；否则使用
            括号下标写法（``(CH₂)₄``）。
    """

    items: tuple["Node", ...]
    count: Count = 1
    prefix: bool = False


@dataclass(frozen=True)
class Bond:
    """键符号节点。``symbol`` 取 ``-``、``=``、``≡``、``·`` 之一。"""

    symbol: str = "-"


Node = Union[Atom, Group, Bond]

#: 下标既可以是具体数字，也可以是聚合度这类变量（淀粉 ``(C6H10O5)n``）。
Count = Union[int, str]

# ---------------------------------------------------------------- 字符映射

SUBSCRIPT_DIGITS = "₀₁₂₃₄₅₆₇₈₉"
SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"

#: 聚合度这类变量下标对应的 Unicode 下标字符。
VARIABLE_SUBSCRIPTS: dict[str, str] = {"n": "ₙ", "m": "ₘ", "x": "ₓ"}

#: 允许作为聚合度变量下标的字母。
#:
#: **只允许小写**。曾经把大写 N/M/X 也算进来，结果 ``CH3CN``（乙腈）里的氮
#: 被当成前一个碳的聚合度下标，解析成 ``CH₃Cₙ``；``CH2=CHCN``（丙烯腈）、
#: ``NN`` 同样受害。聚合度按惯例就是小写 ``n``（``(C6H10O5)n``），
#: 大写字母一律按元素符号处理。
VARIABLE_SUBSCRIPT_LETTERS = frozenset("nmx")

_PLAIN_TO_SUBSCRIPT = str.maketrans("0123456789", SUBSCRIPT_DIGITS)
_SUBSCRIPT_TO_PLAIN = str.maketrans(SUBSCRIPT_DIGITS, "0123456789")
_PLAIN_TO_SUPERSCRIPT = str.maketrans("0123456789", SUPERSCRIPT_DIGITS)
_SUPERSCRIPT_TO_PLAIN = str.maketrans(SUPERSCRIPT_DIGITS, "0123456789")

# 各类等价书写归一化：全角符号、Unicode 连字符、三键的多种写法等。
BOND_ALIASES: dict[str, str] = {
    "-": "-",
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "−": "-",
    "=": "=",
    "＝": "=",
    "≡": "≡",
    "#": "≡",
    "≣": "≡",
    "·": "·",
    "•": "·",
    ".": "·",
}

BOND_SYMBOLS = frozenset({"-", "=", "≡", "·"})

#: 渲染风格取值。
RenderStyle = str


def to_subscript(number: int | str) -> str:
    """把数字或聚合度变量转成 Unicode 下标字符串。"""
    if isinstance(number, int):
        return str(number).translate(_PLAIN_TO_SUBSCRIPT)
    return VARIABLE_SUBSCRIPTS.get(number, number)


def to_superscript(number: int) -> str:
    """把整数转成 Unicode 上标字符串。"""
    return str(number).translate(_PLAIN_TO_SUPERSCRIPT)


def has_variable_count(nodes: Iterable[Node]) -> bool:
    """AST 中是否存在聚合度这类不确定的下标。"""
    for node in nodes:
        if isinstance(node, Atom):
            if isinstance(node.count, str):
                return True
        elif isinstance(node, Group):
            if isinstance(node.count, str) or has_variable_count(node.items):
                return True
    return False


def normalize_unicode_formula(text: str) -> str:
    """把 Unicode 上下标还原成 ASCII 数字，并把全角字符转成半角。

    这样用户直接粘贴 ``CH₃CH₂OH`` 或带全角括号的式子也能被解析。
    """
    normalized = text.translate(_SUBSCRIPT_TO_PLAIN).translate(_SUPERSCRIPT_TO_PLAIN)
    result_chars: list[str] = []
    for char in normalized:
        code = ord(char)
        if code == 0x3000:
            result_chars.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            result_chars.append(chr(code - 0xFEE0))
        else:
            result_chars.append(char)
    return "".join(result_chars)


# ---------------------------------------------------------------- 统计与公式


class IndeterminateFormula(ValueError):
    """AST 中含聚合度这类变量下标，无法给出确定的分子式。"""


def collect_counts(nodes: Iterable[Node]) -> dict[str, int]:
    """递归统计 AST 中每种元素的原子个数（忽略电荷与键）。

    Raises:
        IndeterminateFormula: 含 ``n`` 这类变量下标时无法统计。
    """
    counts: dict[str, int] = {}

    def walk(items: Iterable[Node], multiplier: int) -> None:
        for node in items:
            if isinstance(node, Atom):
                if isinstance(node.count, str):
                    raise IndeterminateFormula(
                        f"下标 {node.count!r} 不是具体数字，无法确定分子式",
                    )
                counts[node.symbol] = counts.get(node.symbol, 0) + node.count * multiplier
            elif isinstance(node, Group):
                if isinstance(node.count, str):
                    raise IndeterminateFormula(
                        f"下标 {node.count!r} 不是具体数字，无法确定分子式",
                    )
                walk(node.items, multiplier * node.count)

    walk(nodes, 1)
    return counts


def total_charge(nodes: Iterable[Node]) -> int:
    """统计 AST 的总电荷。"""
    total = 0

    def walk(items: Iterable[Node], multiplier: int) -> None:
        nonlocal total
        for node in items:
            if isinstance(node, Atom):
                total += node.charge * multiplier
            elif isinstance(node, Group):
                if isinstance(node.count, str):
                    raise IndeterminateFormula(
                        f"下标 {node.count!r} 不是具体数字，无法确定电荷",
                    )
                walk(node.items, multiplier * node.count)

    walk(nodes, 1)
    return total


def hill_formula(counts: dict[str, int], charge: int = 0) -> str:
    """按 Hill 记法输出分子式。

    含碳时按 ``C、H、其余字母序`` 排列，不含碳时全部按字母序排列；
    带电时在末尾附上电荷。
    """
    if not counts:
        return ""

    def render_one(symbol: str, count: int) -> str:
        return symbol if count == 1 else f"{symbol}{count}"

    if "C" in counts:
        order: list[str] = ["C"]
        if "H" in counts:
            order.append("H")
        order.extend(sorted(s for s in counts if s not in {"C", "H"}))
    else:
        order = sorted(counts)

    text = "".join(render_one(symbol, counts[symbol]) for symbol in order)
    if charge:
        sign = "+" if charge > 0 else "-"
        magnitude = abs(charge)
        text += sign if magnitude == 1 else f"{magnitude}{sign}"
    return text


def molar_mass(counts: dict[str, int]) -> float:
    """按相对原子质量计算相对分子质量。"""
    return sum(ATOMIC_MASS[symbol] * count for symbol, count in counts.items())


def element_breakdown(counts: dict[str, int]) -> list[tuple[str, int, float]]:
    """返回 ``[(符号, 原子数, 质量分数)]``，按质量分数降序、符号升序。"""
    total = molar_mass(counts)
    if total <= 0:
        return []
    rows = [
        (symbol, count, ATOMIC_MASS[symbol] * count / total)
        for symbol, count in counts.items()
    ]
    rows.sort(key=lambda row: (-row[2], row[0]))
    return rows


# ---------------------------------------------------------------- 文本渲染


def _charge_text(charge: int, superscript: bool) -> str:
    """把电荷数渲染成 ``+``、``2+``、``-`` 等形式。

    ``superscript`` 为真时使用 Unicode 上标字符（``⁺``、``²⁺``），
    用于纯文本渲染；HTML 渲染另有 ``<sup>`` 包装。
    """
    if charge == 0:
        return ""
    sign = "+" if charge > 0 else "-"
    magnitude = abs(charge)
    if not superscript:
        return sign if magnitude == 1 else f"{magnitude}{sign}"
    sign_char = "⁺" if charge > 0 else "⁻"
    if magnitude == 1:
        return sign_char
    return str(magnitude).translate(_PLAIN_TO_SUPERSCRIPT) + sign_char


def _count_text(count: Count, style: RenderStyle) -> str:
    """渲染下标；``count`` 为 1 时返回空串。"""
    if count == 1:
        return ""
    if isinstance(count, int):
        text = str(count)
        if style == "unicode":
            text = text.translate(_PLAIN_TO_SUBSCRIPT)
    else:
        # 聚合度变量：unicode 用 ₙ，ascii/html 用普通字母
        text = VARIABLE_SUBSCRIPTS.get(count, count) if style == "unicode" else count
    return f"<sub>{text}</sub>" if style == "html" else text


def render(nodes: Iterable[Node], style: RenderStyle = "unicode") -> str:
    """把 AST 渲染成文本。

    Args:
        nodes: 节点序列。
        style: ``ascii``（``CH3CH2OH``）、``unicode``（``CH₃CH₂OH``）
            或 ``html``（``CH<sub>3</sub>``）。

    Returns:
        渲染结果字符串。

    Raises:
        ValueError: ``style`` 不是受支持的取值时抛出。
    """
    if style not in {"ascii", "unicode", "html"}:
        raise ValueError(f"未知渲染风格: {style!r}")
    escape = style == "html"
    superscript_charge = style == "unicode"

    parts: list[str] = []
    for node in nodes:
        if isinstance(node, Atom):
            parts.append(_escape_html(node.symbol) if escape else node.symbol)
            parts.append(_count_text(node.count, style))
            charge_text = _charge_text(node.charge, superscript_charge)
            if charge_text:
                parts.append(f"<sup>{charge_text}</sup>" if escape else charge_text)
        elif isinstance(node, Group):
            inner = render(node.items, style)
            if node.prefix and isinstance(node.count, int) and node.count > 1:
                # 系数前缀写法，例如结晶水 5H₂O；系数用正常字号，不下标
                parts.append(str(node.count) + inner)
                continue
            parts.append("(" + inner + ")")
            parts.append(_count_text(node.count, style))
        else:
            parts.append(_escape_html(node.symbol) if escape else node.symbol)
    return "".join(parts)


def render_ascii(nodes: Iterable[Node]) -> str:
    """渲染为纯 ASCII，例如 ``CH3CH2OH``，用于日志、缓存键与降级文本。

    已知歧义：电荷用 ``-``/``+`` 表示，而 ``-`` 同时是键符号。对于**电荷
    出现在分子中间**的写法（如硝酸根的 ``O⁻NO₂`` → ``O-NO2``），本函数的
    输出重新解析时会被当成键，从而丢失电荷。位于末尾的电荷没有这个问题
    （``CH3COO-``、``NH4+``），而教材写法里的离子电荷本来就在末尾。
    图片与 Unicode 形式使用上标字符，不受影响。
    """
    return render(nodes, "ascii")


def render_unicode(nodes: Iterable[Node]) -> str:
    """渲染为带真下标的文本，例如 ``CH₃CH₂OH``。"""
    return render(nodes, "unicode")


def render_html(nodes: Iterable[Node]) -> str:
    """渲染为 HTML 片段，使用 ``<sub>``/``<sup>`` 而非 Unicode 上下标。

    Web 端与 t2i 走这个函数，排版不依赖字体是否带上下标字形。
    """
    return render(nodes, "html")


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def decorate_formula(text: str) -> str:
    """把分子式里的数字下标改成真下标字符，如 ``C2H6O`` → ``C₂H₆O``。

    只转换连续数字，不会动元素符号或 ``(C6H10O5)n`` 里的 ``n``。
    """
    if not text:
        return ""
    parts: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char.isdigit():
            end = index
            while end < length and text[end].isdigit():
                end += 1
            parts.append(text[index:end].translate(_PLAIN_TO_SUBSCRIPT))
            index = end
        else:
            parts.append(char)
            index += 1
    return "".join(parts)


def describe_elements(counts: dict[str, int]) -> str:
    """把元素组成写成 ``碳 2 · 氢 6 · 氧 1`` 形式。"""
    if not counts:
        return ""
    if "C" in counts:
        order: list[str] = ["C"]
        if "H" in counts:
            order.append("H")
        order.extend(sorted(s for s in counts if s not in {"C", "H"}))
    else:
        order = sorted(counts)
    return " · ".join(f"{element_zh(s)} {counts[s]}" for s in order)
