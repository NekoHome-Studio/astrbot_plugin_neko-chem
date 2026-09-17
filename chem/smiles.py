"""SMILES 解析器（有机化学常用子集）。

实现的是教科书与常见数据库里真正会用到的那部分 SMILES：

* 原子：有机子集 ``B C N O P S F Cl Br I`` 与芳香小写 ``b c n o p s``；
  其他元素用方括号书写，如 ``[Si]``、``[Fe+3]``、``[nH]``；
* 键：``- = # : / \\``，其中 ``:`` 表示芳香键；
* 支链：``( )``；
* 成环：``1``…``9`` 与 ``%10``…``%99``；
* 断开：``.``（多组分，如盐 ``[Na+].[Cl-]``）；
* 方括号内的同位素、手性 ``@``/``@@``、氢数 ``H``/``H2``、电荷 ``+``/``-``。

不做的事情：立体化学的三维含义、原子映射编号的保留（会被忽略）、
以及任何价键合法性以外的化学合理性判断。隐式氢按标准价态推断。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .elements import (
    AROMATIC_SUBSET,
    ATOMIC_MASS,
    DEFAULT_VALENCE,
    ELEMENT_SYMBOLS,
    ORGANIC_SUBSET,
)

#: 芳香键在键级求和时的权重。用 1.5 可以一次性算对苯环、吡啶、
#: 萘的稠合碳等常见情况的隐式氢数。
AROMATIC_BOND_ORDER = 1.5


class SmilesError(ValueError):
    """SMILES 语法错误。

    Attributes:
        smiles: 原始输入。
        position: 出错位置。
        reason: 中文错误说明。
    """

    def __init__(self, smiles: str, position: int, reason: str) -> None:
        self.smiles = smiles
        self.position = position
        self.reason = reason
        pointer = " " * max(position, 0) + "^"
        super().__init__(f"{reason}\n{smiles}\n{pointer}")


@dataclass
class SmilesAtom:
    """SMILES 中的一个原子。"""

    index: int
    symbol: str
    aromatic: bool = False
    charge: int = 0
    isotope: int | None = None
    chirality: str | None = None
    explicit_h: int | None = None
    """方括号中显式写出的氢数；``None`` 表示按价态推断。"""
    hydrogens: int = 0
    """最终氢原子个数（显式或推断）。"""
    degree: int = 0
    """重原子邻居个数，由 :meth:`Molecule.build_adjacency` 回填。"""

    @property
    def is_carbon(self) -> bool:
        """是否为非芳香碳原子。"""
        return self.symbol == "C" and not self.aromatic


@dataclass(frozen=True)
class SmilesBond:
    """SMILES 中的一根键。"""

    a: int
    b: int
    order: float = 1.0

    @property
    def is_aromatic(self) -> bool:
        return self.order == AROMATIC_BOND_ORDER

    def other(self, index: int) -> int:
        """返回键另一端的原子下标。"""
        return self.b if index == self.a else self.a


@dataclass
class Molecule:
    """SMILES 解析得到的分子图。"""

    smiles: str
    atoms: list[SmilesAtom]
    bonds: list[SmilesBond]
    adjacency: list[list[tuple[int, float]]] = field(default_factory=list)

    def build_adjacency(self) -> None:
        """建立邻接表并回填每个原子的重原子度数。"""
        self.adjacency = [[] for _ in self.atoms]
        for bond in self.bonds:
            self.adjacency[bond.a].append((bond.b, bond.order))
            self.adjacency[bond.b].append((bond.a, bond.order))
        for atom in self.atoms:
            atom.degree = len(self.adjacency[atom.index])

    # ------------------------------------------------------------ 性质查询

    @property
    def ring_count(self) -> int:
        """独立环的个数（环烷数，即环的秩）。

        由图论公式 ``环数 = 键数 - 原子数 + 组分数`` 得到：苯为 1，
        萘为 2，环己烷为 1，链状分子为 0。
        """
        return len(self.bonds) - len(self.atoms) + self.component_count

    @property
    def has_ring(self) -> bool:
        """是否含环。"""
        return self.ring_count > 0

    @property
    def has_aromatic_ring(self) -> bool:
        """是否含芳香环。"""
        return any(bond.is_aromatic for bond in self.bonds)

    @property
    def component_count(self) -> int:
        """组分数（``.`` 断开的片段数）。"""
        seen: set[int] = set()
        components = 0
        for atom in self.atoms:
            if atom.index in seen:
                continue
            components += 1
            stack = [atom.index]
            seen.add(atom.index)
            while stack:
                current = stack.pop()
                for neighbor, _ in self.adjacency[current]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
        return components

    def formula_counts(self) -> dict[str, int]:
        """统计分子式（含隐式氢）。"""
        counts: dict[str, int] = {}
        for atom in self.atoms:
            counts[atom.symbol] = counts.get(atom.symbol, 0) + 1
            if atom.hydrogens:
                counts["H"] = counts.get("H", 0) + atom.hydrogens
        return counts

    def total_charge(self) -> int:
        """净电荷。"""
        return sum(atom.charge for atom in self.atoms)

    def molar_mass(self) -> float:
        """相对分子质量。"""
        return sum(
            ATOMIC_MASS[atom.symbol] + ATOMIC_MASS["H"] * atom.hydrogens
            for atom in self.atoms
        )

    def neighbors(self, index: int) -> list[tuple[int, float]]:
        """返回 ``[(邻居下标, 键级)]``。"""
        return self.adjacency[index]

    def bond_between(self, a: int, b: int) -> SmilesBond | None:
        """查找两原子之间的键。"""
        for bond in self.bonds:
            if {bond.a, bond.b} == {a, b}:
                return bond
        return None

    def find_cycles(self) -> list[list[int]]:
        """用 DFS 找出独立环，每个环返回一条原子路径（按环大小升序）。

        只用于判断"是否含环"、给出环大小与环上原子集合，不做最小环基计算。
        同一个环可能被多条回边发现，这里按原子集合去重。
        """
        cycles: list[list[int]] = []
        visited = [False] * len(self.atoms)
        parent = [-1] * len(self.atoms)
        depth = [0] * len(self.atoms)
        seen_edges: set[tuple[int, int]] = set()
        seen_sets: set[frozenset[int]] = set()

        for start in range(len(self.atoms)):
            if visited[start]:
                continue
            # 迭代式 DFS，避免长链结构触发递归深度上限
            stack: list[tuple[int, int]] = [(start, 0)]
            visited[start] = True
            while stack:
                node, edge_index = stack[-1]
                if edge_index < len(self.adjacency[node]):
                    stack[-1] = (node, edge_index + 1)
                    neighbor, _ = self.adjacency[node][edge_index]
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        parent[neighbor] = node
                        depth[neighbor] = depth[node] + 1
                        stack.append((neighbor, 0))
                    elif neighbor != parent[node]:
                        key = (min(node, neighbor), max(node, neighbor))
                        if key in seen_edges:
                            continue
                        seen_edges.add(key)
                        cycle = _trace_cycle(parent, depth, node, neighbor)
                        fingerprint = frozenset(cycle)
                        if fingerprint in seen_sets:
                            continue
                        seen_sets.add(fingerprint)
                        cycles.append(cycle)
                else:
                    stack.pop()
        cycles.sort(key=len)
        return cycles


def _trace_cycle(parent: list[int], depth: list[int], a: int, b: int) -> list[int]:
    """沿父指针回溯出 ``a`` 与 ``b`` 之间的环路径。"""
    left: list[int] = []
    right: list[int] = []
    x, y = a, b
    while depth[x] > depth[y]:
        left.append(x)
        x = parent[x]
    while depth[y] > depth[x]:
        right.append(y)
        y = parent[y]
    while x != y:
        left.append(x)
        right.append(y)
        x = parent[x]
        y = parent[y]
    left.append(x)
    return left + right[::-1]


# ---------------------------------------------------------------- 解析实现


def parse_smiles(smiles: str) -> Molecule:
    """解析 SMILES 字符串。

    Args:
        smiles: SMILES 表达式，如 ``CCO``、``c1ccccc1``、``CC(=O)O``。

    Returns:
        解析好的 :class:`Molecule`。

    Raises:
        SmilesError: 表达式不合法时抛出。
    """
    text = smiles.strip()
    if not text:
        raise SmilesError(smiles, 0, "SMILES 为空")

    atoms: list[SmilesAtom] = []
    bonds: list[SmilesBond] = []
    branch_stack: list[int] = []
    ring_map: dict[str, tuple[int, float | None]] = {}
    previous: int | None = None
    pending_order: float | None = None
    index = 0
    length = len(text)

    def add_atom(
        symbol: str,
        *,
        aromatic: bool = False,
        charge: int = 0,
        explicit_h: int | None = None,
        isotope: int | None = None,
        chirality: str | None = None,
    ) -> int:
        nonlocal previous, pending_order
        atom_index = len(atoms)
        atoms.append(
            SmilesAtom(
                index=atom_index,
                symbol=symbol,
                aromatic=aromatic,
                charge=charge,
                explicit_h=explicit_h,
                isotope=isotope,
                chirality=chirality,
            ),
        )
        if previous is not None:
            if pending_order is not None:
                order = pending_order
            elif atoms[previous].aromatic and aromatic:
                order = AROMATIC_BOND_ORDER
            else:
                order = 1.0
            bonds.append(SmilesBond(previous, atom_index, order))
        previous = atom_index
        pending_order = None
        return atom_index

    while index < length:
        char = text[index]

        if char == "(":
            if previous is None:
                raise SmilesError(smiles, index, "左括号前没有原子")
            branch_stack.append(previous)
            index += 1

        elif char == ")":
            if not branch_stack:
                raise SmilesError(smiles, index, "多余的右括号")
            previous = branch_stack.pop()
            pending_order = None
            index += 1

        elif char in "-=#:~/\\":
            order = {"-": 1.0, "=": 2.0, "#": 3.0, ":": AROMATIC_BOND_ORDER,
                     "/": 1.0, "\\": 1.0}[char]
            if previous is None:
                raise SmilesError(smiles, index, f"键符号 {char!r} 前没有原子")
            if pending_order is not None:
                raise SmilesError(smiles, index, "连续出现了两个键符号")
            pending_order = order
            index += 1

        elif char == ".":
            if previous is None and not atoms:
                raise SmilesError(smiles, index, "SMILES 不能以 '.' 开头")
            previous = None
            pending_order = None
            index += 1

        elif char == "%" or char.isdigit():
            index = _handle_ring_closure(
                text, index, smiles, ring_map, bonds, previous, pending_order, atoms,
            )
            pending_order = None

        elif char == "[":
            end = text.find("]", index)
            if end == -1:
                raise SmilesError(smiles, index, "方括号没有闭合")
            atom_data = _parse_bracket_atom(text[index + 1:end], smiles, index)
            add_atom(**atom_data)
            index = end + 1

        elif char.isalpha():
            symbol, aromatic, next_index = _parse_bare_atom(text, index, smiles)
            add_atom(symbol, aromatic=aromatic)
            index = next_index

        else:
            raise SmilesError(smiles, index, f"无法识别的字符 {char!r}")

    if branch_stack:
        raise SmilesError(smiles, length, "有未闭合的左括号")
    if ring_map:
        pending = "、".join(sorted(ring_map))
        raise SmilesError(smiles, length, f"环编号 {pending} 没有闭合")
    if pending_order is not None:
        raise SmilesError(smiles, length, "结尾的键符号后面缺少原子")
    if not atoms:
        raise SmilesError(smiles, 0, "SMILES 中没有原子")

    molecule = Molecule(smiles=smiles, atoms=atoms, bonds=bonds)
    molecule.build_adjacency()
    _assign_hydrogens(molecule)
    return molecule


def _handle_ring_closure(
    text: str,
    index: int,
    smiles: str,
    ring_map: dict[str, tuple[int, float | None]],
    bonds: list[SmilesBond],
    previous: int | None,
    pending_order: float | None,
    atoms: list[SmilesAtom],
) -> int:
    """处理环闭合编号，返回新的下标。"""
    if previous is None:
        raise SmilesError(smiles, index, "环编号前没有原子")

    if text[index] == "%":
        digits = text[index + 1:index + 3]
        if len(digits) < 2 or not digits.isdigit():
            raise SmilesError(smiles, index, "'%' 后面需要两位环编号")
        key = digits
        index += 3
    else:
        key = text[index]
        index += 1

    if key not in ring_map:
        ring_map[key] = (previous, pending_order)
        return index

    other, other_order = ring_map.pop(key)
    if other == previous:
        raise SmilesError(smiles, index, f"环编号 {key} 不能连在同一个原子上")
    if pending_order is not None:
        order = pending_order
    elif other_order is not None:
        order = other_order
    elif atoms[other].aromatic and atoms[previous].aromatic:
        order = AROMATIC_BOND_ORDER
    else:
        order = 1.0
    bonds.append(SmilesBond(other, previous, order))
    return index


def _parse_bare_atom(text: str, index: int, smiles: str) -> tuple[str, bool, int]:
    """解析不带方括号的原子，返回 ``(符号, 是否芳香, 新下标)``。"""
    two = text[index:index + 2]
    if len(two) == 2 and two in ORGANIC_SUBSET:
        return two, False, index + 2
    char = text[index]
    if char in ORGANIC_SUBSET:
        return char, False, index + 1
    if char in AROMATIC_SUBSET:
        return char.upper(), True, index + 1
    if two in ELEMENT_SYMBOLS or char in ELEMENT_SYMBOLS:
        raise SmilesError(
            smiles, index,
            f"{two if two in ELEMENT_SYMBOLS else char} 不在有机子集内，"
            f"请写成方括号形式，例如 [{two if two in ELEMENT_SYMBOLS else char}]",
        )
    raise SmilesError(smiles, index, f"无法识别的原子 {char!r}")


def _parse_bracket_atom(content: str, smiles: str, offset: int) -> dict:
    """解析方括号原子，返回 :meth:`add_atom` 的关键字参数。"""
    if not content:
        raise SmilesError(smiles, offset, "空的方括号")

    index = 0
    length = len(content)
    isotope: int | None = None

    start = index
    while index < length and content[index].isdigit():
        index += 1
    if index > start:
        isotope = int(content[start:index])

    if index >= length:
        raise SmilesError(smiles, offset, "方括号内缺少元素符号")

    symbol_char = content[index]
    if not symbol_char.isalpha():
        raise SmilesError(smiles, offset + index, f"方括号内无法识别的字符 {symbol_char!r}")

    aromatic = symbol_char.islower()
    if aromatic:
        if symbol_char not in AROMATIC_SUBSET:
            raise SmilesError(
                smiles, offset + index, f"方括号内不支持芳香小写 {symbol_char!r}",
            )
        symbol = symbol_char.upper()
        index += 1
    else:
        two = content[index:index + 2]
        if len(two) == 2 and two in ELEMENT_SYMBOLS:
            symbol = two
            index += 2
        elif symbol_char in ELEMENT_SYMBOLS:
            symbol = symbol_char
            index += 1
        else:
            raise SmilesError(
                smiles, offset + index, f"未知元素符号 {symbol_char!r}",
            )

    chirality: str | None = None
    if content.startswith("@@", index):
        chirality = "@@"
        index += 2
    elif content.startswith("@", index):
        chirality = "@"
        index += 1

    explicit_h: int | None = None
    if index < length and content[index] == "H":
        index += 1
        count_start = index
        while index < length and content[index].isdigit():
            index += 1
        explicit_h = int(content[count_start:index]) if index > count_start else 1

    charge = 0
    if index < length and content[index] in "+-":
        sign_char = content[index]
        index += 1
        magnitude_start = index
        while index < length and content[index].isdigit():
            index += 1
        if index > magnitude_start:
            charge = int(content[magnitude_start:index])
        else:
            charge = 1
            while index < length and content[index] == sign_char:
                charge += 1
                index += 1
        if sign_char == "-":
            charge = -charge

    # 原子映射编号（:1 之类）直接忽略
    if index < length and content[index] == ":":
        index += 1
        while index < length and content[index].isdigit():
            index += 1
    # 剩余的 @@TH 之类手性类别标记忽略
    if index < length and content[index] == "@":
        index += 1
        if index < length and content[index] == "@":
            index += 1

    if index != length:
        raise SmilesError(
            smiles, offset + index, f"方括号内无法解析的剩余内容 {content[index:]!r}",
        )

    if explicit_h is None and not aromatic:
        # 方括号原子未写 H 时，标准规定隐式氢为 0
        explicit_h = 0

    return {
        "symbol": symbol,
        "aromatic": aromatic,
        "charge": charge,
        "explicit_h": explicit_h,
        "isotope": isotope,
        "chirality": chirality,
    }


def _assign_hydrogens(molecule: Molecule) -> None:
    """按价态模型推断每个原子的隐式氢数。"""
    for atom in molecule.atoms:
        if atom.explicit_h is not None:
            atom.hydrogens = atom.explicit_h
            continue
        bond_sum = sum(order for _, order in molecule.adjacency[atom.index])
        valence = _effective_valence(atom)
        if valence is None:
            atom.hydrogens = 0
        else:
            atom.hydrogens = max(0, int(round(valence - bond_sum)))


def _effective_valence(atom: SmilesAtom) -> int | None:
    """返回考虑电荷后的目标价态，未知元素返回 ``None``。"""
    base = DEFAULT_VALENCE.get(atom.symbol)
    if base is None:
        return None
    if atom.symbol == "B":
        return base - atom.charge
    if atom.symbol == "C":
        return base - abs(atom.charge)
    if atom.symbol in {"N", "P"}:
        return base + atom.charge
    if atom.symbol in {"O", "S", "Se", "Te"}:
        return base + atom.charge
    return base
