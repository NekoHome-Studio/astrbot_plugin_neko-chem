"""分子图 → 结构简式。

中学教材对"结构简式"的书写约定，本模块按下列规则实现：

1. **省略键线**：C 与 C 之间的单键不写 ``-``；C 与杂原子之间的键也不写。
2. **只保留碳碳重键**：``C=C`` 写作 ``=``、``C≡C`` 写作 ``≡``；
   ``C=O``、``C≡N`` 这类碳与杂原子的重键直接连写（``CO``、``CN``），
   因为简式里它们本来就是连写的基团。
3. **主链 + 支链**：选定一条主链直写，支链用括号写在**分支点之后、
   主链延续之前**，例如 ``CH3CH(CH3)COOH``。相同支链合并下标。
4. **官能团位置**：羧酸/酯的酰基一侧作为书写起点，得到
   ``CH3COOH``、``CH3COOCH2CH3``、``HCOOCH3`` 这类教材写法。
5. **醛基的氢**：``C=O`` 上的氢紧跟在 C 之后写（``CH3CHO``）；
   若该碳是书写起点则提到最前（``HCHO``、``HCOOH``）。
6. **苯环**：以苯基形式展开，单取代为 ``C6H5-``、二取代为 ``C6H4-`` 等。

环状（非苯环）结构与稠环、杂环无法用简式无歧义表达，本模块会
返回 ``ok=False`` 并说明原因，由上层降级到文本 + 分子式或键线式。

本模块只依赖 :mod:`chem.smiles` 与 :mod:`chem.model`，不依赖 AstrBot。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .model import Atom, Bond, Group, Node, collect_counts, render_ascii, total_charge
from .smiles import AROMATIC_BOND_ORDER, Molecule

#: 视为"杂原子"的元素：与碳之间的重键在简式中不写符号。
HETERO_SYMBOLS = frozenset({"N", "O", "P", "S", "F", "Cl", "Br", "I", "Se", "Te"})

#: 卤素在简式中永远与碳连写并可用下标合并（CH2Cl2、CHCl3、CCl4）。
HALOGEN_SYMBOLS = frozenset({"F", "Cl", "Br", "I"})

#: 参与简式生成的原子数上限，超过则放弃，避免组合爆炸拖慢响应。
MAX_HEAVY_ATOMS = 60


@dataclass
class CondenseResult:
    """结构简式生成结果。

    Attributes:
        ok: 是否成功生成简式。
        nodes: 生成的 AST；失败时为空元组。
        text: 纯 ASCII 简式。
        reason: 失败原因（``ok`` 为假时有意义）。
        notes: 生成过程中的提示，例如苯环按苯基展开。
    """

    ok: bool
    nodes: tuple[Node, ...] = ()
    text: str = ""
    reason: str = ""
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 骨架模型


@dataclass
class Unit:
    """骨架节点：一个重原子，或一个整体展开的苯环。"""

    id: int
    symbol: str
    hydrogens: int = 0
    charge: int = 0
    ring_carbons: int = 0
    """大于 0 表示这是一个苯环整体（值为环上碳数 6）。"""
    source_atoms: tuple[int, ...] = ()

    @property
    def is_carbon(self) -> bool:
        """是否为碳（含苯环整体）。"""
        return self.symbol == "C"

    @property
    def is_ring(self) -> bool:
        """是否为苯环整体。"""
        return self.ring_carbons > 0


@dataclass
class Skeleton:
    """把苯环折叠成单个节点后的骨架图。"""

    units: list[Unit]
    adjacency: list[list[tuple[int, float]]]
    notes: list[str] = field(default_factory=list)

    def neighbors(self, unit_id: int) -> list[tuple[int, float]]:
        """返回 ``[(邻居 id, 键级)]``。"""
        return self.adjacency[unit_id]

    @property
    def carbon_total(self) -> int:
        """骨架中的碳原子总数（苯环按 6 计）。"""
        return sum(unit.ring_carbons or (1 if unit.is_carbon else 0)
                   for unit in self.units)


# ---------------------------------------------------------------- 苯环识别


def _benzene_rings(molecule: Molecule) -> list[list[int]]:
    """找出可视为"苯环"的六元芳香碳环。"""
    rings: list[list[int]] = []
    for cycle in molecule.find_cycles():
        if len(cycle) != 6:
            continue
        if not all(
            molecule.atoms[index].aromatic and molecule.atoms[index].symbol == "C"
            for index in cycle
        ):
            continue
        pairs_ok = all(
            (bond := molecule.bond_between(cycle[offset], cycle[(offset + 1) % 6]))
            is not None
            and bond.order == AROMATIC_BOND_ORDER
            for offset in range(6)
        )
        if pairs_ok:
            rings.append(cycle)
    return rings


def _build_skeleton(molecule: Molecule) -> tuple[Skeleton | None, str]:
    """把分子图转成骨架，成功返回 ``(骨架, "")``，失败返回 ``(None, 原因)``。"""
    notes: list[str] = []
    benzene = _benzene_rings(molecule)
    covered: set[int] = set()
    for ring in benzene:
        covered.update(ring)

    if molecule.has_ring:
        # 除已识别的苯环外，还有别的环（杂环、脂环、稠环）就无法用简式表达
        remaining_rings = [
            cycle for cycle in molecule.find_cycles()
            if not all(index in covered for index in cycle)
        ]
        if remaining_rings:
            sizes = sorted({len(cycle) for cycle in remaining_rings})
            size_text = "、".join(str(size) for size in sizes)
            return None, (
                f"结构中含 {size_text} 元环（非苯环），简式无法无歧义表达，"
                "请改用键线式/结构式"
            )
        stray_aromatic = [
            atom for atom in molecule.atoms
            if atom.aromatic and atom.index not in covered
        ]
        if stray_aromatic:
            return None, "结构中含苯环以外的芳香原子（杂芳环），简式无法无歧义表达"

    units: list[Unit] = []
    adjacency: list[list[tuple[int, float]]] = []
    atom_to_unit: dict[int, int] = {}

    def new_unit(unit: Unit) -> int:
        units.append(unit)
        adjacency.append([])
        return unit.id

    for ring in benzene:
        ring_set = set(ring)
        external = 0
        for index in ring:
            for neighbor, _ in molecule.neighbors(index):
                if neighbor not in ring_set:
                    external += 1
        unit_id = new_unit(
            Unit(
                id=len(units),
                symbol="C",
                ring_carbons=6,
                hydrogens=max(0, 6 - external),
                source_atoms=tuple(ring),
            ),
        )
        for index in ring:
            atom_to_unit[index] = unit_id

    for atom in molecule.atoms:
        if atom.index in atom_to_unit:
            continue
        atom_to_unit[atom.index] = new_unit(
            Unit(
                id=len(units),
                symbol=atom.symbol,
                hydrogens=atom.hydrogens,
                charge=atom.charge,
                source_atoms=(atom.index,),
            ),
        )

    for bond in molecule.bonds:
        left, right = atom_to_unit[bond.a], atom_to_unit[bond.b]
        if left == right:
            continue
        adjacency[left].append((right, bond.order))
        adjacency[right].append((left, bond.order))

    if benzene:
        notes.append(
            f"苯环按苯基形式展开（共 {len(benzene)} 个），例如 C6H5-、C6H4-",
        )

    return Skeleton(units=units, adjacency=adjacency, notes=notes), ""


# ---------------------------------------------------------------- 生成实现


def _is_carboxyl_carbon(skeleton: Skeleton, unit_id: int) -> bool:
    """判断是否为羧基/酯基碳（同时连有 ``=O`` 与 ``-O``）。"""
    unit = skeleton.units[unit_id]
    if unit.symbol != "C":
        return False
    has_double_o = False
    has_single_o = False
    for neighbor, order in skeleton.neighbors(unit_id):
        other = skeleton.units[neighbor]
        if other.symbol != "O":
            continue
        if order >= 2:
            has_double_o = True
        else:
            has_single_o = True
    return has_double_o and has_single_o


def _preferred_roots(skeleton: Skeleton) -> set[int]:
    """计算"应当作为书写起点"的节点集合。

    羧酸与酯按教材习惯从酰基一侧写起（``CH3COOH`` 而不是 ``HOCOCH3``），
    甲酸/甲酸酯这类羰基碳两侧都不是碳的，直接把羰基碳当作起点。
    """
    preferred: set[int] = set()
    found_carboxyl = False
    for unit in skeleton.units:
        if not _is_carboxyl_carbon(skeleton, unit.id):
            continue
        found_carboxyl = True
        carbon_neighbors = [
            neighbor for neighbor, _ in skeleton.neighbors(unit.id)
            if skeleton.units[neighbor].is_carbon
        ]
        if not carbon_neighbors:
            preferred.add(unit.id)
            continue
        # 取"不经过羰基碳"能到达的那一侧
        allowed: set[int] = set()
        stack = list(carbon_neighbors)
        while stack:
            current = stack.pop()
            if current in allowed or current == unit.id:
                continue
            allowed.add(current)
            stack.extend(
                neighbor for neighbor, _ in skeleton.neighbors(current)
                if neighbor != unit.id
            )
        preferred |= allowed
    return preferred if found_carboxyl else set()


def _carbon_reach(skeleton: Skeleton, unit_id: int, parent: int) -> int:
    """返回以 ``unit_id`` 为根、避开 ``parent`` 的子树中碳原子总数。"""
    total = skeleton.units[unit_id].ring_carbons or (
        1 if skeleton.units[unit_id].is_carbon else 0
    )
    for neighbor, _ in skeleton.neighbors(unit_id):
        if neighbor == parent:
            continue
        total += _carbon_reach(skeleton, neighbor, unit_id)
    return total


def _subtree_has_function(skeleton: Skeleton, unit_id: int, parent: int) -> bool:
    """子树中是否含"主要官能团"。

    用于在碳链长度相同的分支点处，把官能团所在的一侧选为主链延续方向，
    从而得到 ``CH3CH(CH3)CHO``、``CH3CH(CH3)COOH`` 这类教材写法。
    """
    unit = skeleton.units[unit_id]
    if unit.symbol == "C":
        for neighbor, order in skeleton.neighbors(unit_id):
            other = skeleton.units[neighbor]
            if other.symbol in {"O", "N", "S"} and (order >= 2 or other.hydrogens):
                return True
    elif unit.symbol in HETERO_SYMBOLS and unit.hydrogens:
        return True
    return any(
        _subtree_has_function(skeleton, neighbor, unit_id)
        for neighbor, _ in skeleton.neighbors(unit_id)
        if neighbor != parent
    )


def _element_nodes(symbol: str, hydrogens: int, charge: int = 0) -> list[Node]:
    """生成元素自身的节点，带电时把电荷写在氢之后（``NH3+``、``OH-``）。"""
    if hydrogens <= 0:
        return [Atom(symbol, 1, charge)]
    return [Atom(symbol), Atom("H", hydrogens, charge)]


def _self_text(
    unit: Unit,
    *,
    is_root: bool,
    is_carbonyl: bool,
    is_nitro: bool,
) -> list[Node]:
    """生成节点自身的文字（含自身氢原子）。"""
    if is_nitro:
        # 硝基整体写成 NO2，不再单独输出 N 上的电荷
        return [Atom("N"), Atom("O", 2)]

    if unit.is_ring:
        nodes: list[Node] = [Atom("C", unit.ring_carbons)]
        if unit.hydrogens:
            nodes.append(Atom("H", unit.hydrogens))
        return nodes

    hydrogens = unit.hydrogens
    if unit.symbol == "C" and is_carbonyl and hydrogens:
        # 羰基碳上的氢按教材写法当作独立的 H
        if is_root:
            self_nodes: list[Node] = [Atom("H"), Atom("C")]
            if hydrogens > 1:
                self_nodes.append(Atom("H", hydrogens - 1))
            return self_nodes
        return [Atom("C"), *([Atom("H")] * hydrogens)]

    if unit.symbol in {"O", "S", "Se", "Te", "N", "P"} and 1 <= hydrogens <= 2 \
            and is_root:
        # 作为书写起点的杂原子把氢写在前面：HO-、H2N-、HS-
        return [Atom("H", hydrogens), Atom(unit.symbol, charge=unit.charge)]

    return _element_nodes(unit.symbol, hydrogens, unit.charge)


def _merge_identical(renders: list[tuple[list[Node], str]]) -> list[Node]:
    """把相邻的、文字完全相同的片段合并成带下标的组（``Cl`` + ``Cl`` → ``Cl2``）。"""
    merged: list[Node] = []
    for _, chunk_iter in itertools.groupby(renders, key=lambda item: item[1]):
        chunk = list(chunk_iter)
        nodes = chunk[0][0]
        count = len(chunk)
        if count == 1:
            merged.extend(nodes)
            continue
        if len(nodes) == 1 and isinstance(nodes[0], Atom) and nodes[0].charge == 0:
            merged.append(Atom(nodes[0].symbol, nodes[0].count * count))
        else:
            merged.append(Group(tuple(nodes), count))
    return merged


def _merge_branches(renders: list[tuple[list[Node], str]]) -> list[Node]:
    """支链一律加括号；完全相同的相邻支链合并下标（``(OH)`` ×2 → ``(OH)2``）。"""
    merged: list[Node] = []
    for _, chunk_iter in itertools.groupby(renders, key=lambda item: item[1]):
        chunk = list(chunk_iter)
        merged.append(Group(tuple(chunk[0][0]), len(chunk)))
    return merged


@dataclass
class _Rendered:
    """一次渲染的中间结果。"""

    nodes: list[Node]
    spine_carbons: int


def _render(
    skeleton: Skeleton,
    unit_id: int,
    parent: int | None,
    is_root: bool,
) -> _Rendered:
    """递归渲染子树。"""
    unit = skeleton.units[unit_id]
    children = [
        (neighbor, order)
        for neighbor, order in skeleton.neighbors(unit_id)
        if neighbor != parent
    ]

    # 1) 硝基：N 上挂两个 O 时整体写作 NO2
    oxygen_children = [
        (neighbor, order) for neighbor, order in children
        if skeleton.units[neighbor].symbol == "O"
    ]
    is_nitro = (
        unit.symbol == "N"
        and not unit.is_ring
        and unit.hydrogens == 0
        and len(children) == 2
        and len(oxygen_children) == 2
        and any(order >= 2 for _, order in oxygen_children)
    )
    if is_nitro:
        children = []

    # 2) 吸收：卤素、以及连到杂原子上的重键，都直接跟在自身后面连写
    absorbed: list[tuple[list[Node], str]] = []
    remaining: list[tuple[int, float]] = []
    for neighbor, order in children:
        other = skeleton.units[neighbor]
        terminal = len(skeleton.neighbors(neighbor)) == 1
        is_halogen = other.symbol in HALOGEN_SYMBOLS and terminal
        is_multiple_hetero = (
            order > 1 and other.symbol in HETERO_SYMBOLS and terminal
        )
        if is_halogen or is_multiple_hetero:
            rendered = _render(skeleton, neighbor, unit_id, False)
            absorbed.append((rendered.nodes, render_ascii(rendered.nodes)))
        else:
            remaining.append((neighbor, order))

    is_carbonyl = any(
        order >= 2 and skeleton.units[neighbor].symbol == "O"
        for neighbor, order in skeleton.neighbors(unit_id)
    )

    # 3) 选择主链延续方向
    branch_candidates: list[tuple[int, float]] = []
    continuation: tuple[int, float] | None = None
    if remaining:
        carbon_children = [
            item for item in remaining
            if (_carbon_reach(skeleton, item[0], unit_id) > 0)
        ]
        if carbon_children:
            def key(item: tuple[int, float]) -> tuple:
                neighbor, order = item
                return (
                    _carbon_reach(skeleton, neighbor, unit_id),
                    int(_subtree_has_function(skeleton, neighbor, unit_id)),
                    order,
                    -neighbor,
                )

            continuation = max(carbon_children, key=key)
            branch_candidates = [
                item for item in remaining if item != continuation
            ]
        elif len(remaining) == 1:
            # 只有一个非碳取代基时直写在末尾（CH3CH2OH 而不是 CH3CH2(OH)）
            continuation = remaining[0]
        else:
            # 多个非碳取代基则全部作为支链，便于合并下标（CH(OH)2）
            branch_candidates = list(remaining)

    # 4) 组装
    nodes: list[Node] = _self_text(
        unit, is_root=is_root, is_carbonyl=is_carbonyl, is_nitro=is_nitro,
    )
    nodes.extend(_merge_identical(absorbed))

    spine_carbons = unit.ring_carbons or (1 if unit.is_carbon else 0)

    branch_renders: list[tuple[list[Node], str]] = []
    for neighbor, order in branch_candidates:
        rendered = _render(skeleton, neighbor, unit_id, False)
        payload: list[Node] = []
        if order >= 2 and unit.is_carbon and skeleton.units[neighbor].is_carbon:
            payload.append(Bond("≡" if order >= 3 else "="))
        payload.extend(rendered.nodes)
        branch_renders.append((payload, render_ascii(payload)))
    nodes.extend(_merge_branches(branch_renders))

    if continuation is not None:
        neighbor, order = continuation
        if order >= 2 and unit.is_carbon and skeleton.units[neighbor].is_carbon:
            nodes.append(Bond("≡" if order >= 3 else "="))
        rendered = _render(skeleton, neighbor, unit_id, False)
        nodes.extend(rendered.nodes)
        spine_carbons += rendered.spine_carbons

    return _Rendered(nodes=nodes, spine_carbons=spine_carbons)


def _count_parentheses(nodes: list[Node] | tuple[Node, ...]) -> int:
    """统计 AST 中显式括号的个数。"""
    total = 0
    for node in nodes:
        if isinstance(node, Group):
            total += 1 + _count_parentheses(node.items)
    return total


#: 出现在括号里就"不像教材写法"的主要官能团标志。
_FUNCTIONAL_MARKERS = ("COOH", "COO", "CHO", "CN", "NO2", "SO3H")


def _functional_branch_penalty(nodes: list[Node] | tuple[Node, ...]) -> int:
    """统计"被塞进括号的主要官能团"数量。

    羧基、醛基这类主要官能团应当位于主链末端直写（``CH3CH2CH(CH3)COOH``），
    而不是被括号包成支链（``CH3CH(COOH)CH2CH3``）。这个惩罚项用于在
    其他条件打平时挑出教材写法。
    """
    penalty = 0
    for node in nodes:
        if not isinstance(node, Group):
            continue
        text = render_ascii(node.items)
        if any(marker in text for marker in _FUNCTIONAL_MARKERS):
            penalty += 1
        penalty += _functional_branch_penalty(node.items)
    return penalty


def molecule_to_condensed(molecule: Molecule) -> CondenseResult:
    """把分子图转成结构简式。

    Args:
        molecule: :func:`chem.smiles.parse_smiles` 得到的分子图。

    Returns:
        生成结果；``ok`` 为假时 ``reason`` 说明无法生成的原因。
    """
    if molecule.component_count > 1:
        return CondenseResult(
            ok=False,
            reason="该 SMILES 含多个互不相连的组分（如盐、混合物），简式无法整体表达",
        )
    if not molecule.atoms:
        return CondenseResult(ok=False, reason="分子为空")
    if len(molecule.atoms) > MAX_HEAVY_ATOMS:
        return CondenseResult(
            ok=False,
            reason=f"分子含有 {len(molecule.atoms)} 个重原子，超出简式自动生成上限",
        )

    skeleton, error = _build_skeleton(molecule)
    if skeleton is None:
        return CondenseResult(ok=False, reason=error)

    preferred_roots = _preferred_roots(skeleton)
    best: tuple[tuple, tuple[Node, ...], int] | None = None

    for unit in skeleton.units:
        rendered = _render(skeleton, unit.id, None, True)
        text = render_ascii(rendered.nodes)
        score = (
            _count_parentheses(rendered.nodes),
            0 if unit.id in preferred_roots else 1,
            0 if unit.is_carbon else 1,
            0 if len(skeleton.neighbors(unit.id)) <= 1 else 1,
            -rendered.spine_carbons,
            _functional_branch_penalty(rendered.nodes),
            len(text),
            text,
        )
        if best is None or score < best[0]:
            best = (score, tuple(rendered.nodes), unit.id)

    assert best is not None
    nodes, _ = best[1], best[2]

    # 校验：简式还原出的分子式必须与原分子一致，否则说明生成逻辑有漏
    generated = collect_counts(nodes)
    expected = molecule.formula_counts()
    if generated != expected or total_charge(nodes) != molecule.total_charge():
        return CondenseResult(
            ok=False,
            reason=(
                "自动生成的结构简式与原分子式不一致（"
                f"生成 {generated}，应为 {expected}），已放弃输出以免误导"
            ),
        )

    return CondenseResult(
        ok=True,
        nodes=nodes,
        text=render_ascii(nodes),
        notes=list(skeleton.notes),
    )
