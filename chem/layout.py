"""分子图的二维坐标生成（键线式排版）。

这是自研键线式绘图的"排版引擎"。输入是 :class:`chem.smiles.Molecule`，
输出是每个重原子的二维坐标（键长归一化为 1.0）。整个模块只用标准库。

算法分四步：

1. **找环（SSSR）**：对每根键，暂时断开它并求两端点间的最短路，得到一个
   包含该键的"最小环"；再用 GF(2) 上的线性无关性，从小到大贪心挑出
   ``环的秩 = 键数 − 原子数 + 组分数`` 个环。这样苯得到 1 个六元环、
   萘得到 2 个六元环（而不是一个十元环）。
2. **摆环**：第一个环摆成正多边形；其余环按三种连接方式依次拼上去 ——
   共享一条边（稠环）、共享一个原子（螺环）、以及环上某个原子已经
   与骨架成键（联苯、被链隔开的两个环）。三种情况都保持键长为 1。
   摆环与摆链交替进行，直到没有新进展。
3. **摆链与取代基**：从已定位的原子向外 BFS。环上原子朝环外伸展；
   链上原子相对上一根键交替转 ±60°，连续转下去就是教材里的锯齿形
   （相邻两根键夹角 60°，即键角 120°）。
4. **松弛**：对整体做若干轮"键长弹簧 + 非键斥力"的迭代，键长拉回 1.0、
   过近的非成键原子推开。环上原子视为刚性，不参与移动。

已知边界：这是"教学级"排版，不是 RDKit 那种通用坐标生成器。常见环
（3~7 元）、苯环、萘/蒽这类稠环、链与取代基都能排得规整；桥环与复杂
立体中心可能不够漂亮，代码会用 ``warnings`` 说明。
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field

from .smiles import AROMATIC_BOND_ORDER, Molecule

#: 归一化键长。所有坐标都以它为 1。
BOND_LENGTH = 1.0

_EPSILON = 1e-9

#: 链上相邻两根键的转角（度）。±60° 交替 ⇒ 键角 120°，即教材的锯齿形。
_CHAIN_TURN = 60.0


@dataclass
class Layout:
    """二维排版结果。

    Attributes:
        coords: 原子下标 → ``(x, y)``。
        rings: 识别出的最小环集合（每个环按成环顺序给出原子下标）。
        double_bonds: 需要画成双键的键集合（``frozenset({a, b})``）。
            芳香环的键由凯库勒化决定。
        warnings: 排版过程中的提示。
    """

    coords: dict[int, tuple[float, float]]
    rings: list[list[int]] = field(default_factory=list)
    double_bonds: set[frozenset[int]] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)
    quality: str = "ok"
    """``ok`` / ``approximate`` / ``poor``，见 :attr:`issues`。"""
    issues: list[str] = field(default_factory=list)
    """排版质量问题说明；``quality`` 为 ``ok`` 时为空。"""

    @property
    def drawable(self) -> bool:
        """排版质量是否足以出图。

        ``poor`` 表示环系在平面上无法用规整多边形表示（典型例子是桥环），
        这时候画出来的图会误导人，宁可不画。
        """
        return self.quality != "poor"

    def bounds(self) -> tuple[float, float, float, float]:
        """返回 ``(min_x, min_y, max_x, max_y)``。"""
        if not self.coords:
            return 0.0, 0.0, 0.0, 0.0
        xs = [point[0] for point in self.coords.values()]
        ys = [point[1] for point in self.coords.values()]
        return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------- 向量工具


def _sub(p: tuple[float, float], q: tuple[float, float]) -> tuple[float, float]:
    return p[0] - q[0], p[1] - q[1]


def _add(p: tuple[float, float], q: tuple[float, float]) -> tuple[float, float]:
    return p[0] + q[0], p[1] + q[1]


def _scale(p: tuple[float, float], k: float) -> tuple[float, float]:
    return p[0] * k, p[1] * k


def _norm(p: tuple[float, float]) -> float:
    return math.hypot(p[0], p[1])


def _unit(p: tuple[float, float]) -> tuple[float, float]:
    length = _norm(p)
    if length < _EPSILON:
        return 1.0, 0.0
    return p[0] / length, p[1] / length


def _rotate(p: tuple[float, float], degrees: float) -> tuple[float, float]:
    radians = math.radians(degrees)
    cos_t, sin_t = math.cos(radians), math.sin(radians)
    return p[0] * cos_t - p[1] * sin_t, p[0] * sin_t + p[1] * cos_t


def _centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def _bond_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


# ---------------------------------------------------------------- 环识别


def _shortest_path(
    molecule: Molecule,
    start: int,
    goal: int,
    banned: tuple[int, int],
) -> list[int] | None:
    """求 ``start`` 到 ``goal`` 的最短路，禁止经过 ``banned`` 这条边。"""
    previous: dict[int, int | None] = {start: None}
    queue: deque[int] = deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            path: list[int] = []
            node: int | None = goal
            while node is not None:
                path.append(node)
                node = previous[node]
            path.reverse()
            return path
        for neighbor, _ in molecule.neighbors(current):
            if _bond_key(current, neighbor) == banned:
                continue
            if neighbor in previous:
                continue
            previous[neighbor] = current
            queue.append(neighbor)
    return None


def find_sssr(molecule: Molecule) -> list[list[int]]:
    """求最小环集合（SSSR 的贪心近似）。

    返回的每个环都是按成环顺序排列的原子下标列表。
    """
    rank = len(molecule.bonds) - len(molecule.atoms) + molecule.component_count
    if rank <= 0:
        return []

    bond_index = {
        _bond_key(bond.a, bond.b): index
        for index, bond in enumerate(molecule.bonds)
    }

    # 候选：每根键所在的"最小环"
    candidates: list[tuple[int, list[int], int]] = []
    seen: set[frozenset[int]] = set()
    for bond in molecule.bonds:
        path = _shortest_path(
            molecule, bond.a, bond.b, _bond_key(bond.a, bond.b),
        )
        if not path or len(path) < 3:
            continue
        fingerprint = frozenset(path)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        mask = 0
        for index, atom in enumerate(path):
            key = _bond_key(atom, path[(index + 1) % len(path)])
            if key in bond_index:
                mask |= 1 << bond_index[key]
        if mask:
            candidates.append((len(path), path, mask))

    # 从小到大贪心，用 GF(2) 行化简保证线性无关
    candidates.sort(key=lambda item: item[0])
    pivots: dict[int, int] = {}
    selected: list[list[int]] = []
    for _, path, mask in candidates:
        if len(selected) >= rank:
            break
        reduced = mask
        while reduced:
            pivot = reduced.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = reduced
                selected.append(path)
                break
            reduced ^= pivots[pivot]
    return selected


# ---------------------------------------------------------------- 环摆放


def _regular_polygon(order: int) -> list[tuple[float, float]]:
    """生成边长为 1 的正 ``order`` 边形，``v0`` 在原点、``v1`` 在 ``(1, 0)``。

    顶点按逆时针方向排列，最后一条边正好回到 ``v0``。
    """
    turn = 2 * math.pi / order
    points = [(0.0, 0.0)]
    direction = 0.0
    current = (0.0, 0.0)
    for _ in range(order - 1):
        current = (
            current[0] + math.cos(direction),
            current[1] + math.sin(direction),
        )
        points.append(current)
        direction += turn
    return points


def _ring_starting_at(ring: list[int], atom: int) -> list[int]:
    """旋转环列表，使 ``atom`` 成为第一个元素。"""
    index = ring.index(atom)
    return ring[index:] + ring[:index]


def _ring_starting_at_edge(
    ring: list[int],
    first: int,
    second: int,
) -> list[int] | None:
    """旋转环列表使其以边 ``first → second`` 开头；两者不相邻时返回 ``None``。"""
    size = len(ring)
    for index in range(size):
        if ring[index] == first and ring[(index + 1) % size] == second:
            return ring[index:] + ring[:index]
    return None


def _place_first_ring(
    ring: list[int],
    coords: dict[int, tuple[float, float]],
    center: tuple[float, float],
    angle: float = 30.0,
) -> tuple[float, float]:
    """把环摆成正多边形（整体旋转 ``angle`` 度），返回环心。"""
    local = _regular_polygon(len(ring))
    local_center = _ring_center(local)
    offsets = [_rotate(_sub(point, local_center), angle) for point in local]
    for atom, offset in zip(ring, offsets):
        coords[atom] = _add(center, offset)
    return center


def _place_ring_anchored(
    ring: list[int],
    anchor: int,
    anchor_point: tuple[float, float],
    outward: tuple[float, float],
    coords: dict[int, tuple[float, float]],
) -> tuple[float, float]:
    """把环摆成：``anchor`` 落在 ``anchor_point``，环心沿 ``outward`` 方向。

    用于螺环、以及"环上某个原子已与骨架成键"的情形（联苯、被链隔开的环）。
    """
    ordered = _ring_starting_at(ring, anchor)
    local = _regular_polygon(len(ring))
    local_center = _ring_center(local)

    # local 中 v0 → 环心的方向，与目标 outward 对齐
    local_angle = math.degrees(
        math.atan2(
            local_center[1] - local[0][1],
            local_center[0] - local[0][0],
        ),
    )
    target_angle = math.degrees(math.atan2(outward[1], outward[0]))
    delta = target_angle - local_angle

    for atom, point in zip(ordered, local):
        coords[atom] = _add(anchor_point, _rotate(point, delta))

    radius = 1.0 / (2 * math.sin(math.pi / len(ring)))
    return _add(anchor_point, _scale(_unit(outward), radius))


def _place_ring_on_edge(
    ring: list[int],
    shared: tuple[int, int],
    coords: dict[int, tuple[float, float]],
    reference_center: tuple[float, float],
) -> tuple[float, float] | None:
    """沿一条已摆放的边向外拼接一个新环（稠环）。"""
    ordered = _ring_starting_at_edge(ring, shared[0], shared[1])
    if ordered is None:
        return None

    point_a, point_b = coords[shared[0]], coords[shared[1]]
    local = _regular_polygon(len(ring))
    local_center = _ring_center(local)

    best: tuple[float, list[tuple[float, float]]] | None = None
    for mirrored in (False, True):
        source = [
            (point[0], -point[1]) if mirrored else point for point in local
        ]
        transform = _similarity_transform(
            source[0], source[1], point_a, point_b,
        )
        placed = [transform(point) for point in source]
        center = _centroid(placed)
        distance = _norm(_sub(center, reference_center))
        if best is None or distance > best[0]:
            best = (distance, placed)

    assert best is not None
    for atom, point in zip(ordered, best[1]):
        # 与已有环共享的原子保持原位置：桥环里一个新环可能与已有环共享
        # 连续多条边，此时"两边都画成正多边形"在平面上无解，只能保留
        # 原骨架、把新环的其余原子摆上去，再靠松弛把键长拉匀。
        if atom in coords:
            continue
        coords[atom] = point
    return _centroid(best[1])


def _similarity_transform(
    source_a: tuple[float, float],
    source_b: tuple[float, float],
    target_a: tuple[float, float],
    target_b: tuple[float, float],
):
    """构造把 ``source_a → target_a``、``source_b → target_b`` 的相似变换。"""
    sx, sy = _sub(source_b, source_a)
    tx, ty = _sub(target_b, target_a)
    source_length = math.hypot(sx, sy)
    target_length = math.hypot(tx, ty)
    scale = target_length / source_length if source_length > _EPSILON else 1.0
    theta = math.atan2(ty, tx) - math.atan2(sy, sx)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def transform(point: tuple[float, float]) -> tuple[float, float]:
        x, y = _sub(point, source_a)
        return (
            target_a[0] + scale * (x * cos_t - y * sin_t),
            target_a[1] + scale * (x * sin_t + y * cos_t),
        )

    return transform


def _ring_center(points: list[tuple[float, float]]) -> tuple[float, float]:
    return _centroid(points)


def _try_attach_ring(
    molecule: Molecule,
    ring: list[int],
    coords: dict[int, tuple[float, float]],
    centers: list[tuple[float, float]],
) -> tuple[float, float] | None:
    """尝试把一个未摆放的环拼到已摆放的骨架上；成功返回环心。"""
    if not coords:
        return None

    placed_in_ring = [atom for atom in ring if atom in coords]

    # 情况一：共享一条边 → 稠环
    if len(placed_in_ring) >= 2:
        size = len(ring)
        for index in range(size):
            first, second = ring[index], ring[(index + 1) % size]
            if first in coords and second in coords:
                reference = _nearest_center(coords[first], centers)
                return _place_ring_on_edge(ring, (first, second), coords, reference)

    # 情况二：共享一个原子 → 螺环
    if len(placed_in_ring) == 1:
        anchor = placed_in_ring[0]
        anchor_point = coords[anchor]
        placed_neighbors = [
            n for n, _ in molecule.neighbors(anchor) if n in coords
        ]
        if placed_neighbors:
            reference = _centroid([coords[n] for n in placed_neighbors])
            outward = _unit(_sub(anchor_point, reference))
        else:
            outward = _unit(_sub(anchor_point, _nearest_center(anchor_point, centers)))
        if _norm(outward) < _EPSILON:
            outward = (0.0, 1.0)
        return _place_ring_anchored(ring, anchor, anchor_point, outward, coords)

    # 情况三：环上某个原子与骨架原子成键（联苯、被链隔开的两个环）
    for atom in ring:
        for neighbor, _ in molecule.neighbors(atom):
            if neighbor not in coords:
                continue
            anchor_point = coords[neighbor]
            placed_neighbors = [
                n for n, _ in molecule.neighbors(neighbor) if n in coords
            ]
            reference = _centroid([coords[n] for n in placed_neighbors])
            outward = _unit(_sub(anchor_point, reference))
            if _norm(outward) < _EPSILON:
                outward = (0.0, 1.0)
            landing = _add(anchor_point, _scale(_unit(outward), BOND_LENGTH))
            # 环心继续沿同一方向往外，保证新环不压在旧环上
            return _place_ring_anchored(ring, atom, landing, outward, coords)
    return None


def _nearest_center(
    point: tuple[float, float],
    centers: list[tuple[float, float]],
) -> tuple[float, float]:
    """找离 ``point`` 最近的已摆放环中心；没有环时回退到坐标原点。"""
    if not centers:
        return (0.0, 0.0)
    return min(centers, key=lambda center: _norm(_sub(point, center)))


# ---------------------------------------------------------------- 链与取代基


#: 取代基相对"主链延续方向"的排布角度。一个取代基时用 +120°
#: （四面体中心的典型画法），更多取代基继续取 -120°、180° 等。
_BRANCH_ANGLES: tuple[float, ...] = (120.0, -120.0, 180.0, 90.0, -90.0, 0.0)


def _subtree_size(
    molecule: Molecule,
    atom: int,
    parent: int,
    seen: set[int],
) -> int:
    """统计以 ``atom`` 为根、避开 ``parent`` 与已摆放原子的子树规模。"""
    total = 1
    for neighbor, _ in molecule.neighbors(atom):
        if neighbor == parent or neighbor in seen:
            continue
        seen.add(neighbor)
        total += _subtree_size(molecule, neighbor, atom, seen)
    return total


def _pick_branch_direction(
    coords: dict[int, tuple[float, float]],
    atom: int,
    base: tuple[float, float],
    angle_offset: float = 0.0,
) -> tuple[float, float]:
    """在主链方向的若干候选张角里，挑一个离已有原子最远的。

    单纯按固定角度挂取代基，偶尔会正好指回骨架里（例如苯甲酸的羰基氧
    会贴到邻位碳上，距离缩到 1.0）。这里对每个候选方向算一下"到最近的
    非母体原子的距离"，取最大的那个，代价很小但效果立竿见影。

    ``angle_offset`` 用来做多起点重试：整体旋转候选方向，换一组落点。
    """
    parent = coords[atom]
    best: tuple[float, tuple[float, float]] | None = None
    for angle in _BRANCH_ANGLES:
        direction = _unit(_rotate(base, angle + angle_offset))
        candidate = _add(parent, _scale(direction, BOND_LENGTH))
        clearance = min(
            (
                _norm(_sub(candidate, point))
                for index, point in coords.items()
                if index != atom
            ),
            default=math.inf,
        )
        if best is None or clearance > best[0]:
            best = (clearance, direction)
    assert best is not None
    return best[1]


def _place_chains(
    molecule: Molecule,
    coords: dict[int, tuple[float, float]],
    depth: dict[int, int],
    blocked: set[int] | None = None,
    angle_offset: float = 0.0,
) -> None:
    """从已摆放的原子出发，BFS 摆放剩余的链与取代基。

    每个待扩展的原子里，**延续主链的那一个取代基**单独选出来：优先选能长出
    最长链的碳，它按 ±60° 交替转角摆（形成锯齿骨架）；其余取代基按
    :data:`_BRANCH_ANGLES` 挂在旁边。这样带一堆羟基的分支链不会一边转
    一边绕回自己身上。

    ``depth`` 记录每个原子离起点有几根键，用来让链上的转角交替。

    ``blocked`` 是尚未摆放的环上的原子集合 —— 这些原子必须留给
    :func:`_try_attach_ring` 整体摆放，否则环会被拆散成链，键长也就乱了。
    """
    blocked = blocked or set()
    queue: deque[int] = deque(sorted(coords))
    queued = set(coords)
    while queue:
        atom = queue.popleft()
        if atom not in coords:
            continue
        pending = [
            n for n, _ in molecule.neighbors(atom)
            if n not in coords and n not in blocked
        ]
        if not pending:
            continue

        placed = [n for n, _ in molecule.neighbors(atom) if n in coords]

        # 选出主链延续方向
        if pending:
            seen = set(coords) | set(blocked)
            seen.add(atom)
            continuation = max(
                pending,
                key=lambda n: (
                    molecule.atoms[n].symbol == "C",
                    _subtree_size(molecule, n, atom, set(seen)),
                    -n,
                ),
            )
        else:  # pragma: no cover - pending 非空，保险起见
            continuation = None

        if not placed:
            base = (1.0, 0.0)
        elif len(placed) == 1:
            incoming = _unit(_sub(coords[atom], coords[placed[0]]))
            if continuation is not None and len(pending) == 1:
                # 纯链：交替 ±60°，相邻键夹角 60° ⇒ 键角 120°
                turn = _CHAIN_TURN if depth.get(atom, 0) % 2 == 0 else -_CHAIN_TURN
                base = _unit(_rotate(incoming, turn))
            else:
                # 有分支：主链沿原方向延伸，取代基再向两侧张开
                base = incoming
        else:
            reference = _centroid([coords[n] for n in placed])
            base = _unit(_sub(coords[atom], reference))
            if _norm(base) < _EPSILON:
                fallback = _unit(_sub(coords[atom], coords[placed[0]]))
                base = (-fallback[1], fallback[0])

        # 主链延续
        if continuation is not None:
            coords[continuation] = _add(
                coords[atom], _scale(_unit(base), BOND_LENGTH),
            )
            depth[continuation] = depth.get(atom, 0) + 1
            if continuation not in queued:
                queue.append(continuation)
                queued.add(continuation)

        # 其余取代基：相对主链方向张开，并避开已摆放的原子
        branches = [n for n in pending if n != continuation]
        for neighbor in branches:
            direction = _pick_branch_direction(coords, atom, base, angle_offset)
            coords[neighbor] = _add(
                coords[atom], _scale(_unit(direction), BOND_LENGTH),
            )
            depth[neighbor] = depth.get(atom, 0) + 1
            if neighbor not in queued:
                queue.append(neighbor)
                queued.add(neighbor)


# ---------------------------------------------------------------- 松弛


def _relax(
    molecule: Molecule,
    coords: dict[int, tuple[float, float]],
    iterations: int | None = None,
) -> None:
    """键长弹簧 + 非键斥力的迭代松弛。

    所有原子都参与移动：规整的环里键长已经是 1.0、非键距离又都大于阈值，
    受力为零，所以环不会被推歪；真正被修正的是桥环那种"两环都画成正
    多边形在平面上无解"的应变结构 —— 弹簧会把所有键长拉回 1.0。

    原子越多，需要的迭代次数越多（位移要一层层传开），所以迭代上限按
    原子数放大，并给出上限，避免大分子卡住。
    """
    atoms = sorted(coords)
    if len(atoms) < 2:
        return
    if iterations is None:
        iterations = min(2400, 400 + 60 * len(atoms))

    bonded = {_bond_key(bond.a, bond.b) for bond in molecule.bonds}
    pairs = [
        (left, right)
        for index, left in enumerate(atoms)
        for right in atoms[index + 1:]
        if _bond_key(left, right) not in bonded
    ]

    minimum = 0.92 * BOND_LENGTH
    bond_stiffness = 0.50
    repulsion = 0.85
    max_step = 0.25

    for _ in range(iterations):
        forces: dict[int, list[float]] = {atom: [0.0, 0.0] for atom in atoms}
        largest = 0.0

        for bond in molecule.bonds:
            left, right = bond.a, bond.b
            delta = _sub(coords[right], coords[left])
            length = _norm(delta)
            if length < 1e-6:
                coords[right] = _add(coords[right], (0.02, 0.0))
                continue
            magnitude = (length - BOND_LENGTH) * bond_stiffness
            direction = _scale(delta, 1.0 / length)
            forces[left][0] += direction[0] * magnitude
            forces[left][1] += direction[1] * magnitude
            forces[right][0] -= direction[0] * magnitude
            forces[right][1] -= direction[1] * magnitude
            largest = max(largest, abs(length - BOND_LENGTH))

        for left, right in pairs:
            delta = _sub(coords[left], coords[right])
            length = _norm(delta)
            if length >= minimum:
                continue
            if length < 1e-6:
                delta, length = (1e-6, 0.0), 1e-6
            magnitude = (minimum - length) * repulsion
            direction = _scale(delta, 1.0 / length)
            forces[left][0] += direction[0] * magnitude
            forces[left][1] += direction[1] * magnitude
            forces[right][0] -= direction[0] * magnitude
            forces[right][1] -= direction[1] * magnitude

        for atom in atoms:
            fx, fy = forces[atom]
            magnitude = math.hypot(fx, fy)
            if magnitude < 1e-9:
                continue
            if magnitude > max_step:
                fx, fy = fx / magnitude * max_step, fy / magnitude * max_step
            point = coords[atom]
            coords[atom] = (point[0] + fx, point[1] + fy)

        if largest < 0.002:
            break


# ---------------------------------------------------------------- 凯库勒化


def kekulize(molecule: Molecule, rings: list[list[int]]) -> set[frozenset[int]]:
    """为芳香环分配交替双键（凯库勒式），返回需要画成双键的键集合。

    规则与教材一致：芳香碳各需要一个双键；吡啶型氮（不带氢）可以参与双键；
    吡咯型氮（带氢）、呋喃氧、噻吩硫靠孤对电子参与共轭，不画双键。

    做法是在"可参与双键的芳香原子"上求一个尽量大的匹配。**不能简单地
    顺序贪心**：吡咯里按原子序数贪心会先配掉 C0=C1，剩下 C2 与 C4 各自
    只能连氮，结果只配出 1 根双键；正确解是 C0=C4 与 C1=C2。所以这里
    每次挑"可选伙伴最少"的原子先配 —— 先把死路走完，就能得到完整匹配。
    两种平局顺序各跑一遍取更优，保证确定性。
    """
    aromatic_present = any(atom.aromatic for atom in molecule.atoms)
    if not aromatic_present:
        return set()

    def matchable(index: int) -> bool:
        atom = molecule.atoms[index]
        if not atom.aromatic:
            return False
        if atom.symbol == "C":
            return True
        if atom.symbol == "N":
            return atom.hydrogens == 0 and atom.charge == 0
        return False

    candidates = {
        atom.index for atom in molecule.atoms if matchable(atom.index)
    }
    if not candidates:
        return set()

    def match_once(reverse_tie_break: bool) -> set[frozenset[int]]:
        doubles: set[frozenset[int]] = set()
        matched: set[int] = set()
        while True:
            choice: tuple[int, list[int]] | None = None
            best_degree = 99
            for index in sorted(candidates, reverse=reverse_tie_break):
                if index in matched:
                    continue
                partners = [
                    neighbor
                    for neighbor, order in molecule.neighbors(index)
                    if order == AROMATIC_BOND_ORDER
                    and neighbor in candidates
                    and neighbor not in matched
                ]
                if not partners:
                    continue
                if len(partners) < best_degree:
                    best_degree = len(partners)
                    choice = (index, partners)
                    if best_degree == 1:
                        break
            if choice is None:
                break
            atom, partners = choice
            partner = partners[0]
            doubles.add(frozenset({atom, partner}))
            matched.add(atom)
            matched.add(partner)
        return doubles

    forward = match_once(False)
    backward = match_once(True)
    return forward if len(forward) >= len(backward) else backward


# ---------------------------------------------------------------- 主入口


def _build_coords(
    molecule: Molecule,
    rings: list[list[int]],
    angle_offset: float,
) -> tuple[dict[int, tuple[float, float]], list[str]]:
    """按给定的取代基张角偏移排一次版，返回 ``(坐标, 提示)``。"""
    warnings: list[str] = []
    coords: dict[int, tuple[float, float]] = {}
    centers: list[tuple[float, float]] = []
    depth: dict[int, int] = {}
    pending = sorted(rings, key=len)

    # 先落下第一个环（或第一个原子），作为后续拼接的基准
    if pending:
        first = pending.pop(0)
        centers.append(_place_first_ring(first, coords, (0.0, 0.0)))
        for atom in first:
            depth[atom] = 0
    else:
        start = min(atom.index for atom in molecule.atoms)
        coords[start] = (0.0, 0.0)
        depth[start] = 0

    # 摆环与摆链交替进行，直到没有新进展
    progress = True
    while progress:
        progress = False
        remaining: list[list[int]] = []
        for ring in pending:
            center = _try_attach_ring(molecule, ring, coords, centers)
            if center is None:
                remaining.append(ring)
                continue
            centers.append(center)
            for atom in ring:
                depth.setdefault(atom, 0)
            progress = True
        pending = remaining

        # 尚未摆放的环上的原子不能交给摆链逻辑，否则环会被拆散
        blocked = {atom for ring in pending for atom in ring}
        before = len(coords)
        _place_chains(molecule, coords, depth, blocked, angle_offset)
        if len(coords) > before:
            progress = True

    # 仍然放不下的环：与主结构完全不相连（多组分），各自另起一块
    for order, ring in enumerate(pending):
        angle = math.radians(60.0 * order)
        direction = (math.cos(angle), math.sin(angle))
        offset = _scale(direction, 2.4 * (len(centers) + order + 1))
        centers.append(_place_first_ring(ring, coords, offset))
        for atom in ring:
            depth.setdefault(atom, 0)
        warnings.append("存在与主结构不相连的环系，已单独摆放")

    _place_chains(molecule, coords, depth, None, angle_offset)

    # 极端情形：还有孤立原子（例如 [Na+].[Cl-] 里已经处理过，这里是兜底）
    if coords:
        max_x = max(point[0] for point in coords.values())
        for atom in molecule.atoms:
            if atom.index not in coords:
                max_x += 1.6
                coords[atom.index] = (max_x, 0.0)

    _relax(molecule, coords)
    return coords, warnings


def _layout_score(
    molecule: Molecule,
    coords: dict[int, tuple[float, float]],
) -> tuple[int, float, float]:
    """给一次排版打分，越小越好。"""
    lengths = [
        _norm(_sub(coords[bond.a], coords[bond.b]))
        for bond in molecule.bonds
    ]
    deviation = max((abs(length - BOND_LENGTH) for length in lengths), default=0.0)
    minimum, pair = closest_nonbonded(Layout(coords=coords), molecule)
    if pair is None:
        minimum = 99.0
    if deviation <= 0.05 and minimum >= 0.70:
        rank = 0
    elif deviation <= 0.12 and minimum >= 0.55:
        rank = 1
    else:
        rank = 2
    # 先比等级，再比"键长尽量准"，最后比"原子尽量别挤"
    return rank, round(deviation, 4), -round(minimum, 4)


def compute_layout(molecule: Molecule) -> Layout:
    """为分子图生成二维坐标。

    排版是"贪心落点 + 力场松弛"，对分支很多的分子容易掉进局部最优。
    因此第一次失败时会用不同的取代基张角重试几次（确定性偏移，不用随机数），
    取最好的一次；仍然不达标就标记为 ``poor``，由上层拒绝出图。

    Args:
        molecule: SMILES 解析得到的分子图。

    Returns:
        :class:`Layout`，含坐标、环、双键分配与提示。
    """
    if not molecule.atoms:
        return Layout(coords={}, rings=[], warnings=["分子为空"])

    rings = find_sssr(molecule)

    #: 重试用的取代基张角偏移。原子太多时只跑第一次，避免耗时过长。
    offsets = [0.0]
    if len(molecule.atoms) <= 36:
        offsets += [37.0, -37.0, 71.0, -71.0]

    best: tuple[tuple[int, float, float], dict, list[str]] | None = None
    for offset in offsets:
        coords, warnings = _build_coords(molecule, rings, offset)
        score = _layout_score(molecule, coords)
        if best is None or score < best[0]:
            best = (score, coords, warnings)
        if score[0] == 0:
            break

    assert best is not None
    score, coords, warnings = best

    # 居中
    center = _centroid(list(coords.values()))
    coords = {atom: _sub(point, center) for atom, point in coords.items()}

    double_bonds = {
        frozenset({bond.a, bond.b})
        for bond in molecule.bonds
        if bond.order >= 2
    }
    double_bonds |= kekulize(molecule, rings)

    if len(rings) > 2:
        warnings.append(
            f"该分子含 {len(rings)} 个环，自动排版在复杂稠环上可能不够美观",
        )

    quality, issues = _assess_quality(molecule, coords)

    return Layout(
        coords=coords,
        rings=rings,
        double_bonds=double_bonds,
        warnings=warnings,
        quality=quality,
        issues=issues,
    )


def _assess_quality(
    molecule: Molecule,
    coords: dict[int, tuple[float, float]],
) -> tuple[str, list[str]]:
    """评估排版质量，决定这张图能不能出。"""
    issues: list[str] = []
    if len(coords) < 2:
        return "ok", issues

    lengths = [
        _norm(_sub(coords[bond.a], coords[bond.b]))
        for bond in molecule.bonds
    ]
    deviation = max((abs(length - BOND_LENGTH) for length in lengths), default=0.0)
    if deviation > 1e-6:
        issues.append(f"最大键长偏差 {deviation:.2f}（理想为 0）")

    minimum, pair = closest_nonbonded(Layout(coords=coords), molecule)
    if pair is not None and minimum < BOND_LENGTH:
        issues.append(f"最近的非成键原子距离 {minimum:.2f}（键长为 1.00）")

    if deviation <= 0.05 and (pair is None or minimum >= 0.70):
        return "ok", []
    if deviation <= 0.12 and (pair is None or minimum >= 0.55):
        return "approximate", issues
    return "poor", issues


def bond_orders_for_drawing(
    molecule: Molecule,
    layout: Layout,
) -> dict[frozenset[int], float]:
    """给每根键一个"用于绘制"的键级。

    芳香键若被凯库勒化选为双键则返回 2，否则返回 1 —— 键线式里芳香性
    是靠交替双键表现的，不存在 1.5 这种画法。
    """
    result: dict[frozenset[int], float] = {}
    for bond in molecule.bonds:
        key = frozenset({bond.a, bond.b})
        if bond.order == AROMATIC_BOND_ORDER:
            result[key] = 2.0 if key in layout.double_bonds else 1.0
        else:
            result[key] = float(bond.order)
    return result


def bond_length_stats(layout: Layout, molecule: Molecule) -> tuple[float, float]:
    """返回 ``(最短键长, 最长键长)``，用于自检排版质量。"""
    lengths = [
        _norm(_sub(layout.coords[bond.a], layout.coords[bond.b]))
        for bond in molecule.bonds
    ]
    if not lengths:
        return 0.0, 0.0
    return min(lengths), max(lengths)


def closest_nonbonded(
    layout: Layout,
    molecule: Molecule,
) -> tuple[float, tuple[int, int] | None]:
    """返回 ``(最近的非成键原子距离, 原子对)``，用于自检是否重叠。"""
    bonded = {_bond_key(bond.a, bond.b) for bond in molecule.bonds}
    best = math.inf
    pair: tuple[int, int] | None = None
    for left, right in itertools.combinations(sorted(layout.coords), 2):
        if _bond_key(left, right) in bonded:
            continue
        distance = _norm(_sub(layout.coords[left], layout.coords[right]))
        if distance < best:
            best, pair = distance, (left, right)
    return (best if pair else math.inf), pair
