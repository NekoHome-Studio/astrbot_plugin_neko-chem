"""化学核心库的离线测试。

不依赖 AstrBot，也不依赖 pytest：直接 ``python tests/test_chem.py`` 就能跑。
装了 pytest 的话也可以 ``pytest tests/test_chem.py``。

覆盖范围：

* 内置词表自检（SMILES 可解析、教材简式与 SMILES 的分子式一致）；
* 结构简式解析（下标、括号、结晶水、基团缩写、错误输入）；
* SMILES 解析（分子式、环数、隐式氢、错误输入）；
* 分子图 → 结构简式（对照教材写法）；
* 统一入口的四类输入识别；
* Pillow 渲染（尺寸、是否贴边、下标是否真的下沉）。
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chem  # noqa: E402
from chem import (  # noqa: E402
    FormulaSyntaxError,
    ResolveError,
    SmilesError,
    hill_formula,
    molar_mass,
    molecule_to_condensed,
    parse_condensed,
    parse_smiles,
    resolve,
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    """断言并记录失败，不中断后续用例。"""
    if not condition:
        FAILURES.append(message)
        print(f"  [失败] {message}")


def section(title: str) -> None:
    """打印分节标题。"""
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------- 词表自检


def test_name_database() -> None:
    """内置词表的每条 SMILES 都能解析，且教材简式与之分子式一致。"""
    section("内置词表自检")
    checked = 0
    for entry in chem.NAME_ENTRIES:
        if entry.smiles:
            try:
                molecule = parse_smiles(entry.smiles)
            except SmilesError as error:
                check(False, f"{entry.name} 的 SMILES 无法解析：{error.reason}")
                continue
            checked += 1
        else:
            molecule = None
        if entry.condensed:
            try:
                parsed = parse_condensed(entry.condensed)
            except FormulaSyntaxError as error:
                check(False, f"{entry.name} 的简式无法解析：{error.reason}")
                continue
            if molecule is not None and not parsed.indeterminate:
                want = hill_formula(
                    molecule.formula_counts(), molecule.total_charge(),
                )
                got = hill_formula(parsed.counts, parsed.charge)
                check(
                    want == got,
                    f"{entry.name} 的简式 {entry.condensed} 分子式为 {got}，"
                    f"但 SMILES 给出 {want}",
                )
    check(checked >= 100, f"词表可用条目过少：{checked}")
    print(f"  条目 {len(chem.NAME_ENTRIES)} 条，可解析 SMILES {checked} 条")
    check(chem.entry_count() == len(chem.NAME_ENTRIES), "entry_count 与词表长度不一致")


def test_name_lookup() -> None:
    """名称、别名、分子式反查。"""
    section("名称与分子式反查")
    check(chem.lookup_by_name("乙醇").name == "乙醇", "乙醇 未命中")
    check(chem.lookup_by_name("酒精").name == "乙醇", "别名 酒精 未命中")
    check(chem.lookup_by_name("甘油").name == "丙三醇", "别名 甘油 未命中")
    check(chem.lookup_by_name("TNT").name == "三硝基甲苯", "TNT 未命中")
    hits = [entry.name for entry in chem.lookup_by_formula("C2H6O")]
    check("乙醇" in hits and "甲醚" in hits, f"C2H6O 反查结果异常：{hits}")
    print(f"  C2H6O -> {hits}")


# ---------------------------------------------------------------- 简式解析


def test_condensed_parser() -> None:
    """简式解析的规范化、分子式与错误处理。"""
    section("结构简式解析")
    cases = [
        ("CH3CH2OH", "CH3CH2OH", "C2H6O"),
        ("CH3-CH2-OH", "CH3-CH2-OH", "C2H6O"),
        ("CH2=CH2", "CH2=CH2", "C2H4"),
        ("HC#CH", "HC≡CH", "C2H2"),
        ("CH3(CH2)4CH3", "CH3(CH2)4CH3", "C6H14"),
        ("CH2OH(CHOH)4CHO", "CH2OH(CHOH)4CHO", "C6H12O6"),
        ("CuSO4.5H2O", "CuSO4·5H2O", "CuH10O9S"),
        ("CH₃CH₂OH", "CH3CH2OH", "C2H6O"),
        ("NH4+", "NH4+", "H4N+"),
        ("CH3COO-", "CH3COO-", "C2H3O2-"),
        ("PhCH3", "C6H5CH3", "C7H8"),
        ("iPrOH", "CH(CH3)2OH", "C3H8O"),
        ("nBuOH", "CH2CH2CH2CH3OH", "C4H10O"),
    ]
    for text, ascii_text, formula in cases:
        parsed = parse_condensed(text)
        check(parsed.ascii_text == ascii_text,
              f"{text} 规范化为 {parsed.ascii_text}，应为 {ascii_text}")
        check(parsed.formula == formula,
              f"{text} 分子式为 {parsed.formula}，应为 {formula}")
    print(f"  通过 {len(cases)} 组规范化用例")

    # 结晶水的系数前缀与聚合度变量
    parsed = parse_condensed("(C6H10O5)n")
    check(parsed.indeterminate, "(C6H10O5)n 应被标记为分子式不确定")
    check(parsed.ascii_text == "(C6H10O5)n", "聚合度下标渲染异常")

    # 与分子式无法区分的写法要给出提示
    bare = parse_condensed("C2H6O")
    check(bare.is_bare_formula, "C2H6O 应被识别为与分子式相同")
    check(bool(bare.warnings), "C2H6O 应给出同分异构提示")

    errors = [
        "CH3(CH2",
        "CH3)CH3",
        "XxO2",
        "",
        "CH3[CH2]",
    ]
    for text in errors:
        try:
            parse_condensed(text)
        except FormulaSyntaxError:
            continue
        check(False, f"{text!r} 应当解析失败但没有")
    print(f"  通过 {len(errors)} 组错误输入用例")


# ---------------------------------------------------------------- SMILES


SMILE_FORMULA_CASES = [
    ("C", "CH4"), ("CC", "C2H6"), ("CCO", "C2H6O"), ("CC(=O)O", "C2H4O2"),
    ("C=C", "C2H4"), ("C#C", "C2H2"), ("c1ccccc1", "C6H6"),
    ("Cc1ccccc1", "C7H8"), ("CC(C)C", "C4H10"), ("CC(C)(C)C", "C5H12"),
    ("CCOC(C)=O", "C4H8O2"), ("OCC(O)CO", "C3H8O3"), ("C=O", "CH2O"),
    ("CC=O", "C2H4O"), ("CC(C)=O", "C3H6O"), ("C(=O)O", "CH2O2"),
    ("N", "H3N"), ("CCN", "C2H7N"), ("ClCCl", "CH2Cl2"),
    ("ClC(Cl)(Cl)Cl", "CCl4"), ("c1ccncc1", "C5H5N"),
    ("c1cc[nH]c1", "C4H5N"), ("c1ccc2ccccc2c1", "C10H8"),
    ("OCCO", "C2H6O2"), ("NC(N)=O", "CH4N2O"), ("C1CCCCC1", "C6H12"),
    ("C1=CC=CC=C1", "C6H6"), ("[Na+].[Cl-]", "ClNa"),
    ("OS(=O)(=O)O", "H2O4S"), ("O=C=O", "CO2"), ("N#N", "N2"),
    ("OCC1OC(O)C(O)C(O)C1O", "C6H12O6"),
]


def test_smiles_parser() -> None:
    """SMILES 解析出的分子式与标准值一致。"""
    section("SMILES 解析")
    for smiles, formula in SMILE_FORMULA_CASES:
        molecule = parse_smiles(smiles)
        got = hill_formula(molecule.formula_counts(), molecule.total_charge())
        check(got == formula, f"{smiles} 分子式为 {got}，应为 {formula}")
    print(f"  通过 {len(SMILE_FORMULA_CASES)} 组分子式用例")

    check(parse_smiles("c1ccccc1").ring_count == 1, "苯应含 1 个独立环")
    check(parse_smiles("c1ccc2ccccc2c1").ring_count == 2, "萘应含 2 个独立环")
    check(parse_smiles("CCO").ring_count == 0, "乙醇不应含环")
    check(parse_smiles("c1ccccc1").has_aromatic_ring, "苯应为芳香环")
    check(len(parse_smiles("c1ccccc1").find_cycles()[0]) == 6, "苯环应为 6 元环")

    errors = ["C(C", "C)", "C1CC", "Xx", "[Q]", "C=", "=C", "[C", "C%1"]
    for smiles in errors:
        try:
            parse_smiles(smiles)
        except SmilesError:
            continue
        check(False, f"{smiles!r} 应当解析失败但没有")
    print(f"  通过 {len(errors)} 组错误输入用例")


# ------------------------------------------------------- 分子图 → 结构简式


CONDENSE_CASES = [
    # 烷烃与支链
    ("C", "CH4"),
    ("CCO", "CH3CH2OH"),
    ("CC(C)C", "CH3CH(CH3)CH3"),
    ("CC(C)(C)C", "CH3C(CH3)2CH3"),
    # 醇
    ("OCCO", "HOCH2CH2OH"),
    ("OCC(O)CO", "HOCH2CH(OH)CH2OH"),
    ("CC(C)O", "CH3CH(OH)CH3"),
    # 烯烃与炔烃
    ("C=C", "CH2=CH2"),
    ("CC#C", "CH3C≡CH"),
    # 醛酮
    ("C=O", "HCHO"),
    ("CC=O", "CH3CHO"),
    ("CC(C)=O", "CH3COCH3"),
    # 羧酸与酯
    ("C(=O)O", "HCOOH"),
    ("CC(=O)O", "CH3COOH"),
    ("CCC(=O)O", "CH3CH2COOH"),
    ("CC(C)C(=O)O", "CH3CH(CH3)COOH"),
    ("CC(O)C(=O)O", "CH3CH(OH)COOH"),
    ("CCC(C)C(=O)O", "CH3CH2CH(CH3)COOH"),
    ("CC(C)CC(=O)O", "CH3CH(CH3)CH2COOH"),
    ("CCOC(C)=O", "CH3COOCH2CH3"),
    ("COC=O", "HCOOCH3"),
    ("CC(=O)OC", "CH3COOCH3"),
    # 腈与卤代烃
    ("CC#N", "CH3CN"),
    ("C=CC#N", "CH2=CHCN"),
    ("ClC(Cl)Cl", "CHCl3"),
    ("ClC(Cl)(Cl)Cl", "CCl4"),
    # 含氮
    ("CCN", "CH3CH2NH2"),
    ("NC(N)=O", "H2NCONH2"),
    # 苯系
    ("c1ccccc1", "C6H6"),
    ("Cc1ccccc1", "C6H5CH3"),
    ("Oc1ccccc1", "C6H5OH"),
    ("OC(=O)c1ccccc1", "C6H5COOH"),
    ("C=Cc1ccccc1", "C6H5CH=CH2"),
    ("O=[N+]([O-])c1ccccc1", "C6H5NO2"),
    # 无机小分子
    ("N", "NH3"),
    ("O", "H2O"),
    ("S", "H2S"),
    ("O=C=O", "CO2"),
]


def test_condense_generation() -> None:
    """分子图转简式，逐条对照教材写法。"""
    section("分子图 → 结构简式")
    for smiles, expected in CONDENSE_CASES:
        result = molecule_to_condensed(parse_smiles(smiles))
        check(result.ok, f"{smiles} 生成简式失败：{result.reason}")
        if result.ok:
            check(
                result.text == expected,
                f"{smiles} 生成 {result.text}，应为 {expected}",
            )
    print(f"  通过 {len(CONDENSE_CASES)} 组简式用例")

    # 环状结构应当明确拒绝，而不是给出错误答案
    for smiles in ("C1CCCCC1", "c1ccncc1", "c1ccc2ccccc2c1"):
        result = molecule_to_condensed(parse_smiles(smiles))
        check(not result.ok, f"{smiles} 是环状结构，不应自动给出连写式简式")
        check(bool(result.reason), f"{smiles} 被拒绝时应给出原因")
    print("  环状结构按预期拒绝并给出原因")


# ---------------------------------------------------------------- 统一入口


RESOLVE_CASES = [
    # (输入, 期望识别类型, 期望名称或 None, 期望分子式)
    ("乙醇", "name", "乙醇", "C2H6O"),
    ("酒精", "name", "乙醇", "C2H6O"),
    ("甘油", "name", "丙三醇", "C3H8O3"),
    ("葡萄糖", "name", "葡萄糖", "C6H12O6"),
    ("淀粉", "name", "淀粉", None),
    ("CCO", "smiles", None, "C2H6O"),
    ("c1ccccc1", "smiles", None, "C6H6"),
    ("CC(=O)O", "smiles", None, "C2H4O2"),
    ("CC(C)C(=O)O", "smiles", None, "C4H8O2"),
    ("C1CCCCC1", "smiles", None, "C6H12"),
    ("CH3CH2OH", "condensed", None, "C2H6O"),
    ("CH2=CH2", "condensed", None, "C2H4"),
    ("CH3(CH2)4CH3", "condensed", None, "C6H14"),
    ("CH3CH(CH3)COOH", "condensed", None, "C4H8O2"),
    ("CuSO4·5H2O", "formula", "胆矾", "CuH10O9S"),
    ("C2H6O", "formula", "乙醇", "C2H6O"),
    ("C6H6", "formula", "苯", "C6H6"),
    ("NaCl", "formula", "氯化钠", "ClNa"),
    ("KMnO4", "formula", "高锰酸钾", "KMnO4"),
    ("CCl4", "formula", "四氯化碳", "CCl4"),
]


def test_resolver() -> None:
    """四类输入的识别结果。"""
    section("统一入口识别")
    for text, kind, name, formula in RESOLVE_CASES:
        try:
            compound = resolve(text)
        except ResolveError as error:
            check(False, f"{text} 解析失败：{error.reason}")
            continue
        check(
            compound.source_kind == kind,
            f"{text} 识别为 {compound.source_kind}，应为 {kind}",
        )
        if name is not None:
            check(
                compound.name == name,
                f"{text} 匹配到 {compound.name}，应为 {name}",
            )
        if formula is not None:
            check(
                compound.formula == formula,
                f"{text} 分子式为 {compound.formula}，应为 {formula}",
            )
    print(f"  通过 {len(RESOLVE_CASES)} 组识别用例")

    # 同分异构体提示
    compound = resolve("C2H6O")
    check("甲醚" in compound.candidates, "C2H6O 应提示同分异构体 甲醚")

    # 错误输入
    for text in ("NOPE", "香蕉", ""):
        try:
            resolve(text)
        except ResolveError:
            continue
        check(False, f"{text!r} 应当解析失败但没有")
    print("  错误输入按预期拒绝")

    # 相对分子质量
    ethanol = resolve("乙醇")
    check(
        abs((ethanol.molar_mass or 0) - 46.07) < 0.02,
        f"乙醇相对分子质量异常：{ethanol.molar_mass}",
    )
    check(
        abs(molar_mass(ethanol.counts) - 46.069) < 0.01,
        "molar_mass 计算异常",
    )


# ---------------------------------------------------------------- 渲染


def test_rendering() -> None:
    """Pillow 渲染的尺寸、边距与下标位置。"""
    section("Pillow 图片渲染")
    try:
        from PIL import Image
    except ImportError:
        print("  跳过：未安装 pillow")
        return

    from chem.render_png import RenderOptions, describe_font_setup, render_condensed_png

    print(f"  {describe_font_setup()}")

    samples = [
        "CH3CH2OH",
        "HOCH2CH(OH)CH2OH",
        "CH2OH(CHOH)4CHO",
        "CuSO4·5H2O",
        "HC≡CH",
        "CH3COO-",
        "(C6H10O5)n",
    ]
    for text in samples:
        parsed = parse_condensed(text)
        png = render_condensed_png(parsed.nodes, RenderOptions())
        check(png.startswith(b"\x89PNG"), f"{text} 渲染结果不是 PNG")
        image = Image.open(io.BytesIO(png)).convert("L")
        width, height = image.size
        check(width > 40 and height > 20, f"{text} 图片尺寸异常：{width}x{height}")
        pixels = image.load()
        columns = [
            x for x in range(width)
            if any(pixels[x, y] < 200 for y in range(height))
        ]
        rows = [
            y for y in range(height)
            if any(pixels[x, y] < 200 for x in range(width))
        ]
        check(bool(columns) and bool(rows), f"{text} 渲染结果为空图")
        if columns and rows:
            check(
                columns[0] > 0 and columns[-1] < width - 1,
                f"{text} 内容横向贴边，可能被裁切",
            )
            check(
                rows[0] > 0 and rows[-1] < height - 1,
                f"{text} 内容纵向贴边，可能被裁切",
            )
    print(f"  通过 {len(samples)} 组渲染用例")

    # 带标题与说明的版式
    options = RenderOptions(title="乙醇", subtitle="C2H6O　M = 46.07")
    png = render_condensed_png(parse_condensed("CH3CH2OH").nodes, options)
    image = Image.open(io.BytesIO(png))
    plain = render_condensed_png(parse_condensed("CH3CH2OH").nodes, RenderOptions())
    check(
        image.size[1] > Image.open(io.BytesIO(plain)).size[1],
        "带标题/说明的图片应当更高",
    )

    # 透明背景
    png = render_condensed_png(
        parse_condensed("H2O").nodes, RenderOptions(background=None),
    )
    check(
        Image.open(io.BytesIO(png)).mode == "RGBA",
        "透明背景应输出 RGBA",
    )

    # 下标必须真的落在主基线下方
    png = render_condensed_png(
        parse_condensed("CH3").nodes, RenderOptions(padding=20),
    )
    image = Image.open(io.BytesIO(png)).convert("L")
    width, height = image.size
    pixels = image.load()
    inked = [
        x for x in range(width)
        if any(pixels[x, y] < 200 for y in range(height))
    ]
    groups: list[tuple[int, int]] = []
    start = previous = inked[0]
    for x in inked[1:]:
        if x - previous > 1:
            groups.append((start, previous))
            start = x
        previous = x
    groups.append((start, previous))
    check(len(groups) == 3, f"CH3 应切分出 3 个字形，实际 {len(groups)}")
    if len(groups) == 3:
        spans = []
        for left, right in groups:
            ys = [
                y for y in range(height)
                if any(pixels[x, y] < 200 for x in range(left, right + 1))
            ]
            spans.append((ys[0], ys[-1]))
        check(
            spans[2][0] > spans[0][0] and spans[2][1] > spans[0][1],
            f"下标 3 应当整体低于字符 C：{spans}",
        )
        check(
            (spans[2][1] - spans[2][0]) < (spans[0][1] - spans[0][0]),
            f"下标 3 应当比字符 C 更小：{spans}",
        )
        print(f"  下标位置检查通过：{spans}")


# ---------------------------------------------------------------- 键线式


#: 应当排得规整（键长误差小、原子不重叠）的分子。
LAYOUT_GOOD_CASES = [
    "c1ccccc1", "C1CCCCC1", "C1CC1", "C1CCCC1", "C1CCCCCC1",
    "c1ccncc1", "c1cc[nH]c1", "c1ccoc1", "c1ccsc1",
    "c1ccc2ccccc2c1", "c1ccc2cc3ccccc3cc2c1",
    "Cc1ccccc1", "Oc1ccccc1", "OC(=O)c1ccccc1", "C=Cc1ccccc1",
    "O=[N+]([O-])c1ccccc1", "Cc1ccc(C)cc1",
    "c1ccccc1-c1ccccc1", "c1ccccc1CCc1ccccc1", "C1CC1C1CC1",
    "CCOC(C)=O", "CCCCCC", "CC(C)C(=O)O", "OCC(O)CO", "CC(=O)O",
    "N#N", "O=C=O",
    "OCC1OC(O)C(O)C(O)C1O",           # 葡萄糖环状
    "OCC(O)C(O)C(O)C(O)C=O",           # 葡萄糖开链（分支多）
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",      # 柠檬酸（季碳 + 三个分支）
    "CC(=O)Nc1ccc(O)cc1",              # 对乙酰氨基酚
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",    # 咖啡因（稠合杂环）
]


def test_layout() -> None:
    """二维排版：键长、环规整性与重叠检查。"""
    section("键线式二维排版")
    from chem.layout import bond_length_stats, closest_nonbonded, compute_layout

    for smiles in LAYOUT_GOOD_CASES:
        molecule = parse_smiles(smiles)
        layout = compute_layout(molecule)
        check(layout.drawable, f"{smiles} 排版被判为不可绘制：{layout.issues}")
        low, high = bond_length_stats(layout, molecule)
        check(
            low > 0.95 and high < 1.05,
            f"{smiles} 键长异常：[{low:.3f}, {high:.3f}]",
        )
        minimum, pair = closest_nonbonded(layout, molecule)
        if pair is not None:
            check(
                minimum >= 0.90,
                f"{smiles} 原子 {pair} 过近：{minimum:.3f}",
            )
        # 所有重原子都应当被定位
        check(
            len(layout.coords) == len(molecule.atoms),
            f"{smiles} 有原子没有坐标",
        )
    print(f"  通过 {len(LAYOUT_GOOD_CASES)} 组排版用例")

    # 正多边形：苯与环己烷的六个键角都应当是 120°
    import math as _math

    for smiles in ("c1ccccc1", "C1CCCCC1"):
        molecule = parse_smiles(smiles)
        layout = compute_layout(molecule)
        ring = layout.rings[0]
        angles = []
        for index, atom in enumerate(ring):
            previous = layout.coords[ring[index - 1]]
            current = layout.coords[atom]
            following = layout.coords[ring[(index + 1) % len(ring)]]
            v1 = (previous[0] - current[0], previous[1] - current[1])
            v2 = (following[0] - current[0], following[1] - current[1])
            cosine = (v1[0] * v2[0] + v1[1] * v2[1]) / (
                _math.hypot(*v1) * _math.hypot(*v2)
            )
            angles.append(_math.degrees(_math.acos(max(-1.0, min(1.0, cosine)))))
        check(
            all(abs(angle - 120.0) < 0.5 for angle in angles),
            f"{smiles} 环内键角不是 120°：{[round(a, 2) for a in angles]}",
        )
    print("  苯与环己烷的环内键角均为 120°")

    # 锯齿链：己烷相邻两根键的夹角应当接近 60°（即键角 120°）
    molecule = parse_smiles("CCCCCC")
    layout = compute_layout(molecule)
    turns = []
    for atom in range(1, 5):
        v1 = (
            layout.coords[atom][0] - layout.coords[atom - 1][0],
            layout.coords[atom][1] - layout.coords[atom - 1][1],
        )
        v2 = (
            layout.coords[atom + 1][0] - layout.coords[atom][0],
            layout.coords[atom + 1][1] - layout.coords[atom][1],
        )
        cosine = (v1[0] * v2[0] + v1[1] * v2[1]) / (
            _math.hypot(*v1) * _math.hypot(*v2)
        )
        turns.append(
            _math.degrees(_math.acos(max(-1.0, min(1.0, cosine)))),
        )
    check(
        all(abs(turn - 60.0) < 1.0 for turn in turns),
        f"己烷链不是锯齿形：相邻键夹角 {[round(t, 2) for t in turns]}",
    )
    print(f"  己烷链为锯齿形，相邻键夹角 {[round(t, 1) for t in turns]}")

    # 凯库勒化：苯应当正好 3 根双键，吡咯是 2 根
    benzene = parse_smiles("c1ccccc1")
    layout = compute_layout(benzene)
    ring_doubles = [
        bond for bond in benzene.bonds
        if frozenset({bond.a, bond.b}) in layout.double_bonds
    ]
    check(len(ring_doubles) == 3, f"苯的凯库勒式应有 3 根双键，实际 {len(ring_doubles)}")

    pyrrole = parse_smiles("c1cc[nH]c1")
    layout = compute_layout(pyrrole)
    pyrrole_doubles = [
        bond for bond in pyrrole.bonds
        if frozenset({bond.a, bond.b}) in layout.double_bonds
    ]
    check(
        len(pyrrole_doubles) == 2,
        f"吡咯的凯库勒式应有 2 根双键，实际 {len(pyrrole_doubles)}",
    )
    print("  凯库勒化正确（苯 3 根双键、吡咯 2 根双键）")

    # 几何上无解的结构应当被拒绝，而不是画出错误结果
    for smiles in ("C1CC2CCC1C2",):
        layout = compute_layout(parse_smiles(smiles))
        check(
            not layout.drawable,
            f"{smiles} 是桥环，二维无解，应当被拒绝出图",
        )
        check(bool(layout.issues), f"{smiles} 被拒绝时应当给出原因")
    print("  桥环结构按预期拒绝并给出原因")


def test_skeletal_rendering() -> None:
    """键线式图片渲染。"""
    section("键线式图片渲染")
    try:
        from PIL import Image
    except ImportError:
        print("  跳过：未安装 pillow")
        return

    from chem.layout import compute_layout
    from chem.render_skeletal import (
        SkeletalOptions,
        atom_label,
        render_skeletal_png,
    )

    samples = [
        "c1ccccc1", "C1CCCCC1", "c1ccncc1", "CCO", "CC(=O)O",
        "O=[N+]([O-])c1ccccc1", "c1ccc2ccccc2c1", "OCC1OC(O)C(O)C(O)C1O",
        "[Na+].[Cl-]",
    ]
    inks: dict[str, int] = {}
    for smiles in samples:
        molecule = parse_smiles(smiles)
        layout = compute_layout(molecule)
        png = render_skeletal_png(molecule, layout, SkeletalOptions())
        check(png.startswith(b"\x89PNG"), f"{smiles} 渲染结果不是 PNG")
        image = Image.open(io.BytesIO(png)).convert("RGB")
        width, height = image.size
        check(width > 40 and height > 40, f"{smiles} 图片尺寸异常：{width}x{height}")
        pixels = image.load()
        ink = sum(
            1
            for y in range(height)
            for x in range(width)
            if pixels[x, y] != (255, 255, 255)
        )
        inks[smiles] = ink
        check(ink > 200, f"{smiles} 渲染结果几乎是空白（墨迹 {ink} 像素）")
        columns = [
            x for x in range(width)
            if any(pixels[x, y] != (255, 255, 255) for y in range(height))
        ]
        rows = [
            y for y in range(height)
            if any(pixels[x, y] != (255, 255, 255) for x in range(width))
        ]
        check(
            columns[0] > 0 and columns[-1] < width - 1,
            f"{smiles} 内容横向贴边，可能被裁切",
        )
        check(
            rows[0] > 0 and rows[-1] < height - 1,
            f"{smiles} 内容纵向贴边，可能被裁切",
        )
    print(f"  通过 {len(samples)} 组渲染用例")

    # 苯应当比环己烷多出 3 条内侧双键线
    check(
        inks["c1ccccc1"] > inks["C1CCCCC1"],
        "苯应当比环己烷多画 3 条双键线",
    )
    print(f"  双键线已画出（苯 {inks['c1ccccc1']} > 环己烷 {inks['C1CCCCC1']} 墨迹像素)")

    # 杂原子标注：乙醇的羟基氧应当产生红色像素，且位置就在氧原子上
    molecule = parse_smiles("CCO")
    layout = compute_layout(molecule)
    options = SkeletalOptions()
    png = render_skeletal_png(molecule, layout, options)
    image = Image.open(io.BytesIO(png)).convert("RGB")
    width, height = image.size
    pixels = image.load()
    reds = [
        (x, y)
        for y in range(height)
        for x in range(width)
        if pixels[x, y][0] > 150 and pixels[x, y][1] < 90 and pixels[x, y][2] < 90
    ]
    check(bool(reds), "乙醇的氧原子应当有红色标注")
    if reds:
        centroid = (
            sum(p[0] for p in reds) / len(reds),
            sum(p[1] for p in reds) / len(reds),
        )
        # 复刻渲染器里的坐标变换，验证标签画在原子上
        scale = options.supersample
        bond = options.bond_length * scale
        pad = options.padding * scale
        margin = int(bond * options.font_scale) * 0.9
        min_x, min_y, max_x, max_y = layout.bounds()
        oxygen = layout.coords[2]
        expected = (
            ((oxygen[0] - min_x) * bond + pad + margin) / scale,
            ((max_y - oxygen[1]) * bond + pad + margin) / scale,
        )
        distance = (
            (centroid[0] - expected[0]) ** 2 + (centroid[1] - expected[1]) ** 2
        ) ** 0.5
        check(distance < 8, f"氧标签偏离原子位置 {distance:.1f}px")
        print(f"  杂原子标签定位准确（偏离 {distance:.1f}px）")

    # 标签内容规则
    ethanol = parse_smiles("CCO")
    check(atom_label(ethanol, 0) == ("", 0, ""), "碳原子不应标注符号")
    check(atom_label(ethanol, 2)[0] == "O", "氧原子应当标注 O")
    check(atom_label(ethanol, 2)[1] == 1, "羟基应当写出 1 个氢")

    acetate = parse_smiles("CC(=O)[O-]")
    negative = [
        atom_label(acetate, atom.index)
        for atom in acetate.atoms
        if atom.charge < 0
    ]
    check(
        any(charge == "-" for _, _, charge in negative),
        "乙酸根的负电荷应当被标注",
    )

    # 排版不可用的分子必须拒绝出图
    from chem.render_skeletal import SkeletalRenderError

    broken = parse_smiles("C1CC2CCC1C2")
    try:
        render_skeletal_png(broken, compute_layout(broken), SkeletalOptions())
    except SkeletalRenderError:
        print("  排版不可用时按预期拒绝出图")
    else:
        check(False, "排版不可用的分子应当拒绝出图")


def main() -> int:
    """跑完全部用例并汇总。"""
    print("化学结构简式插件 离线自检")
    test_name_database()
    test_name_lookup()
    test_condensed_parser()
    test_smiles_parser()
    test_condense_generation()
    test_resolver()
    test_rendering()
    test_layout()
    test_skeletal_rendering()

    print("\n" + "=" * 46)
    if FAILURES:
        print(f"失败 {len(FAILURES)} 项：")
        for item in FAILURES:
            print(f"  · {item}")
        return 1
    print("全部用例通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
