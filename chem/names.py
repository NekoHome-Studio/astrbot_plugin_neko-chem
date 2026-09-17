"""常见物质的中文名/俗名 → SMILES / 结构简式 数据表。

数据来源与取舍
--------------
* 结构与 SMILES 均为公开的化学常识，参照中学教材与教材配套资料整理；
* 没有引用任何第三方代码或数据文件，全部为手工录入的通识条目；
* ``condensed`` 字段用于覆盖自动推导结果，存放**教材通用写法**，
  例如葡萄糖写作 ``CH2OH(CHOH)4CHO``、蔗糖直接写分子式；
* 高分子与组成不定的物质（淀粉、纤维素）没有确定 SMILES，
  ``smiles`` 置空、只给出 ``formula``。

类别字段仅用于 ``/chem 列表`` 的分类展示。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# (名称, SMILES, 教材简式覆盖, 别名, 类别)
_RAW_ENTRIES: tuple[tuple[str, str | None, str | None, tuple[str, ...], str], ...] = (
    # ---------------- 烷烃 / 环烷烃
    ("甲烷", "C", None, ("沼气",), "烷烃"),
    ("乙烷", "CC", None, (), "烷烃"),
    ("丙烷", "CCC", None, (), "烷烃"),
    ("正丁烷", "CCCC", None, ("丁烷",), "烷烃"),
    ("异丁烷", "CC(C)C", None, ("2-甲基丙烷",), "烷烃"),
    ("正戊烷", "CCCCC", None, ("戊烷",), "烷烃"),
    ("异戊烷", "CC(C)CC", None, ("2-甲基丁烷",), "烷烃"),
    ("新戊烷", "CC(C)(C)C", None, ("2,2-二甲基丙烷",), "烷烃"),
    ("正己烷", "CCCCCC", None, ("己烷",), "烷烃"),
    ("正庚烷", "CCCCCCC", None, ("庚烷",), "烷烃"),
    ("正辛烷", "CCCCCCCC", None, ("辛烷",), "烷烃"),
    ("环丙烷", "C1CC1", None, (), "环烷烃"),
    ("环丁烷", "C1CCC1", None, (), "环烷烃"),
    ("环戊烷", "C1CCCC1", None, (), "环烷烃"),
    ("环己烷", "C1CCCCC1", None, (), "环烷烃"),
    ("甲基环己烷", "CC1CCCCC1", None, (), "环烷烃"),
    # ---------------- 烯烃 / 炔烃
    ("乙烯", "C=C", None, (), "烯烃"),
    ("丙烯", "CC=C", "CH2=CHCH3", ("甲基乙烯",), "烯烃"),
    ("1-丁烯", "CCC=C", None, ("正丁烯",), "烯烃"),
    ("2-丁烯", "CC=CC", None, (), "烯烃"),
    ("异丁烯", "CC(=C)C", "CH2=C(CH3)2", ("2-甲基丙烯",), "烯烃"),
    ("1,3-丁二烯", "C=CC=C", None, ("丁二烯",), "二烯烃"),
    ("异戊二烯", "CC(=C)C=C", "CH2=C(CH3)CH=CH2", ("2-甲基-1,3-丁二烯",), "二烯烃"),
    ("环己烯", "C1=CCCCC1", None, (), "烯烃"),
    ("乙炔", "C#C", "HC≡CH", ("电石气",), "炔烃"),
    ("丙炔", "CC#C", None, ("甲基乙炔",), "炔烃"),
    ("1-丁炔", "CCC#C", None, (), "炔烃"),
    ("2-丁炔", "CC#CC", None, (), "炔烃"),
    # ---------------- 芳香族
    ("苯", "c1ccccc1", None, (), "芳香烃"),
    ("甲苯", "Cc1ccccc1", None, ("甲基苯",), "芳香烃"),
    ("乙苯", "CCc1ccccc1", None, (), "芳香烃"),
    ("邻二甲苯", "Cc1ccccc1C", None, ("1,2-二甲苯",), "芳香烃"),
    ("间二甲苯", "Cc1cccc(C)c1", None, ("1,3-二甲苯",), "芳香烃"),
    ("对二甲苯", "Cc1ccc(C)cc1", None, ("1,4-二甲苯",), "芳香烃"),
    ("苯乙烯", "C=Cc1ccccc1", None, ("乙烯基苯",), "芳香烃"),
    ("苯乙炔", "C#Cc1ccccc1", None, (), "芳香烃"),
    ("萘", "c1ccc2ccccc2c1", None, ("萘球",), "芳香烃"),
    ("蒽", "c1ccc2cc3ccccc3cc2c1", None, (), "芳香烃"),
    ("苯酚", "Oc1ccccc1", None, ("石炭酸",), "酚"),
    ("苯甲醇", "OCc1ccccc1", None, ("苄醇",), "醇"),
    ("苯甲酸", "OC(=O)c1ccccc1", None, ("安息香酸",), "羧酸"),
    ("苯甲醛", "O=Cc1ccccc1", None, ("安息香醛",), "醛"),
    ("苯胺", "Nc1ccccc1", None, ("阿尼林油",), "胺"),
    ("硝基苯", "O=[N+]([O-])c1ccccc1", None, (), "硝基化合物"),
    ("氯苯", "Clc1ccccc1", None, (), "卤代烃"),
    ("溴苯", "Brc1ccccc1", None, (), "卤代烃"),
    ("苯磺酸", "OS(=O)(=O)c1ccccc1", None, (), "磺酸"),
    ("三硝基甲苯", "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]",
     None, ("TNT", "梯恩梯"), "硝基化合物"),
    ("水杨酸", "OC(=O)c1ccccc1O", None, ("邻羟基苯甲酸",), "羧酸"),
    ("阿司匹林", "CC(=O)Oc1ccccc1C(=O)O", None,
     ("乙酰水杨酸", "阿斯匹林"), "酯"),
    ("对乙酰氨基酚", "CC(=O)Nc1ccc(O)cc1", None,
     ("扑热息痛", "醋氨酚"), "酰胺"),
    ("苯甲酸钠", "[Na+].[O-]C(=O)c1ccccc1", None, ("安息香酸钠",), "羧酸盐"),
    ("咖啡因", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", None, ("咖啡碱",), "生物碱"),
    ("抗坏血酸", "OCC(O)C1OC(=O)C(O)=C1O", None, ("维生素C", "维C"), "维生素"),
    # ---------------- 卤代烃
    ("氯甲烷", "CCl", None, ("一氯甲烷", "甲基氯"), "卤代烃"),
    ("二氯甲烷", "ClCCl", None, (), "卤代烃"),
    ("三氯甲烷", "ClC(Cl)Cl", None, ("氯仿",), "卤代烃"),
    ("四氯化碳", "ClC(Cl)(Cl)Cl", None, ("四氯甲烷",), "卤代烃"),
    ("碘甲烷", "CI", None, ("甲基碘",), "卤代烃"),
    ("氯乙烷", "CCCl", None, ("乙基氯",), "卤代烃"),
    ("溴乙烷", "CCBr", None, ("乙基溴",), "卤代烃"),
    ("氯乙烯", "C=CCl", None, ("乙烯基氯",), "卤代烃"),
    ("四氟乙烯", "C(=C(F)F)(F)F", "CF2=CF2", (), "卤代烃"),
    ("1,2-二溴乙烷", "BrCCBr", None, (), "卤代烃"),
    # ---------------- 醇 / 酚 / 醚
    ("甲醇", "CO", None, ("木醇", "木精"), "醇"),
    ("乙醇", "CCO", None, ("酒精",), "醇"),
    ("正丙醇", "CCCO", None, ("1-丙醇",), "醇"),
    ("异丙醇", "CC(C)O", None, ("2-丙醇",), "醇"),
    ("正丁醇", "CCCCO", None, ("1-丁醇",), "醇"),
    ("异丁醇", "CC(C)CO", None, ("2-甲基-1-丙醇",), "醇"),
    ("叔丁醇", "CC(C)(C)O", None, ("2-甲基-2-丙醇",), "醇"),
    ("乙二醇", "OCCO", None, ("甘醇",), "醇"),
    ("丙三醇", "OCC(O)CO", None, ("甘油",), "醇"),
    ("环己醇", "OC1CCCCC1", None, (), "醇"),
    ("甲醚", "COC", None, ("二甲醚",), "醚"),
    ("乙醚", "CCOCC", None, ("二乙醚", "乙氧基乙烷"), "醚"),
    ("甲乙醚", "COCC", None, ("甲氧基乙烷",), "醚"),
    ("环氧乙烷", "C1CO1", None, ("氧化乙烯",), "醚"),
    # ---------------- 醛 / 酮
    ("甲醛", "C=O", "HCHO", ("蚁醛",), "醛"),
    ("乙醛", "CC=O", None, ("醋醛",), "醛"),
    ("丙醛", "CCC=O", None, (), "醛"),
    ("丙烯醛", "C=CC=O", None, ("2-丙烯醛",), "醛"),
    ("丁醛", "CCCC=O", None, (), "醛"),
    ("丙酮", "CC(C)=O", None, ("二甲基酮",), "酮"),
    ("丁酮", "CCC(C)=O", None, ("2-丁酮", "甲乙酮"), "酮"),
    ("环己酮", "O=C1CCCCC1", None, (), "酮"),
    # ---------------- 羧酸 / 酯
    ("甲酸", "C(=O)O", "HCOOH", ("蚁酸",), "羧酸"),
    ("乙酸", "CC(=O)O", None, ("醋酸", "冰醋酸"), "羧酸"),
    ("丙酸", "CCC(=O)O", None, (), "羧酸"),
    ("丁酸", "CCCC(=O)O", None, ("酪酸",), "羧酸"),
    ("乙二酸", "OC(=O)C(=O)O", "HOOCCOOH", ("草酸",), "羧酸"),
    ("丙二酸", "OC(=O)CC(=O)O", None, (), "羧酸"),
    ("丁二酸", "OC(=O)CCC(=O)O", None, ("琥珀酸",), "羧酸"),
    ("丙烯酸", "C=CC(=O)O", None, (), "羧酸"),
    ("乳酸", "CC(O)C(=O)O", None, ("2-羟基丙酸",), "羧酸"),
    ("柠檬酸", "OC(=O)CC(O)(CC(=O)O)C(=O)O", None, ("枸橼酸",), "羧酸"),
    ("硬脂酸", "CCCCCCCCCCCCCCCCCC(=O)O", None, ("十八酸",), "羧酸"),
    ("油酸", "CCCCCCCCC=CCCCCCCCC(=O)O", None, ("顺-9-十八烯酸",), "羧酸"),
    ("乙酸酐", "CC(=O)OC(C)=O", None, ("醋酐",), "酸酐"),
    ("甲酸甲酯", "COC=O", "HCOOCH3", (), "酯"),
    ("甲酸乙酯", "CCOC=O", "HCOOCH2CH3", (), "酯"),
    ("乙酸甲酯", "COC(C)=O", None, (), "酯"),
    ("乙酸乙酯", "CCOC(C)=O", None, ("醋酸乙酯",), "酯"),
    ("乙酸丙酯", "CCCOC(C)=O", None, (), "酯"),
    ("乙酸异戊酯", "CC(C)CCOC(C)=O", None, ("香蕉水", "香蕉油"), "酯"),
    ("三硬脂酸甘油酯", "CCCCCCCCCCCCCCCCCC(=O)OCC(OC(=O)CCCCCCCCCCCCCCCCC)"
     "COC(=O)CCCCCCCCCCCCCCCCC", None, ("硬脂酸甘油酯", "硬脂"), "油脂"),
    # ---------------- 含氮化合物
    ("尿素", "NC(N)=O", "CO(NH2)2", ("脲", "碳酰胺"), "酰胺"),
    ("甲胺", "CN", None, ("一甲胺",), "胺"),
    ("二甲胺", "CNC", None, (), "胺"),
    ("三甲胺", "CN(C)C", None, (), "胺"),
    ("乙胺", "CCN", None, ("氨基乙烷",), "胺"),
    ("乙二胺", "NCCN", None, (), "胺"),
    ("硝基乙烷", "CC[N+](=O)[O-]", None, (), "硝基化合物"),
    ("乙腈", "CC#N", None, ("甲基氰", "氰基甲烷"), "腈"),
    ("丙烯腈", "C=CC#N", None, ("氰基乙烯",), "腈"),
    ("甘氨酸", "NCC(=O)O", None, ("氨基乙酸",), "氨基酸"),
    ("丙氨酸", "CC(N)C(=O)O", None, ("2-氨基丙酸",), "氨基酸"),
    ("苯丙氨酸", "NC(Cc1ccccc1)C(=O)O", None, (), "氨基酸"),
    # ---------------- 糖类与高分子
    ("葡萄糖", "OCC(O)C(O)C(O)C(O)C=O", "CH2OH(CHOH)4CHO", ("右旋糖", "血糖"), "单糖"),
    ("果糖", "OCC(O)C(O)C(O)C(=O)CO", "CH2OH(CHOH)3COCH2OH", ("左旋糖",), "单糖"),
    ("蔗糖", "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)"
     "[C@@H](O)[C@@H]1O", "C12H22O11", ("食糖", "白糖"), "二糖"),
    ("麦芽糖", "OC[C@H]1O[C@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)"
     "[C@H](O)[C@@H](O)[C@@H]1O", "C12H22O11", ("饴糖",), "二糖"),
    ("淀粉", None, "(C6H10O5)n", ("生粉", "芡粉"), "多糖"),
    ("纤维素", None, "(C6H10O5)n", (), "多糖"),
    # ---------------- 常见无机物
    ("水", "O", "H2O", (), "无机物"),
    ("二氧化碳", "O=C=O", "CO2", ("碳酸气", "干冰"), "无机物"),
    ("一氧化碳", "[C-]#[O+]", "CO", ("煤气",), "无机物"),
    ("氨", "N", "NH3", ("氨气",), "无机物"),
    ("硫化氢", "S", "H2S", (), "无机物"),
    ("硫酸", "OS(=O)(=O)O", "H2SO4", (), "无机物"),
    ("硝酸", "O[N+](=O)[O-]", "HNO3", (), "无机物"),
    ("磷酸", "OP(=O)(O)O", "H3PO4", (), "无机物"),
    ("氢氧化钠", "[Na+].[OH-]", "NaOH", ("烧碱", "火碱", "苛性钠"), "无机物"),
    ("氢氧化钙", "[Ca+2].[OH-].[OH-]", "Ca(OH)2", ("熟石灰", "消石灰"), "无机物"),
    ("碳酸钠", "[Na+].[Na+].[O-]C(=O)[O-]", "Na2CO3", ("纯碱", "苏打"), "无机物"),
    ("碳酸氢钠", "[Na+].OC(=O)[O-]", "NaHCO3", ("小苏打",), "无机物"),
    ("氯化钠", "[Na+].[Cl-]", "NaCl", ("食盐",), "无机物"),
    ("硫酸铜", "[Cu+2].[O-]S(=O)(=O)[O-]", "CuSO4", ("胆矾无水物",), "无机物"),
    ("胆矾", "[Cu+2].[O-]S(=O)(=O)[O-].O.O.O.O.O", "CuSO4·5H2O",
     ("五水硫酸铜", "蓝矾"), "无机物"),
    ("高锰酸钾", "[K+].[O-][Mn](=O)(=O)=O", "KMnO4", ("灰锰氧",), "无机物"),
)


@dataclass(frozen=True)
class NameEntry:
    """一条名称条目。"""

    name: str
    smiles: str | None = None
    condensed: str | None = None
    """教材通用简式；为空表示由 SMILES 自动推导。"""
    aliases: tuple[str, ...] = ()
    category: str = ""
    formula: str | None = None
    """没有 SMILES 时直接给出分子式（如淀粉 ``(C6H10O5)n``）。"""
    extra: tuple[str, ...] = field(default=())

    @property
    def all_names(self) -> tuple[str, ...]:
        """主名 + 全部别名。"""
        return (self.name, *self.aliases)


NAME_ENTRIES: tuple[NameEntry, ...] = tuple(
    NameEntry(
        name=name,
        smiles=smiles,
        condensed=condensed,
        aliases=aliases,
        category=category,
    )
    for name, smiles, condensed, aliases, category in _RAW_ENTRIES
)

_NAME_INDEX: dict[str, NameEntry] | None = None
_FORMULA_INDEX: dict[str, list[NameEntry]] | None = None
_CATEGORIES: dict[str, list[NameEntry]] | None = None


def _normalize(text: str) -> str:
    """名称归一化：去空白、全角转半角、统一连字符。"""
    result: list[str] = []
    for char in text.strip():
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            continue
        else:
            result.append(char)
    joined = "".join(result).replace(" ", "").replace("\t", "")
    for dash in ("‐", "‑", "‒", "–", "—", "−"):
        joined = joined.replace(dash, "-")
    return joined.upper().replace("Α", "A").replace("Β", "B")


def _entry_formula(entry: NameEntry) -> str:
    """计算条目的分子式。"""
    if entry.smiles:
        from .smiles import SmilesError, parse_smiles

        try:
            molecule = parse_smiles(entry.smiles)
        except SmilesError:
            return entry.formula or ""
        from .model import hill_formula

        return hill_formula(molecule.formula_counts(), molecule.total_charge())
    return entry.formula or ""


def _ensure_index() -> None:
    """惰性构建名称索引、分子式索引与分类索引。"""
    global _NAME_INDEX, _FORMULA_INDEX, _CATEGORIES
    if _NAME_INDEX is not None:
        return

    name_index: dict[str, NameEntry] = {}
    formula_index: dict[str, list[NameEntry]] = {}
    categories: dict[str, list[NameEntry]] = {}

    for entry in NAME_ENTRIES:
        for alias in entry.all_names:
            name_index[_normalize(alias)] = entry
        formula = _entry_formula(entry)
        if formula:
            formula_index.setdefault(_normalize(formula), []).append(entry)
        categories.setdefault(entry.category, []).append(entry)

    for bucket in formula_index.values():
        bucket.sort(key=lambda item: item.name)

    _NAME_INDEX = name_index
    _FORMULA_INDEX = formula_index
    _CATEGORIES = categories


def lookup_by_name(text: str) -> NameEntry | None:
    """按中文名或俗名精确查找。"""
    _ensure_index()
    assert _NAME_INDEX is not None
    return _NAME_INDEX.get(_normalize(text))


def search_names(keyword: str, limit: int = 20) -> list[NameEntry]:
    """按关键字模糊搜索名称，用于"没找到"时的提示。"""
    _ensure_index()
    needle = _normalize(keyword)
    if not needle:
        return []
    hits: list[NameEntry] = []
    seen: set[str] = set()
    for entry in NAME_ENTRIES:
        if entry.name in seen:
            continue
        if any(needle in _normalize(alias) for alias in entry.all_names) or any(
            _normalize(alias) in needle for alias in entry.all_names
        ):
            hits.append(entry)
            seen.add(entry.name)
        if len(hits) >= limit:
            break
    return hits


def lookup_by_formula(formula: str) -> list[NameEntry]:
    """按分子式反查物质，可能返回多个同分异构体。"""
    _ensure_index()
    assert _FORMULA_INDEX is not None
    return list(_FORMULA_INDEX.get(_normalize(formula), ()))


def categories() -> dict[str, list[NameEntry]]:
    """返回"类别 → 条目列表"的映射。"""
    _ensure_index()
    assert _CATEGORIES is not None
    return {key: list(value) for key, value in _CATEGORIES.items()}


def entry_count() -> int:
    """条目总数。"""
    return len(NAME_ENTRIES)
