"""化学结构简式核心库。

本包不依赖 AstrBot，只依赖标准库（结构式绘图是可选的 RDKit 增强），
可以脱离机器人单独运行与测试。
"""

from __future__ import annotations

from .condense import CondenseResult, molecule_to_condensed
from .model import (
    Atom,
    Bond,
    Group,
    IndeterminateFormula,
    Node,
    collect_counts,
    decorate_formula,
    describe_elements,
    element_breakdown,
    has_variable_count,
    hill_formula,
    molar_mass,
    render,
    render_ascii,
    render_html,
    render_unicode,
    total_charge,
)
from .names import (
    NAME_ENTRIES,
    NameEntry,
    categories,
    entry_count,
    lookup_by_formula,
    lookup_by_name,
    search_names,
)
from .resolve import Compound, ResolveError, resolve, suggest_examples
from .smiles import Molecule, SmilesAtom, SmilesBond, SmilesError, parse_smiles
from .text_formula import FormulaSyntaxError, ParsedCondensed, parse_condensed

__all__ = [
    "Atom",
    "Bond",
    "Compound",
    "CondenseResult",
    "FormulaSyntaxError",
    "Group",
    "IndeterminateFormula",
    "Molecule",
    "NAME_ENTRIES",
    "NameEntry",
    "Node",
    "ParsedCondensed",
    "ResolveError",
    "SmilesAtom",
    "SmilesBond",
    "SmilesError",
    "categories",
    "collect_counts",
    "decorate_formula",
    "describe_elements",
    "element_breakdown",
    "entry_count",
    "has_variable_count",
    "hill_formula",
    "lookup_by_formula",
    "lookup_by_name",
    "molar_mass",
    "molecule_to_condensed",
    "parse_condensed",
    "parse_smiles",
    "render",
    "render_ascii",
    "render_html",
    "render_unicode",
    "resolve",
    "search_names",
    "suggest_examples",
    "total_charge",
]
