"""Fiduciary Audit & Verification — Step-by-step tax engine transparency."""

import sys
from pathlib import Path

import streamlit as st

# Allow importing from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-built test scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    # --- Standard profiles ---
    {
        "name": "1. Single 50k — Lausanne",
        "cat": "Standard",
        "desc": "Basic single person, average income",
        "params": dict(gross=50_000, age=35, married=False, children=0,
                       commune="Lausanne", church_tax=False, has_canteen=False,
                       has_2nd_pillar=True, children_in_daycare=0),
    },
    {
        "name": "2. Single 80k — Lausanne",
        "cat": "Standard",
        "desc": "Single, above-average income",
        "params": dict(gross=80_000, age=40, married=False, children=0,
                       commune="Lausanne", church_tax=False, has_canteen=False,
                       has_2nd_pillar=True, children_in_daycare=0),
    },
    {
        "name": "3. Married 100k — Lausanne",
        "cat": "Standard",
        "desc": "Married couple, no children",
        "params": dict(gross=100_000, age=35, married=True, children=0,
                       commune="Lausanne", church_tax=False, has_canteen=False,
                       has_2nd_pillar=True, children_in_daycare=0),
    },
    {
        "name": "4. Family 120k — 2 kids",
        "cat": "Standard",
        "desc": "Married, 2 children, no daycare",
        "params": dict(gross=120_000, age=40, married=True, children=2,
                       commune="Lausanne", church_tax=False, has_canteen=False,
                       has_2nd_pillar=True, children_in_daycare=0),
    },
    {
        "name": "5. Family 120k — 2 kids, 1 daycare",
        "cat": "Standard",
        "desc": "Same as #4 but 1 child in daycare",
        "params": dict(gross=120_000, age=40, married=True, children=2,
                       commune="Lausanne", church_tax=False, has_canteen=False,
                       has_2nd_pillar=True, children_in_daycare=1),
    },
    {
        "name": "6. Single parent 60k — 1 kid",
        "cat": "Standard",
        "desc": "Single parent, 1 child, no daycare",
        "params": dict(gross=60_000, age=30, married=False, children=1,
                       commune="Lausanne", church_tax=False, has_canteen=False,
                       has_2nd_pillar=True, children_in_daycare=0),
    },
    {
        "name": "7. High income 200k",
        "cat": "Standard",
        "desc": "Single, high income, age 50",
        "params": dict(gross=200_000, age=50, married=False, children=0,
                       commune="Lausanne", church_tax=False, has_canteen=False,
                       has_2nd_pillar=True, children_in_daycare=0),
    },
    # --- Edge cases ---
    {
        "name": "8. Low income 35k — code 695 full",
        "cat": "Edge",
        "desc": "Code 695 deduction = income (taxable → 0)",
        "params": dict(gross=35_000, age=25, married=False, children=0,
                       commune="Lausanne", church_tax=False, has_canteen=False,
                       has_2nd_pillar=True, children_in_daycare=0),
    },
    {
        "name": "9. Code 695 boundary — 42k",
        "cat": "Edge",
        "desc": "Taxable income near 17k threshold, partial code 695",
        "params": dict(gross=42_000, age=30, married=False, children=0,
                       commune="Lausanne", church_tax=False, has_canteen=False,
                       has_2nd_pillar=True, children_in_daycare=0),
    },
    {
        "name": "10. Family ded phase-out — 200k married+2kids",
        "cat": "Edge",
        "desc": "Near 126,300 taxable → family deduction phase-out",
        "params": dict(gross=200_000, age=45, married=True, children=2,
                       commune="Lausanne", church_tax=False, has_canteen=False,
                       has_2nd_pillar=True, children_in_daycare=0),
    },
    {
        "name": "11. AC salary cap — 160k",
        "cat": "Edge",
        "desc": "Gross 160k > AC cap 148,200",
        "params": dict(gross=160_000, age=45, married=False, children=0,
                       commune="Lausanne", church_tax=False, has_canteen=False,
                       has_2nd_pillar=True, children_in_daycare=0),
    },
    {
        "name": "12. BVG below threshold — 22k",
        "cat": "Edge",
        "desc": "Gross 22k < BVG entry threshold 22,680",
        "params": dict(gross=22_000, age=30, married=False, children=0,
                       commune="Lausanne", church_tax=False, has_canteen=False,
                       has_2nd_pillar=True, children_in_daycare=0),
    },
    {
        "name": "13. BVG max coord — 150k",
        "cat": "Edge",
        "desc": "Coordinated salary capped at max 64,260",
        "params": dict(gross=150_000, age=55, married=False, children=0,
                       commune="Lausanne", church_tax=False, has_canteen=False,
                       has_2nd_pillar=True, children_in_daycare=0),
    },
]


# ---------------------------------------------------------------------------
# Audit step builders — recompute with formula strings
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    """Format a number as CHF with thousands separator."""
    return f"{v:,.2f}"


def _pct(v: float) -> str:
    return f"{v}%"


def audit_social(gross: float) -> list[tuple]:
    """Audit trail for social contributions."""
    avs = gross * 5.30 / 100
    ac_base = min(gross, 148_200)
    ac = ac_base * 1.10 / 100
    aanp = gross * 1.06 / 100
    total = avs + ac + aanp

    steps = [
        ("AVS/AI/APG",
         f"{_fmt(gross)} x 5.30%",
         avs, "LAVS Art. 5 / LAI Art. 3 / LAPG Art. 27"),
        ("AC (unemployment)",
         f"min({_fmt(gross)}, 148,200) = {_fmt(ac_base)} x 1.10%",
         ac, "LACI Art. 3"),
        ("AANP (accident)",
         f"{_fmt(gross)} x 1.06%",
         aanp, "LAA Art. 91 (avg. rate)"),
        ("**Total social**",
         f"{_fmt(avs)} + {_fmt(ac)} + {_fmt(aanp)}",
         total, ""),
    ]
    return steps


def audit_bvg(gross: float, age: int) -> list[tuple]:
    """Audit trail for BVG / LPP."""
    entry = 22_680
    coord_deduction = 26_460
    min_coord = 3_780
    max_coord = 64_260

    steps = []
    if gross < entry:
        steps.append(("Entry check",
                       f"{_fmt(gross)} < {_fmt(entry)}",
                       0, "LPP Art. 2"))
        steps.append(("**BVG employee share**", "Not subject to BVG", 0, ""))
        return steps

    steps.append(("Entry check",
                   f"{_fmt(gross)} >= {_fmt(entry)} -> subject",
                   gross, "LPP Art. 2"))

    raw_coord = gross - coord_deduction
    steps.append(("Coordinated salary (raw)",
                   f"{_fmt(gross)} - {_fmt(coord_deduction)}",
                   raw_coord, "LPP Art. 8"))

    coord = max(min_coord, min(max_coord, raw_coord))
    if raw_coord < min_coord:
        steps.append(("Apply minimum",
                       f"max({_fmt(raw_coord)}, {_fmt(min_coord)})",
                       coord, "LPP Art. 8"))
    elif raw_coord > max_coord:
        steps.append(("Apply cap",
                       f"min({_fmt(raw_coord)}, {_fmt(max_coord)})",
                       coord, "LPP Art. 8"))
    else:
        steps.append(("Coordinated salary (final)",
                       f"{_fmt(coord)} (within [{_fmt(min_coord)}, {_fmt(max_coord)}])",
                       coord, "LPP Art. 8"))

    if age < 25:
        rate, bracket = 0, "<25"
    elif age < 35:
        rate, bracket = 7.0, "25-34"
    elif age < 45:
        rate, bracket = 10.0, "35-44"
    elif age < 55:
        rate, bracket = 15.0, "45-54"
    else:
        rate, bracket = 18.0, "55-65"

    steps.append(("BVG rate",
                   f"Age {age} -> bracket {bracket} -> {rate}%",
                   rate, "LPP Art. 16"))

    total_contrib = coord * rate / 100
    steps.append(("Total BVG contribution",
                   f"{_fmt(coord)} x {rate}%",
                   total_contrib, ""))

    employee = total_contrib / 2
    steps.append(("**BVG employee share (50%)**",
                   f"{_fmt(total_contrib)} / 2",
                   employee, "LPP Art. 66"))
    return steps


def audit_cantonal_deductions(
    gross: float, net_after_social: float, married: bool, children: int,
    has_2nd_pillar: bool, has_canteen: bool, children_in_daycare: int,
) -> list[tuple]:
    """Audit trail for cantonal deductions."""
    steps = []
    single_parent = not married and children > 0

    # Professional deduction
    raw_prof = net_after_social * 0.03
    prof = max(2_000, min(4_000, raw_prof))
    steps.append(("Professional expenses (code 160)",
                   f"max(2,000, min(4,000, {_fmt(net_after_social)} x 3%)) = max(2,000, min(4,000, {_fmt(raw_prof)}))",
                   prof, "Art. 37, al. 1, let. a LI"))

    # Meal
    meal = 1_600 if has_canteen else 3_200
    canteen_str = "with canteen" if has_canteen else "without canteen"
    steps.append(("Meal costs (code 150)",
                   f"{'1,600' if has_canteen else '3,200'} ({canteen_str})",
                   meal, "Art. 37, al. 1, let. b LI"))

    # Insurance
    if married:
        ins_base, ins_desc = 9_900, "9,900 (married)"
    else:
        ins_base, ins_desc = 5_000, "5,000 (single)"
    ins_child = 1_300 * children
    insurance = ins_base + ins_child
    if children > 0:
        steps.append(("Insurance (code 300)",
                       f"{ins_desc} + {children} x 1,300",
                       insurance, "Art. 37, al. 1, let. g LI"))
    else:
        steps.append(("Insurance (code 300)",
                       ins_desc,
                       insurance, "Art. 37, al. 1, let. g LI"))

    # Pillar 3a
    if has_2nd_pillar:
        p3a = 7_258
        p3a_formula = "7,258 (with 2nd pillar)"
    else:
        p3a = min(gross * 0.20, 36_288)
        p3a_formula = f"min({_fmt(gross)} x 20%, 36,288) = {_fmt(p3a)}"
    steps.append(("Pillar 3a (code 310)",
                   p3a_formula,
                   p3a, "Art. 37, al. 1, let. f LI / LPP Art. 82"))

    # Savings interest
    if married:
        sav_base, sav_desc = 3_300, "3,300 (married)"
    else:
        sav_base, sav_desc = 1_600, "1,600 (single)"
    sav_child = 300 * children
    savings = sav_base + sav_child
    if children > 0:
        steps.append(("Savings interest (code 480)",
                       f"{sav_desc} + {children} x 300",
                       savings, "Art. 37, al. 1, let. g LI"))
    else:
        steps.append(("Savings interest (code 480)",
                       sav_desc,
                       savings, "Art. 37, al. 1, let. g LI"))

    # Childcare
    childcare = children_in_daycare * 15_200
    if children_in_daycare > 0:
        steps.append(("Childcare (code 670)",
                       f"{children_in_daycare} x 15,200",
                       childcare, "Art. 37, al. 1, let. k LI"))

    total_ded = prof + meal + insurance + p3a + savings + childcare
    steps.append(("**Subtotal professional deductions**",
                   f"{_fmt(prof)} + {_fmt(meal)} + {_fmt(insurance)} + {_fmt(p3a)} + {_fmt(savings)}"
                   + (f" + {_fmt(childcare)}" if childcare > 0 else ""),
                   total_ded, ""))

    cantonal_pre = max(0, net_after_social - total_ded)
    steps.append(("Cantonal taxable (pre-social ded.)",
                   f"max(0, {_fmt(net_after_social)} - {_fmt(total_ded)})",
                   cantonal_pre, ""))

    # Code 695 — contribuable modeste
    base_695 = 17_000
    if married:
        base_695 += 5_700
    elif single_parent:
        base_695 += 3_200
    base_695 += 3_500 * children

    modeste = app._calc_contribuable_modeste(cantonal_pre, married, children)

    if cantonal_pre <= 0:
        steps.append(("Code 695 (contribuable modeste)",
                       f"Income = 0 -> no deduction",
                       0, "Art. 42 LI"))
    elif cantonal_pre <= base_695:
        steps.append(("Code 695 (contribuable modeste)",
                       f"Income {_fmt(cantonal_pre)} <= base {_fmt(base_695)} -> deduction = income",
                       modeste, "Art. 42 LI"))
    else:
        excess = cantonal_pre - base_695
        reduction = int(excess / 200 + 0.5) * 100
        steps.append(("Code 695 (contribuable modeste)",
                       f"Base: {_fmt(base_695)}"
                       + (" (17,000 + 5,700 married)" if married else
                          f" (17,000 + 3,200 single parent + {children} x 3,500)" if single_parent else
                          " (17,000 single)")
                       + f" | Excess: {_fmt(cantonal_pre)} - {_fmt(base_695)} = {_fmt(excess)}"
                       + f" | 50% = {_fmt(excess/2)} -> round100 = {_fmt(reduction)}"
                       + f" | Ded: {_fmt(base_695)} - {_fmt(reduction)}",
                       modeste, "Art. 42 LI"))

    # Code 810 — family deduction
    family = app._calc_family_deduction(cantonal_pre, married, children)
    if family > 0:
        if married:
            fam_base_str = "1,300 (married)"
        elif single_parent:
            fam_base_str = "2,800 (single parent)"
        else:
            fam_base_str = "0"
        if children > 0:
            fam_base_str += f" + {children} x 1,000"
        fam_base = (1_300 if married else (2_800 if single_parent else 0)) + 1_000 * children

        if cantonal_pre <= 126_300:
            steps.append(("Code 810 (family deduction)",
                           f"Base: {fam_base_str} = {_fmt(fam_base)} | Income {_fmt(cantonal_pre)} <= 126,300 -> full",
                           family, "Art. 42a LI"))
        else:
            steps.append(("Code 810 (family deduction)",
                           f"Base: {fam_base_str} = {_fmt(fam_base)} | Income {_fmt(cantonal_pre)} > 126,300 -> phase-out",
                           family, "Art. 42a LI"))

    cantonal_final = max(0, cantonal_pre - modeste - family)
    steps.append(("**Cantonal taxable income**",
                   f"max(0, {_fmt(cantonal_pre)} - {_fmt(modeste)} - {_fmt(family)})",
                   cantonal_final, ""))
    return steps


def audit_federal_deductions(
    net_after_social: float, gross: float, married: bool, children: int,
    has_2nd_pillar: bool, has_canteen: bool, children_in_daycare: int,
) -> list[tuple]:
    """Audit trail for federal deductions."""
    steps = []

    # Professional deduction (same as cantonal)
    raw_prof = net_after_social * 0.03
    prof = max(2_000, min(4_000, raw_prof))
    steps.append(("Professional expenses",
                   f"max(2,000, min(4,000, {_fmt(net_after_social)} x 3%))",
                   prof, "LIFD Art. 26"))

    # Meal
    meal = 1_600 if has_canteen else 3_200
    steps.append(("Meal costs",
                   f"{'1,600' if has_canteen else '3,200'}",
                   meal, "LIFD Art. 26"))

    # Federal insurance
    if married:
        fi_base = 3_700 if has_2nd_pillar else 5_550
        fi_desc = f"{'3,700' if has_2nd_pillar else '5,550'} (married, {'with' if has_2nd_pillar else 'without'} 2nd pillar)"
    else:
        fi_base = 1_800 if has_2nd_pillar else 2_700
        fi_desc = f"{'1,800' if has_2nd_pillar else '2,700'} (single, {'with' if has_2nd_pillar else 'without'} 2nd pillar)"
    fi_child = 700 * children
    fed_insurance = fi_base + fi_child
    if children > 0:
        steps.append(("Insurance deduction",
                       f"{fi_desc} + {children} x 700",
                       fed_insurance, "LIFD Art. 33, al. 1, let. d/g"))
    else:
        steps.append(("Insurance deduction",
                       fi_desc,
                       fed_insurance, "LIFD Art. 33, al. 1, let. d/g"))

    # Pillar 3a
    if has_2nd_pillar:
        p3a = 7_258
        p3a_formula = "7,258 (with 2nd pillar)"
    else:
        p3a = min(gross * 0.20, 36_288)
        p3a_formula = f"min({_fmt(gross)} x 20%, 36,288)"
    steps.append(("Pillar 3a", p3a_formula, p3a, "LIFD Art. 33, al. 1, let. e"))

    # Childcare
    fed_childcare = children_in_daycare * 25_500
    if children_in_daycare > 0:
        steps.append(("Childcare",
                       f"{children_in_daycare} x 25,500",
                       fed_childcare, "LIFD Art. 33, al. 3"))

    # Child deduction
    fed_child = 6_800 * children
    if children > 0:
        steps.append(("Child deduction",
                       f"{children} x 6,800",
                       fed_child, "LIFD Art. 35, al. 1, let. a"))

    # Married deduction
    fed_married = 2_800 if married else 0
    if married:
        steps.append(("Married deduction", "2,800", fed_married, "LIFD Art. 35, al. 1, let. c"))

    total = prof + meal + fed_insurance + p3a + fed_childcare + fed_child + fed_married
    steps.append(("**Total federal deductions**",
                   " + ".join([_fmt(x) for x in [prof, meal, fed_insurance, p3a]
                               + ([fed_childcare] if fed_childcare else [])
                               + ([fed_child] if fed_child else [])
                               + ([fed_married] if fed_married else [])]),
                   total, ""))

    federal_taxable = max(0, net_after_social - total)
    steps.append(("**Federal taxable income**",
                   f"max(0, {_fmt(net_after_social)} - {_fmt(total)})",
                   federal_taxable, ""))
    return steps


def audit_cantonal_tax(
    cantonal_taxable: float, married: bool, children: int,
    commune_coeff: float, church_tax: bool,
) -> list[tuple]:
    """Audit trail for ICC calculation."""
    steps = []
    cantonal_coeff = 155.0
    lripp = 5.0

    # Quotient familial
    single_parent = not married and children > 0
    if married:
        parts_base, parts_desc = 1.8, "1.8 (married)"
    elif single_parent:
        parts_base, parts_desc = 1.3, "1.3 (single parent)"
    else:
        parts_base, parts_desc = 1.0, "1.0 (single)"
    parts = parts_base + 0.5 * children
    if children > 0:
        steps.append(("Quotient familial",
                       f"{parts_desc} + {children} x 0.5",
                       parts, "Art. 43 LI"))
    else:
        steps.append(("Quotient familial",
                       parts_desc,
                       parts, "Art. 43 LI"))

    # Income per part
    income_pp = cantonal_taxable / parts if parts > 0 else cantonal_taxable
    steps.append(("Income per part",
                   f"{_fmt(cantonal_taxable)} / {parts}",
                   income_pp, "Art. 43 LI"))

    # Bareme lookup
    tax_pp = app._bareme_base_tax(income_pp)
    lower = int(income_pp // 100) * 100
    upper = lower + 100
    steps.append(("Bareme lookup",
                   f"bareme({_fmt(income_pp)}) -> interpolate [{_fmt(lower)}, {_fmt(upper)}]",
                   tax_pp, "Art. 47 LI / Bareme revenu 2025"))

    # Base tax
    base_tax = tax_pp * parts
    steps.append(("Base tax (x parts)",
                   f"{_fmt(tax_pp)} x {parts}",
                   base_tax, ""))

    # Cantonal part
    cant_raw = base_tax * cantonal_coeff / 100
    cant_after_lripp = cant_raw * (1 - lripp / 100)
    steps.append(("Cantonal coefficient",
                   f"{_fmt(base_tax)} x {cantonal_coeff}% = {_fmt(cant_raw)}",
                   cant_raw, "LICom"))
    steps.append(("LRIPP reduction (-5%)",
                   f"{_fmt(cant_raw)} x (1 - 5%) = {_fmt(cant_raw)} x 0.95",
                   cant_after_lripp, "Art. 4 LRIPP (2026)"))

    # Communal part
    comm = base_tax * commune_coeff / 100
    steps.append(("Communal part",
                   f"{_fmt(base_tax)} x {commune_coeff}%",
                   comm, "LICom / Arrete d'imposition"))

    # Church
    church_add = commune_coeff * 0.10 if church_tax else 0
    church_part = base_tax * church_add / 100
    if church_tax:
        steps.append(("Church tax",
                       f"{_fmt(base_tax)} x {church_add:.1f}% ({commune_coeff}% x 10%)",
                       church_part, "LICom"))

    icc = cant_after_lripp + comm + church_part
    steps.append(("**ICC total**",
                   f"{_fmt(cant_after_lripp)} + {_fmt(comm)}"
                   + (f" + {_fmt(church_part)}" if church_tax else ""),
                   icc, ""))
    return steps


def audit_federal_tax(
    federal_taxable: float, married: bool, children: int,
) -> list[tuple]:
    """Audit trail for federal tax."""
    steps = []

    data = app.load_json("federal_tax_brackets_2026.json")
    brackets = data["married_couple"]["brackets"] if married else data["single_person"]["brackets"]
    tariff = "married" if married else "single"

    raw_tax = app._apply_brackets(federal_taxable, brackets)
    steps.append(("Bracket calculation",
                   f"apply_brackets({_fmt(federal_taxable)}, tariff={tariff})",
                   raw_tax, "LIFD Art. 36"))

    child_red = 263 * children
    if children > 0:
        steps.append(("Child reduction",
                       f"{children} x 263",
                       child_red, "LIFD Art. 36, al. 2bis"))

    after_red = max(0, raw_tax - child_red)
    if children > 0:
        steps.append(("After reduction",
                       f"max(0, {_fmt(raw_tax)} - {_fmt(child_red)})",
                       after_red, ""))

    if after_red > 0 and after_red < 25:
        steps.append(("Minimum threshold",
                       f"{_fmt(after_red)} < 25 -> 0",
                       0, "LIFD Art. 36"))
        final = 0
    else:
        final = after_red

    steps.append(("**Federal tax (IFD)**",
                   _fmt(final),
                   final, ""))
    return steps


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_steps(steps: list[tuple], title: str, expanded: bool = True):
    """Render audit steps as a markdown table inside an expander."""
    with st.expander(title, expanded=expanded):
        header = "| # | Step | Formula | Result (CHF) | Legal ref |\n"
        header += "|--:|------|---------|-------------:|----------|\n"
        rows = ""
        for i, (step, formula, result, ref) in enumerate(steps, 1):
            if isinstance(result, float):
                res_str = _fmt(result)
            else:
                res_str = str(result)
            # Escape pipes in formula
            formula_safe = formula.replace("|", "\\|")
            rows += f"| {i} | {step} | {formula_safe} | {res_str} | {ref} |\n"
        st.markdown(header + rows)


def render_comparison(results: dict, key_prefix: str = ""):
    """Render comparison table with expected value inputs."""
    st.subheader("Fiduciary Comparison")
    st.caption("Enter your expected values. Green = match (+-1 CHF), Orange = close (+-10), Red = mismatch.")

    comparison_keys = [
        ("Total social contributions", "social_total"),
        ("BVG employee share", "bvg_employee"),
        ("Cantonal taxable income", "cant_taxable"),
        ("Federal taxable income", "fed_taxable"),
        ("Code 695 deduction", "code_695"),
        ("Code 810 family deduction", "code_810"),
        ("ICC (cantonal + communal)", "icc"),
        ("Federal tax (IFD)", "ifd"),
        ("Net annual salary", "net_annual"),
    ]

    cols = st.columns([3, 2, 2, 2, 1])
    cols[0].markdown("**Result**")
    cols[1].markdown("**Engine**")
    cols[2].markdown("**Expected**")
    cols[3].markdown("**Diff**")
    cols[4].markdown("**OK?**")

    for label, key in comparison_keys:
        engine_val = results.get(key, 0)
        cols = st.columns([3, 2, 2, 2, 1])
        cols[0].write(label)
        cols[1].write(f"**{_fmt(engine_val)}**")
        expected = cols[2].number_input(
            f"exp_{key}",
            value=0.0,
            step=1.0,
            format="%.2f",
            label_visibility="collapsed",
            key=f"{key_prefix}exp_{key}",
        )
        if expected != 0:
            diff = engine_val - expected
            cols[3].write(f"{diff:+,.2f}")
            if abs(diff) <= 1:
                cols[4].markdown(":green[OK]")
            elif abs(diff) <= 10:
                cols[4].markdown(":orange[~]")
            else:
                cols[4].markdown(":red[DIFF]")
        else:
            cols[3].write("—")
            cols[4].write("—")


def run_full_calculation(p: dict) -> dict:
    """Run the full calculation pipeline and return key results."""
    communes = app.load_communes()
    commune_coeff = communes.get(p["commune"], 79.0)

    social = app.calc_social_contributions(p["gross"])
    social_total = social["Total social contributions"]

    bvg = app.calc_bvg(p["gross"], p["age"])
    bvg_employee = bvg["BVG employee share"]

    taxable = app.calc_taxable_income(
        p["gross"], social_total, bvg_employee,
        p["married"], p["children"], p["has_2nd_pillar"],
        p["has_canteen"], p["children_in_daycare"],
    )

    federal = app.calc_federal_tax(taxable["Federal taxable income"], p["married"], p["children"])
    cantonal = app.calc_cantonal_tax(
        taxable["Cantonal taxable income"], p["married"], p["children"],
        commune_coeff, p["church_tax"],
    )

    alloc = app.calc_allocations_familiales(p["children"])
    icc = cantonal["Cantonal + communal tax (ICC)"]
    ifd = federal["Federal tax (IFD)"]
    total_ded = social_total + bvg_employee + ifd + icc
    net = p["gross"] - total_ded + alloc["Total annual"]

    return {
        "social_total": social_total,
        "bvg_employee": bvg_employee,
        "cant_taxable": taxable["Cantonal taxable income"],
        "fed_taxable": taxable["Federal taxable income"],
        "code_695": taxable["Contribuable modeste (cantonal)"],
        "code_810": taxable["Family deduction (cantonal)"],
        "icc": icc,
        "ifd": ifd,
        "net_annual": net,
        "alloc_annual": alloc["Total annual"],
        "commune_coeff": commune_coeff,
        # Pass-through for audit steps
        "net_after_social": taxable["Net after social"],
    }


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Audit — Swiss Tax Calculator", layout="wide")
    st.title("Fiduciary Audit & Verification")
    st.caption("Step-by-step calculation transparency for Canton de Vaud tax engine")

    # ---- Scenario selector ----
    scenario_names = ["Custom"] + [s["name"] for s in SCENARIOS]
    chosen = st.selectbox("Scenario", scenario_names)

    if chosen == "Custom":
        communes = app.load_communes()
        commune_names = sorted(communes.keys())
        default_idx = commune_names.index("Lausanne") if "Lausanne" in commune_names else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gross = st.number_input("Gross salary (CHF)", 10_000, 500_000, 100_000, 1_000)
            age = st.slider("Age", 20, 65, 35)
        with c2:
            marital = st.selectbox("Marital status", ["Single", "Married"])
            married = marital == "Married"
            children = st.slider("Children", 0, 5, 0)
        with c3:
            commune = st.selectbox("Commune", commune_names, index=default_idx)
            church_tax = st.checkbox("Church tax")
        with c4:
            has_canteen = st.checkbox("Canteen")
            has_2nd_pillar = st.checkbox("2nd pillar (BVG)", True)
            children_in_daycare = st.slider("Kids in daycare", 0, max(children, 1), 0) if children > 0 else 0

        p = dict(gross=gross, age=age, married=married, children=children,
                 commune=commune, church_tax=church_tax, has_canteen=has_canteen,
                 has_2nd_pillar=has_2nd_pillar, children_in_daycare=children_in_daycare)
    else:
        scenario = next(s for s in SCENARIOS if s["name"] == chosen)
        p = scenario["params"]
        st.info(f"**{scenario['cat']}** — {scenario['desc']}")
        # Show params
        cols = st.columns(4)
        cols[0].metric("Gross", f"CHF {p['gross']:,}")
        cols[1].metric("Status", "Married" if p["married"] else "Single")
        cols[2].metric("Children", p["children"])
        cols[3].metric("Age", p["age"])

    # ---- Run calculations ----
    results = run_full_calculation(p)
    communes = app.load_communes()
    commune_coeff = communes.get(p.get("commune", "Lausanne"), 79.0)

    # ---- Summary metrics ----
    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Cant. taxable", f"CHF {results['cant_taxable']:,.0f}")
    m2.metric("Fed. taxable", f"CHF {results['fed_taxable']:,.0f}")
    m3.metric("ICC", f"CHF {results['icc']:,.0f}")
    m4.metric("IFD", f"CHF {results['ifd']:,.0f}")
    m5.metric("Net annual", f"CHF {results['net_annual']:,.0f}")

    # ---- Step-by-step audit ----
    st.divider()
    st.subheader("Step-by-step calculation")

    # 1. Social
    render_steps(audit_social(p["gross"]),
                 "1. Social contributions — LAVS, LAI, LAPG, LACI, LAA")

    # 2. BVG
    render_steps(audit_bvg(p["gross"], p["age"]),
                 "2. BVG / LPP — Occupational pension (2nd pillar)")

    # 3. Cantonal deductions
    render_steps(
        audit_cantonal_deductions(
            p["gross"], results["net_after_social"], p["married"], p["children"],
            p["has_2nd_pillar"], p["has_canteen"], p["children_in_daycare"],
        ),
        "3. Cantonal deductions & taxable income — LI (Loi sur les impots directs cantonaux)",
    )

    # 4. Federal deductions
    render_steps(
        audit_federal_deductions(
            results["net_after_social"], p["gross"], p["married"], p["children"],
            p["has_2nd_pillar"], p["has_canteen"], p["children_in_daycare"],
        ),
        "4. Federal deductions & taxable income — LIFD",
    )

    # 5. Cantonal + communal tax
    render_steps(
        audit_cantonal_tax(
            results["cant_taxable"], p["married"], p["children"],
            commune_coeff, p["church_tax"],
        ),
        f"5. Cantonal + communal tax (ICC) — Commune: {p.get('commune', 'Lausanne')} ({commune_coeff}%)",
    )

    # 6. Federal tax
    render_steps(
        audit_federal_tax(results["fed_taxable"], p["married"], p["children"]),
        "6. Federal tax (IFD) — LIFD Art. 36",
    )

    # ---- Comparison ----
    st.divider()
    render_comparison(results, key_prefix=f"sc_{chosen[:10]}_")

    # ---- Batch test runner ----
    st.divider()
    st.subheader("Batch test runner")

    if st.button("Run all 13 scenarios", type="primary"):
        rows = []
        for s in SCENARIOS:
            r = run_full_calculation(s["params"])
            rows.append({
                "Scenario": s["name"],
                "Gross": f"{s['params']['gross']:,}",
                "Status": s["cat"],
                "Cant. taxable": f"{r['cant_taxable']:,.0f}",
                "Fed. taxable": f"{r['fed_taxable']:,.0f}",
                "Code 695": f"{r['code_695']:,.0f}",
                "Code 810": f"{r['code_810']:,.0f}",
                "ICC": f"{r['icc']:,.0f}",
                "IFD": f"{r['ifd']:,.0f}",
                "Net annual": f"{r['net_annual']:,.0f}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.success(f"All {len(SCENARIOS)} scenarios computed successfully.")


if __name__ == "__main__":
    main()
