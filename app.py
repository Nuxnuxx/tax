"""Swiss Tax Calculator - Vaud Canton MVP"""

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_json(filename: str) -> dict:
    with open(DATA_DIR / filename) as f:
        return json.load(f)


@st.cache_data
def load_communes() -> dict[str, float]:
    """Return {commune_name: coefficient_total_pct}."""
    data = load_json("vaud_commune_coefficients_2026.json")
    return {
        c["commune"]: float(c["coefficient_total_pct"])
        for c in data["communes"]
    }


# ---------------------------------------------------------------------------
# Social contributions
# ---------------------------------------------------------------------------

def calc_social_contributions(gross: float) -> dict:
    """Employee-side social contributions."""
    avs_ai_apg = gross * 5.30 / 100
    ac_base = min(gross, 148_200)
    ac = ac_base * 1.10 / 100
    aanp = gross * 1.06 / 100
    total = avs_ai_apg + ac + aanp
    return {
        "AVS/AI/APG (5.3%)": round(avs_ai_apg, 2),
        "AC (1.1%)": round(ac, 2),
        "AANP (1.06%)": round(aanp, 2),
        "Total social contributions": round(total, 2),
    }


# ---------------------------------------------------------------------------
# BVG / LPP (2nd pillar)
# ---------------------------------------------------------------------------

def calc_bvg(gross: float, age: int) -> dict:
    """Employee share of BVG contribution."""
    if gross < 22_680:
        return {"Coordinated salary": 0, "BVG rate": 0, "BVG employee share": 0}

    coord_salary = gross - 26_460
    coord_salary = max(coord_salary, 3_780)
    coord_salary = min(coord_salary, 64_260)

    if age < 25:
        rate = 0.0
    elif age < 35:
        rate = 7.0
    elif age < 45:
        rate = 10.0
    elif age < 55:
        rate = 15.0
    else:
        rate = 18.0

    total_contribution = coord_salary * rate / 100
    employee_share = total_contribution / 2  # 50/50 split

    return {
        "Coordinated salary": round(coord_salary, 2),
        "BVG rate": rate,
        "BVG employee share": round(employee_share, 2),
    }


# ---------------------------------------------------------------------------
# Deductions & taxable income
# ---------------------------------------------------------------------------

def _calc_family_deduction(cantonal_taxable: float, married: bool, children: int) -> float:
    """Cantonal family deduction (code 810) with phase-out above CHF 126,300."""
    if not married and children == 0:
        return 0.0

    # Base amounts (Art. 42a LI)
    if married:
        base = 1_300
    elif children > 0:
        base = 2_800  # single parent (Art. 43, al. 2, let. c)
    else:
        base = 0
    base += 1_000 * children

    if base == 0:
        return 0.0

    threshold = 126_300
    if cantonal_taxable <= threshold:
        return base

    # Phase-out: reduce by CHF 100 per income step
    excess = cantonal_taxable - threshold
    if excess <= 37_800:  # 126,300 to 164,100
        reduction = (excess // 2_100) * 100
    else:
        reduction = (37_800 // 2_100) * 100  # = 1,800
        excess2 = excess - 37_800
        reduction += (excess2 // 1_000) * 100

    return max(0, base - reduction)


def _calc_contribuable_modeste(
    revenu_net: float, married: bool, children: int,
) -> float:
    """Cantonal deduction for modest taxpayers (code 695, Art. 42 LI).

    Applied on net income (code 690) after professional deductions.
    Base: 17,000 + 5,700 (married) or 3,200 (single parent) + 3,500/child.
    Phase-out: reduction = 50% of excess above base, rounded to nearest 100.
    Complete phase-out at income = 3 × base.
    Source: Instructions générales 2024, pages 39-41.
    """
    if revenu_net <= 0:
        return 0

    single_parent = not married and children > 0

    # Maximum deduction amount (also serves as phase-out threshold)
    base = 17_000
    if married:
        base += 5_700
    elif single_parent:
        base += 3_200
    base += 3_500 * children

    # Cannot deduct more than income
    if revenu_net <= base:
        return revenu_net

    # Phase-out: reduction = 50% of excess, rounded to nearest CHF 100
    # (commercial rounding: 50+ rounds up)
    excess = revenu_net - base
    reduction = int(excess / 200 + 0.5) * 100

    return max(0, base - reduction)


def calc_taxable_income(
    gross: float,
    social_total: float,
    bvg_employee: float,
    married: bool,
    children: int,
    has_2nd_pillar: bool,
    has_canteen: bool = False,
    children_in_daycare: int = 0,
) -> dict:
    """Compute cantonal and federal taxable income with detailed breakdown."""
    # Net income after social + BVG
    net_after_social = gross - social_total - bvg_employee

    # Professional deduction (code 160): 3% of net, min 2k, max 4k
    prof_deduction = max(2_000, min(4_000, net_after_social * 0.03))

    # Meal costs deduction (code 150)
    meal_deduction = 1_600 if has_canteen else 3_200

    # Insurance deduction (code 300)
    if married:
        insurance_ded = 9_900 + 1_300 * children
    else:
        insurance_ded = 5_000 + 1_300 * children

    # Pillar 3a
    pillar_3a = 7_258 if has_2nd_pillar else min(gross * 0.20, 36_288)

    # Savings interest deduction (code 480)
    savings_ded = 3_300 if married else 1_600
    savings_ded += 300 * children

    # Childcare deduction (code 670, Art. 37 let. k): max CHF 15,200/child cantonal
    childcare_ded_cantonal = children_in_daycare * 15_200

    total_deductions = (
        prof_deduction + meal_deduction + insurance_ded
        + pillar_3a + savings_ded + childcare_ded_cantonal
    )

    # -- Cantonal taxable income (before social deductions) --
    cantonal_taxable_pre = max(0, net_after_social - total_deductions)

    # Contribuable modeste (code 695, Art. 42 LI) — income-dependent
    modeste_ded = _calc_contribuable_modeste(cantonal_taxable_pre, married, children)

    # Family deduction (code 810, Art. 42a LI) — applied last, with phase-out
    family_ded = _calc_family_deduction(cantonal_taxable_pre, married, children)

    cantonal_taxable = max(0, cantonal_taxable_pre - modeste_ded - family_ded)

    # -- Federal taxable income (different deductions) --
    # Federal insurance premium deduction
    if married:
        fed_insurance = 3_700 if has_2nd_pillar else 5_550
    else:
        fed_insurance = 1_800 if has_2nd_pillar else 2_700
    fed_insurance += 700 * children

    # Federal childcare deduction: max CHF 25,500/child
    fed_childcare_ded = children_in_daycare * 25_500

    # Federal social deductions
    fed_child_ded = 6_800 * children
    fed_married_ded = 2_800 if married else 0

    fed_total_deductions = (
        prof_deduction + meal_deduction + fed_insurance + pillar_3a
        + fed_childcare_ded + fed_child_ded + fed_married_ded
    )
    federal_taxable = max(0, net_after_social - fed_total_deductions)

    return {
        "Net after social": round(net_after_social, 2),
        "Professional deduction (3%)": round(prof_deduction, 2),
        "Meal costs deduction": round(meal_deduction, 2),
        "Insurance deduction": round(insurance_ded, 2),
        "Pillar 3a": round(pillar_3a, 2),
        "Savings interest deduction": round(savings_ded, 2),
        "Childcare deduction (cantonal)": round(childcare_ded_cantonal, 2),
        "Contribuable modeste (cantonal)": round(modeste_ded, 2),
        "Family deduction (cantonal)": round(family_ded, 2),
        "Total deductions (cantonal)": round(total_deductions + modeste_ded + family_ded, 2),
        "Cantonal taxable income": round(cantonal_taxable, 2),
        "Federal insurance deduction": round(fed_insurance, 2),
        "Federal childcare deduction": round(fed_childcare_ded, 2),
        "Federal child deduction": round(fed_child_ded, 2),
        "Federal married deduction": round(fed_married_ded, 2),
        "Total deductions (federal)": round(fed_total_deductions, 2),
        "Federal taxable income": round(federal_taxable, 2),
    }


# ---------------------------------------------------------------------------
# Federal tax (IFD)
# ---------------------------------------------------------------------------

def calc_federal_tax(taxable_income: float, married: bool, children: int) -> dict:
    data = load_json("federal_tax_brackets_2026.json")
    brackets = data["married_couple"]["brackets"] if married else data["single_person"]["brackets"]

    tax = _apply_brackets(taxable_income, brackets)

    # Child reduction
    child_reduction = 263 * children
    tax = max(0, tax - child_reduction)

    # Minimum threshold
    if tax < 25:
        tax = 0

    return {
        "Federal tax (before reduction)": round(tax + child_reduction, 2),
        "Child reduction": round(child_reduction, 2),
        "Federal tax (IFD)": round(tax, 2),
    }


def _apply_brackets(income: float, brackets: list[dict]) -> float:
    """Walk through federal bracket list and compute tax.

    Bracket format (from LIFD Art. 36):
      bracket[i].base_tax  = cumulative tax at income == bracket[i].up_to
      bracket[i].rate_per_100 = marginal rate for income ABOVE bracket[i].up_to
    """
    if income <= 0:
        return 0.0

    for i, b in enumerate(brackets):
        if "from" in b:
            from_limit = b["from"]
            if income >= from_limit:
                rate = b.get("rate_per_100") or 0
                return b["base_tax"] + (income - from_limit) * rate / 100
            # Income in the gap between prev bracket and this one
            prev = brackets[i - 1]
            prev_rate = prev.get("rate_per_100") or 0
            return prev["base_tax"] + (income - prev["up_to"]) * prev_rate / 100

        limit = b["up_to"]

        if income <= limit:
            if i == 0:
                # Below first threshold: tax is proportional (rate * income)
                # but typically first bracket is the 0-tax zone
                return b["base_tax"]
            # Income is between bracket[i-1].up_to and bracket[i].up_to
            prev = brackets[i - 1]
            rate = prev.get("rate_per_100") or 0
            return prev["base_tax"] + (income - prev["up_to"]) * rate / 100

    # Above all brackets (shouldn't happen with proper data)
    last = brackets[-1]
    return last["base_tax"]


# ---------------------------------------------------------------------------
# Cantonal + communal tax (ICC)
# ---------------------------------------------------------------------------

@st.cache_data
def _load_bareme() -> dict[int, float]:
    """Load the complete Vaud cantonal barème (3000 entries, every CHF 100)."""
    data = load_json("vaud_cantonal_tax_bareme_2025.json")
    return {entry["income"]: entry["base_tax"] for entry in data["bareme"]}


BAREME_MAX_INCOME = 300_000
BAREME_MAX_TAX = 36_343.50
BAREME_RATE_PER_100_ABOVE_MAX = 15.50  # CHF per CHF 100 above max


def _bareme_base_tax(income: float) -> float:
    """Exact lookup + linear interpolation on the full Vaud barème."""
    if income <= 0:
        return 0.0
    if income >= BAREME_MAX_INCOME:
        excess = income - BAREME_MAX_INCOME
        return BAREME_MAX_TAX + excess * BAREME_RATE_PER_100_ABOVE_MAX / 100

    bareme = _load_bareme()

    # Round down to nearest 100
    lower = int(income // 100) * 100
    upper = lower + 100

    if lower < 100:
        # Below first entry (100): linear from 0
        return income * bareme.get(100, 1.0) / 100

    lower_tax = bareme.get(lower, 0.0)
    upper_tax = bareme.get(upper, lower_tax)

    if upper == lower:
        return lower_tax

    # Linear interpolation between the two 100-CHF steps
    frac = (income - lower) / 100
    return lower_tax + frac * (upper_tax - lower_tax)


def calc_cantonal_tax(
    cantonal_taxable: float,
    married: bool,
    children: int,
    commune_coeff: float,
    church_tax: bool,
) -> dict:
    """ICC calculation using quotient familial."""
    cantonal_coeff = 155.0

    # LRIPP (law 642.12) — reduction on cantonal portion only
    # 2024: 3.5%, 2025: 4.0%, 2026: 5.0%
    lripp_reduction = 5.0  # percent for 2026

    # Quotient familial parts (Art. 43 LI)
    if married:
        parts = 1.8
    elif children > 0:
        parts = 1.3  # single parent (Art. 43, al. 2, let. c)
    else:
        parts = 1.0
    parts += 0.5 * children

    # Income per part
    income_per_part = cantonal_taxable / parts if parts > 0 else cantonal_taxable

    # Base tax per part from barème
    tax_per_part = _bareme_base_tax(income_per_part)

    # Multiply back by parts
    base_tax = tax_per_part * parts

    # Church tax adds ~10% to communal coefficient
    church_addition = commune_coeff * 0.10 if church_tax else 0

    # Cantonal part with LRIPP reduction (Art. 4 LRIPP — does not apply to communal)
    cantonal_part = base_tax * cantonal_coeff / 100 * (1 - lripp_reduction / 100)
    communal_part = base_tax * commune_coeff / 100
    church_part = base_tax * church_addition / 100

    # Final ICC
    icc = cantonal_part + communal_part + church_part

    return {
        "Quotient familial parts": parts,
        "Income per part": round(income_per_part, 2),
        "Base tax per part (barème)": round(tax_per_part, 2),
        "Base tax (× parts)": round(base_tax, 2),
        "Cantonal coefficient": f"{cantonal_coeff}%",
        "LRIPP reduction": f"{lripp_reduction}%",
        "Cantonal part (after LRIPP)": round(cantonal_part, 2),
        "Communal coefficient": f"{commune_coeff}%",
        "Communal part": round(communal_part, 2),
        "Church addition": f"{church_addition:.1f}%",
        "Church part": round(church_part, 2),
        "Cantonal + communal tax (ICC)": round(icc, 2),
    }


# ---------------------------------------------------------------------------
# Allocations familiales (family allowances) — universal, not income-dependent
# ---------------------------------------------------------------------------

def calc_allocations_familiales(children: int) -> dict:
    """Family allowances for Canton de Vaud (LAFam / LVLAFam).

    CHF 322/month per child (0-16), CHF 425/month (16-25 in education).
    For simplicity, assumes all children are 0-16.
    """
    monthly = 322 * children
    annual = monthly * 12
    return {
        "Per child (monthly)": 322,
        "Total monthly": monthly,
        "Total annual": annual,
    }


# ---------------------------------------------------------------------------
# LAMal subsidy estimation (subside assurance-maladie)
# ---------------------------------------------------------------------------

def estimate_lamal_subsidy(
    cantonal_taxable: float,
    pillar_3a: float,
    married: bool,
    children: int,
) -> dict:
    """Rough estimate of LAMal subsidy eligibility (LHPS, Canton de Vaud).

    RDU ≈ taxable income + 3a - health premium forfait.
    Rule: premiums should not exceed 10% of RDU.
    """
    adults = 2 if married else 1
    # RDU health premium forfait
    premium_forfait = 2_200 * adults + 1_300 * children

    # Simplified RDU (ignores fortune component)
    rdu = max(0, cantonal_taxable + pillar_3a - premium_forfait)

    # Reference premiums 2025, Region 1 (monthly)
    adult_premium_monthly = 671
    child_premium_monthly = 161
    total_monthly = adult_premium_monthly * adults + child_premium_monthly * children
    annual_premiums = total_monthly * 12

    # 10% rule
    ten_pct = rdu * 0.10
    excess_burden = max(0, annual_premiums - ten_pct)

    eligible = excess_burden > 0 and rdu > 0

    return {
        "RDU (approx.)": round(rdu, 2),
        "Annual ref. premiums": round(annual_premiums, 2),
        "10% of RDU": round(ten_pct, 2),
        "Excess burden": round(excess_burden, 2),
        "Potentially eligible": eligible,
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Swiss Tax Calculator (Vaud)", layout="wide")
    st.title("Swiss Tax Calculator — Canton de Vaud")
    st.caption("MVP for validating scraped tax data files (2025/2026)")

    communes = load_communes()
    commune_names = sorted(communes.keys())

    # Default to Lausanne
    default_idx = commune_names.index("Lausanne") if "Lausanne" in commune_names else 0

    # ---- Sidebar inputs ----
    with st.sidebar:
        st.header("Parameters")
        gross = st.number_input(
            "Gross annual salary (CHF)",
            min_value=30_000,
            max_value=500_000,
            value=100_000,
            step=1_000,
            help="Total yearly salary before any deductions (brut annuel).",
        )
        marital = st.selectbox(
            "Marital status",
            ["Single", "Married"],
            help="Married status applies the 'splitting' tariff for federal tax and a higher quotient familial for cantonal tax.",
        )
        married = marital == "Married"
        children = st.slider(
            "Number of children", 0, 5, 0,
            help="Each child adds 0.5 to the quotient familial (cantonal) and qualifies for federal child deductions.",
        )
        commune = st.selectbox(
            "Commune", commune_names, index=default_idx,
            help="Each commune in Vaud sets its own tax coefficient applied on top of the cantonal base tax.",
        )
        church_tax = st.checkbox(
            "Church tax (~+10% communal coeff)",
            value=False,
            help="If you are registered with a recognised church, an additional ~10% is added to the communal tax coefficient.",
        )
        children_in_daycare = st.slider(
            "Children in daycare (under 14)", 0, max(children, 1), 0,
            help="Number of children in third-party care. "
                 "Deduction: CHF 15,200/child cantonal (Art. 37 let. k), "
                 "CHF 25,500/child federal (LIFD Art. 33).",
        ) if children > 0 else 0
        has_canteen = st.checkbox(
            "Employer canteen available",
            value=False,
            help="Reduces meal deduction from CHF 3,200 to CHF 1,600 (code 150).",
        )
        has_2nd_pillar = st.checkbox(
            "Has 2nd pillar (BVG/LPP)",
            value=True,
            help="BVG (Berufliche Vorsorge) / LPP (Prévoyance professionnelle) — mandatory occupational pension fund (2nd pillar). Affects the Pillar 3a deduction cap.",
        )
        age = st.slider(
            "Age (for BVG rate)",
            20, 65, 35,
            help="BVG contribution rates increase with age: 0% (<25), 7% (25-34), 10% (35-44), 15% (45-54), 18% (55+).",
        )

    commune_coeff = communes[commune]

    # ---- Calculations ----
    social = calc_social_contributions(gross)
    social_total = social["Total social contributions"]

    bvg = calc_bvg(gross, age)
    bvg_employee = bvg["BVG employee share"]

    taxable = calc_taxable_income(
        gross, social_total, bvg_employee, married, children,
        has_2nd_pillar, has_canteen, children_in_daycare,
    )
    cantonal_taxable = taxable["Cantonal taxable income"]
    federal_taxable = taxable["Federal taxable income"]

    federal = calc_federal_tax(federal_taxable, married, children)
    federal_tax = federal["Federal tax (IFD)"]

    cantonal = calc_cantonal_tax(cantonal_taxable, married, children, commune_coeff, church_tax)
    icc = cantonal["Cantonal + communal tax (ICC)"]

    # Allocations familiales (family allowances)
    alloc_fam = calc_allocations_familiales(children)
    alloc_annual = alloc_fam["Total annual"]

    # LAMal subsidy estimate
    pillar_3a_amount = taxable["Pillar 3a"]
    lamal = estimate_lamal_subsidy(cantonal_taxable, pillar_3a_amount, married, children)

    total_deductions = social_total + bvg_employee + federal_tax + icc
    net_annual = gross - total_deductions + alloc_annual
    net_monthly = net_annual / 12
    effective_rate = (total_deductions / gross * 100) if gross > 0 else 0

    # ---- Summary ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Net monthly salary", f"CHF {net_monthly:,.0f}")
    col2.metric("Net annual salary", f"CHF {net_annual:,.0f}")
    col3.metric("Total deductions", f"CHF {total_deductions:,.0f}")
    col4.metric("Effective deduction rate", f"{effective_rate:.1f}%")

    st.info(
        "Cantonal barème: full 3,000-point official table (2025). "
        "Federal tax: official 2026 brackets (LIFD Art. 36). "
        "LRIPP cantonal reduction: 5% (2026).",
        icon="ℹ️",
    )

    # ---- Breakdown table ----
    st.subheader("Payslip Breakdown")

    breakdown = {
        "Gross salary": gross,
        "— AVS/AI/APG *(Old-age, Disability & Loss-of-earnings insurance)*": -social["AVS/AI/APG (5.3%)"],
        "— AC *(Unemployment insurance)*": -social["AC (1.1%)"],
        "— AANP *(Non-occupational accident insurance)*": -social["AANP (1.06%)"],
        "— BVG *(Occupational pension – 2nd pillar)*": -bvg_employee,
        "— IFD *(Federal direct tax)*": -federal_tax,
        "— ICC *(Cantonal + communal tax)*": -icc,
    }
    if alloc_annual > 0:
        breakdown["+ Allocations familiales *(Family allowances)*"] = alloc_annual
    breakdown["= Net annual salary"] = net_annual

    left, right = st.columns([1, 1])

    with left:
        for label, val in breakdown.items():
            c1, c2 = st.columns([3, 1])
            c1.write(label)
            if val < 0:
                c2.write(f"CHF {val:,.2f}")
            else:
                c2.write(f"**CHF {val:,.2f}**")

    # ---- Bar chart ----
    with right:
        measures = ["absolute", "relative", "relative", "relative", "relative"]
        x_labels = [
            "Gross",
            "Social<br><sup>(AVS/AI/APG+AC+AANP)</sup>",
            "BVG<br><sup>(Pension 2nd pillar)</sup>",
            "Federal tax<br><sup>(IFD)</sup>",
            "ICC<br><sup>(Cantonal+communal)</sup>",
        ]
        y_vals = [gross, -social_total, -bvg_employee, -federal_tax, -icc]
        text_vals = [
            f"{gross:,.0f}", f"-{social_total:,.0f}", f"-{bvg_employee:,.0f}",
            f"-{federal_tax:,.0f}", f"-{icc:,.0f}",
        ]
        if alloc_annual > 0:
            measures.append("relative")
            x_labels.append("Alloc. fam.<br><sup>(Family allow.)</sup>")
            y_vals.append(alloc_annual)
            text_vals.append(f"+{alloc_annual:,.0f}")
        measures.append("total")
        x_labels.append("Net")
        y_vals.append(0)
        text_vals.append(f"{net_annual:,.0f}")

        fig = go.Figure(
            go.Waterfall(
                orientation="v",
                measure=measures,
                x=x_labels,
                y=y_vals,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#2ecc71"}},
                decreasing={"marker": {"color": "#e74c3c"}},
                totals={"marker": {"color": "#3498db"}},
                textposition="outside",
                text=text_vals,
            )
        )
        fig.update_layout(
            title="Gross → Net breakdown",
            showlegend=False,
            height=400,
            yaxis_title="CHF",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Detail sections ----
    with st.expander("Social contributions detail — AVS/AI/APG, AC, AANP"):
        st.caption(
            "**AVS** = Old-age & survivors insurance · "
            "**AI** = Disability insurance · "
            "**APG** = Loss-of-earnings compensation · "
            "**AC** = Unemployment insurance · "
            "**AANP** = Non-occupational accident insurance"
        )
        for k, v in social.items():
            st.write(f"**{k}**: CHF {v:,.2f}")

    with st.expander("BVG / LPP detail — Occupational pension (2nd pillar)"):
        st.write(f"**Entry threshold**: CHF 22,680")
        st.write(f"**Coordination deduction**: CHF 26,460")
        for k, v in bvg.items():
            if isinstance(v, float):
                st.write(f"**{k}**: CHF {v:,.2f}")
            else:
                st.write(f"**{k}**: {v}")

    with st.expander("Deductions & taxable income"):
        for k, v in taxable.items():
            st.write(f"**{k}**: CHF {v:,.2f}")

    with st.expander("Federal tax (IFD) calculation — Impot federal direct"):
        for k, v in federal.items():
            st.write(f"**{k}**: CHF {v:,.2f}")
        st.caption("Brackets from LIFD (Loi federale sur l'impot fédéral direct) Art. 36 (2026)")

    with st.expander("Cantonal + communal tax (ICC) calculation — Impot cantonal et communal"):
        for k, v in cantonal.items():
            if isinstance(v, str):
                st.write(f"**{k}**: {v}")
            else:
                st.write(f"**{k}**: CHF {v:,.2f}" if isinstance(v, float) else f"**{k}**: {v}")
        st.caption(f"Commune: {commune} — Coefficient: {commune_coeff}%")

    if children > 0:
        with st.expander("Allocations familiales — Family allowances (LAFam)"):
            st.write(f"**Per child (monthly)**: CHF {alloc_fam['Per child (monthly)']:,}")
            st.write(f"**Total monthly**: CHF {alloc_fam['Total monthly']:,}")
            st.write(f"**Total annual**: CHF {alloc_fam['Total annual']:,}")
            st.caption(
                "Universal allowance, not income-dependent. "
                "CHF 322/month per child (0-16), CHF 425/month (16-25 in education). "
                "Assumes all children are 0-16."
            )

    with st.expander("LAMal subsidy estimate — Subside assurance-maladie"):
        if lamal["Potentially eligible"]:
            st.success(
                f"Based on approximate RDU of CHF {lamal['RDU (approx.)']:,.0f}, "
                f"you may be eligible for a LAMal subsidy of up to "
                f"CHF {lamal['Excess burden']:,.0f}/year."
            )
        else:
            st.write("Based on the rough estimate, you are **not likely eligible** for a LAMal subsidy.")
        st.write(f"**RDU (approx.)**: CHF {lamal['RDU (approx.)']:,.2f}")
        st.write(f"**Annual ref. premiums (est.)**: CHF {lamal['Annual ref. premiums']:,.2f}")
        st.write(f"**10% of RDU**: CHF {lamal['10% of RDU']:,.2f}")
        st.caption(
            "Rough estimate only. Actual subsidy depends on fortune, premium region, "
            "and the annual subsidy rate table published by OVAM. "
            "Reference premiums: Region 1 (2025). Fortune component not included."
        )


if __name__ == "__main__":
    main()
