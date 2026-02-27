"""Tests for improved claim categorizer — verifying that expanded keyword sets
correctly classify claims that previously fell through to 'general'."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from veritas.claim_extract import _classify_category


# ── Finance claims that were previously misclassified as 'general' ──


def test_finance_asset_management():
    """Asset management language should trigger finance."""
    assert _classify_category(
        "BlackRock manages approximately 10 trillion dollars in assets"
    ) == "finance"


def test_finance_hedge_fund():
    """Hedge fund language should trigger finance."""
    assert _classify_category(
        "Renaissance Technologies is the most successful hedge fund in history"
    ) == "finance"


def test_finance_portfolio_returns():
    """Portfolio returns language should trigger finance."""
    assert _classify_category(
        "The portfolio returned 66 percent annually before fees"
    ) == "finance"


def test_finance_wall_street():
    """Wall Street references should trigger finance."""
    assert _classify_category(
        "Wall Street firms have been increasing their investments in AI"
    ) == "finance"


def test_finance_equity_shares():
    """Equity/shares language should trigger finance."""
    assert _classify_category(
        "The company issued new shares and the equity valuation doubled"
    ) == "finance"


def test_finance_growth_percent():
    """Growth + percent should trigger finance."""
    assert _classify_category(
        "Revenue growth reached 14 percent driven by strong demand"
    ) == "finance"


# ── Health claims with expanded vocabulary ──


def test_health_cholesterol_ldl():
    """LDL cholesterol claims should trigger health."""
    assert _classify_category(
        "High LDL cholesterol is associated with increased coronary risk"
    ) == "health"


def test_health_inflammation():
    """Inflammation claims should trigger health."""
    assert _classify_category(
        "Chronic inflammation of the arteries leads to atherosclerosis"
    ) == "health"


def test_health_mediterranean_diet():
    """Diet study references should trigger health."""
    assert _classify_category(
        "The Mediterranean diet significantly reduced cardiac mortality"
    ) == "health"


def test_health_saturated_fat():
    """Nutrition claims should trigger health."""
    assert _classify_category(
        "Saturated fat intake and triglyceride levels affect heart disease"
    ) == "health"


def test_health_placebo():
    """Clinical trial terms should trigger health."""
    assert _classify_category(
        "The randomized double-blind placebo trial showed significant results"
    ) == "health"


# ── Science claims with expanded vocabulary ──


def test_science_researchers():
    """Researcher language should trigger science."""
    assert _classify_category(
        "Researchers at the university published findings in a peer-reviewed journal"
    ) == "science"


def test_science_mathematical():
    """Mathematics language should trigger science."""
    assert _classify_category(
        "The mathematical theorem proved that the equation has no solution"
    ) == "science"


def test_science_statistical():
    """Statistical language should trigger science."""
    assert _classify_category(
        "The correlation was statistically significant with a large sample size"
    ) == "science"


# ── Tech claims with expanded vocabulary ──


def test_tech_platform():
    """Platform/digital language should trigger tech."""
    assert _classify_category(
        "The digital platform uses blockchain technology and encryption"
    ) == "tech"


def test_tech_quantum():
    """Quantum computing should trigger tech."""
    assert _classify_category(
        "Quantum computing and advanced processors will transform automation"
    ) == "tech"


# ── Existing categories should still work ──


def test_finance_still_works():
    """Original finance terms should still work."""
    assert _classify_category(
        "The Federal Reserve raised interest rates by 25 basis points"
    ) == "finance"


def test_health_still_works():
    """Original health terms should still work."""
    assert _classify_category(
        "The FDA approved a new cancer treatment drug"
    ) == "health"


def test_tech_still_works():
    """Original tech terms should still work."""
    assert _classify_category(
        "OpenAI released a new AI model trained on GPUs"
    ) == "tech"


def test_general_still_default():
    """Claims without category signals should still default to general."""
    assert _classify_category(
        "Something interesting happened yesterday in the neighborhood"
    ) == "general"


def test_threshold_still_two():
    """Single keyword hit should still default to general (threshold=2)."""
    # Only "market" matches finance — needs >= 2
    assert _classify_category(
        "He went to the market to buy groceries"
    ) == "general"


# ── Education category ──


def test_education_tuition():
    """Tuition / student language should trigger education."""
    assert _classify_category(
        "College tuition has increased 1200 percent and student loans now exceed 1.7 trillion"
    ) == "education"


def test_education_enrollment():
    """Enrollment / school language should trigger education."""
    assert _classify_category(
        "Enrollment in public schools dropped by 1.3 million students"
    ) == "education"


def test_education_graduation():
    """Graduation / degree language should trigger education."""
    assert _classify_category(
        "The graduation rate for students with a bachelor degree has improved"
    ) == "education"


def test_education_teacher():
    """Teacher / classroom language should trigger education."""
    assert _classify_category(
        "Teachers in the classroom are facing increased student-to-teacher ratios"
    ) == "education"


def test_education_test_scores():
    """Test scores / literacy should trigger education."""
    assert _classify_category(
        "Standardized test scores show declining literacy among high school students"
    ) == "education"


# ── Energy / Climate category ──


def test_energy_climate_carbon():
    """Carbon emissions language should trigger energy_climate."""
    assert _classify_category(
        "Carbon emissions from fossil fuels reached record levels in 2023"
    ) == "energy_climate"


def test_energy_climate_renewable():
    """Renewable energy language should trigger energy_climate."""
    assert _classify_category(
        "Solar and wind power now generate 30 percent of renewable energy"
    ) == "energy_climate"


def test_energy_climate_ev():
    """Electric vehicle language should trigger energy_climate."""
    assert _classify_category(
        "Electric vehicle battery technology has improved sustainable transportation"
    ) == "energy_climate"


def test_energy_climate_warming():
    """Climate change / temperature should trigger energy_climate."""
    assert _classify_category(
        "Global warming is causing temperature increases and sea level rise"
    ) == "energy_climate"


def test_energy_climate_pollution():
    """Pollution / greenhouse should trigger energy_climate."""
    assert _classify_category(
        "Greenhouse gas pollution from methane and carbon continues to increase"
    ) == "energy_climate"


# ── Labor category ──


def test_labor_unemployment():
    """Unemployment / jobs language should trigger labor."""
    assert _classify_category(
        "Unemployment dropped and the labor force participation rate increased"
    ) == "labor"


def test_labor_wages():
    """Wages / salary language should trigger labor."""
    assert _classify_category(
        "Minimum wage workers saw salary increases after the new labor policy"
    ) == "labor"


def test_labor_union():
    """Union / strike language should trigger labor."""
    assert _classify_category(
        "The union organized a strike for better wages and collective bargaining"
    ) == "labor"


def test_labor_gig_economy():
    """Gig economy / remote work should trigger labor."""
    assert _classify_category(
        "Remote work and the gig economy have transformed the workforce"
    ) == "labor"


def test_labor_layoffs():
    """Layoff / hiring language should trigger labor."""
    assert _classify_category(
        "Tech layoffs accelerated while hiring in other jobs recovered"
    ) == "labor"
