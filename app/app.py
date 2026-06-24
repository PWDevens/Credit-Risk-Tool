"""Credit Risk Scorecard — Win98 underwriting terminal (Streamlit).

A lender enters origination-time borrower/loan facts and sees PD, LGD, EAD, and the
resulting Expected Loss (EL = PD x LGD x EAD), with a model toggle (AutoML vs
fine-tuned, once trained). No price fields are collected — they are excluded inputs.

Run:  streamlit run app/app.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "modeling"))
sys.path.insert(0, str(REPO_ROOT / "data"))

from common.predictor import RiskPredictor, available_families  # noqa: E402

# DATA_DICTIONARY §10 — ListingCategory codes (stored as strings in the model).
PURPOSE = {
    "1": "Debt Consolidation", "2": "Home Improvement", "3": "Business", "4": "Personal",
    "5": "Student", "6": "Auto", "7": "Other", "8": "Baby & Adoption", "9": "Boat",
    "10": "Cosmetic", "11": "Engagement Ring", "12": "Green", "13": "Household",
    "14": "Large Purchases", "15": "Medical/Dental", "16": "Motorcycle", "17": "RV",
    "18": "Taxes", "19": "Vacation", "20": "Wedding", "0": "Not Available",
}
INCOME_RANGES = ["$0", "$1-24,999", "$25,000-49,999", "$50,000-74,999",
                 "$75,000-99,999", "$100,000+", "Not employed"]
EMPLOYMENT = ["Employed", "Full-time", "Part-time", "Self-employed", "Retired",
              "Other", "Not employed"]

CSS = """
<style>
  .stApp { background: #008080; }
  html, body, [class*="css"], .stApp, input, select, textarea, button {
    font-family: "Pixelated MS Sans Serif", "MS Sans Serif", Tahoma, "DejaVu Sans", sans-serif !important;
    -webkit-font-smoothing: none;
  }
  .block-container {
    background: #C0C0C0; padding: 6px 18px 48px 18px; max-width: 1100px;
    border: 2px solid; border-color: #FFFFFF #404040 #404040 #FFFFFF;
  }
  /* Win98 title bar */
  .w98-titlebar {
    background: linear-gradient(90deg, #000080, #1084D0); color: #fff; font-weight: bold;
    padding: 3px 6px; display: flex; justify-content: space-between; align-items: center;
    font-size: 13px; margin: 2px 0 8px 0;
  }
  .w98-titlebar .controls { letter-spacing: 2px; }
  .w98-group {
    background: #000080; color: #fff; font-weight: bold; font-size: 12px;
    padding: 2px 6px; margin: 10px 0 4px 0;
  }
  /* Sunken result window */
  .w98-window { border: 2px solid; border-color: #FFFFFF #404040 #404040 #FFFFFF; background: #C0C0C0; margin-bottom: 8px; }
  .w98-window .body { padding: 8px 10px 12px 10px; }
  .metric {
    display: flex; justify-content: space-between; align-items: center;
    background: #fff; border: 2px solid; border-color: #808080 #FFFFFF #FFFFFF #808080;
    padding: 5px 8px; margin: 5px 0; font-size: 13px;
  }
  .metric b { font-size: 15px; }
  .metric.el { background: #000080; color: #fff; }
  .metric.el b { color: #fff; font-size: 18px; }
  .verdict { text-align: center; font-weight: bold; padding: 6px; margin-top: 8px;
             border: 2px solid; border-color: #FFFFFF #404040 #404040 #FFFFFF; }
  .verdict.approve { background: #C0C0C0; color: #006000; }
  .verdict.refer   { background: #C0C0C0; color: #8a6d00; }
  .verdict.decline { background: #C0C0C0; color: #a00000; }
  /* Buttons: raised bevel, square, pressed = sunken */
  .stButton > button, .stFormSubmitButton > button {
    background: #C0C0C0 !important; color: #000 !important; font-weight: bold !important;
    border: 2px solid !important; border-color: #FFFFFF #404040 #404040 #FFFFFF !important;
    border-radius: 0 !important; box-shadow: none !important;
  }
  .stButton > button:active, .stFormSubmitButton > button:active {
    border-color: #404040 #FFFFFF #FFFFFF #404040 !important;
  }
  /* Inputs: sunken white wells */
  div[data-baseweb="input"] input, div[data-baseweb="select"] > div, .stNumberInput input {
    background: #fff !important; border-radius: 0 !important;
    border: 2px solid !important; border-color: #808080 #FFFFFF #FFFFFF #808080 !important;
  }
  /* Status bar */
  .w98-statusbar {
    position: relative; margin-top: 14px; display: flex; gap: 6px; font-size: 11px;
  }
  .w98-statusbar span {
    border: 2px solid; border-color: #808080 #FFFFFF #FFFFFF #808080; background: #C0C0C0;
    padding: 2px 8px; flex: 1;
  }
  .w98-statusbar span.fixed { flex: 0 0 auto; }
  h1, h2, h3 { font-size: 15px !important; }
</style>
"""


def util_bucket(u: float) -> str:
    if u is None or (isinstance(u, float) and math.isnan(u)):
        return "Missing"
    if u <= 0.30: return "<=30%"
    if u <= 0.50: return "30-50%"
    if u <= 0.75: return "50-75%"
    if u <= 1.00: return "75-100%"
    return ">100%"


def build_inputs(f: dict) -> dict:
    """Map friendly form values to a MODEL_FEATURES dict, computing derived features
    consistently with features.add_derived_features (avoids train/serve skew)."""
    monthly = float(f["StatedMonthlyIncome"])
    mid = float(f["credit_score_mid"])
    verifiable = bool(f["IncomeVerifiable"])
    prior_loans = float(f["TotalProsperLoans"])
    return {
        "LoanOriginalAmount": float(f["LoanOriginalAmount"]),
        "Term": float(f["Term"]),
        "ListingCategory (numeric)": f["purpose_code"],
        "StatedMonthlyIncome": monthly,
        "IncomeRange": f["IncomeRange"],
        "IncomeVerifiable": str(verifiable),
        "EmploymentStatus": f["EmploymentStatus"],
        "EmploymentStatusDuration": float(f["EmploymentStatusDuration"]),
        "IsBorrowerHomeowner": str(bool(f["IsBorrowerHomeowner"])),
        "CurrentlyInGroup": "False",
        "credit_score_mid": mid,
        "CreditScoreRangeLower": mid - 10,
        "CreditScoreRangeUpper": mid + 9,
        "credit_history_months": float(f["credit_history_years"]) * 12.0,
        "DebtToIncomeRatio": float(f["DebtToIncomeRatio"]),
        "BankcardUtilization": float(f["BankcardUtilization"]),
        "RevolvingCreditBalance": float(f["RevolvingCreditBalance"]),
        "AvailableBankcardCredit": float(f["AvailableBankcardCredit"]),
        "OpenCreditLines": float(f["OpenCreditLines"]),
        "CurrentDelinquencies": float(f["CurrentDelinquencies"]),
        "DelinquenciesLast7Years": float(f["DelinquenciesLast7Years"]),
        "InquiriesLast6Months": float(f["InquiriesLast6Months"]),
        "PublicRecordsLast10Years": float(f["PublicRecordsLast10Years"]),
        "AmountDelinquent": float(f["AmountDelinquent"]),
        "TotalProsperLoans": prior_loans,
        "OnTimeProsperPayments": float(f["OnTimeProsperPayments"]),
        # derived
        "loan_to_income": float(f["LoanOriginalAmount"]) / max(monthly * 12.0, 1.0),
        "stated_income_log": math.log1p(max(monthly, 0.0)),
        "dti_capped_flag": 1 if float(f["DebtToIncomeRatio"]) >= 10.01 else 0,
        "income_unverified": 0 if verifiable else 1,
        "is_repeat_borrower": 1 if prior_loans > 0 else 0,
        "bankcard_util_bucket": util_bucket(float(f["BankcardUtilization"])),
    }


def decision(pd_: float, loss_rate: float) -> tuple[str, str]:
    if pd_ < 0.15 and loss_rate < 0.05:
        return "approve", "✓ APPROVE — risk within appetite"
    if pd_ < 0.35:
        return "refer", "⚠ REFER — approve only with risk-based pricing"
    return "decline", "✕ DECLINE — expected loss too high"


@st.cache_resource
def get_predictor(family: str) -> RiskPredictor:
    return RiskPredictor(family)


def render_result(family: str, loan_amt: float, r: dict) -> None:
    loss_rate = r["el"] / loan_amt if loan_amt else 0.0
    cls, verdict = decision(r["pd"], loss_rate)
    st.markdown(
        f"""
        <div class="w98-window">
          <div class="w98-titlebar"><span>📊 Risk Assessment — {family}</span><span class="controls">_ ▢ ✕</span></div>
          <div class="body">
            <div class="metric"><span>Probability of Default (PD)</span><b>{r['pd']:.1%}</b></div>
            <div class="metric"><span>Loss Given Default (LGD)</span><b>{r['lgd']:.1%}</b></div>
            <div class="metric"><span>Exposure at Default (EAD)</span><b>${r['ead']:,.0f}</b></div>
            <div class="metric el"><span>Expected Loss&nbsp;&nbsp;(PD × LGD × EAD)</span><b>${r['el']:,.0f}</b></div>
            <div class="metric"><span>Implied loss rate on ${loan_amt:,.0f}</span><b>{loss_rate:.2%}</b></div>
            <div class="verdict {cls}">{verdict}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Underwriting Terminal", page_icon="💾", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="w98-titlebar"><span>💾 Loan Application Review — Underwriting Terminal</span>'
        '<span class="controls">_ ▢ ✕</span></div>',
        unsafe_allow_html=True,
    )

    fams = available_families()
    if not fams:
        st.error("No trained models found. Run:  python modeling/train_baselines.py")
        st.stop()

    family = st.radio("Model", fams, horizontal=True,
                      format_func=lambda x: {"automl": "AutoML baseline",
                                             "finetuned": "Fine-tuned"}.get(x, x))

    left, right = st.columns([1.15, 1])
    with left:
        with st.form("application"):
            st.markdown('<div class="w98-group">Loan</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            loan_amt = c1.number_input("Amount ($)", 1000, 35000, 10000, step=500)
            term = c2.selectbox("Term (months)", [12, 36, 60], index=1)
            purpose_label = c3.selectbox("Purpose", list(PURPOSE.values()), index=0)

            st.markdown('<div class="w98-group">Income &amp; employment</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            income = c1.number_input("Stated monthly income ($)", 0, 60000, 5000, step=250)
            income_range = c2.selectbox("Income range", INCOME_RANGES, index=2)
            emp = c3.selectbox("Employment status", EMPLOYMENT, index=0)
            c1, c2, c3 = st.columns(3)
            emp_dur = c1.number_input("Months in job", 0, 600, 48)
            verifiable = c2.checkbox("Income verifiable", value=True)
            homeowner = c3.checkbox("Homeowner", value=False)

            st.markdown('<div class="w98-group">Credit profile</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            score = c1.slider("Credit score (mid)", 350, 850, 680)
            hist_years = c2.slider("Credit history (yrs)", 0, 40, 10)
            dti = c3.slider("Debt-to-income", 0.0, 1.0, 0.22, step=0.01)
            c1, c2, c3 = st.columns(3)
            util = c1.slider("Bankcard utilization", 0.0, 1.5, 0.50, step=0.05)
            revol = c2.number_input("Revolving balance ($)", 0, 200000, 10000, step=500)
            avail = c3.number_input("Available bankcard ($)", 0, 200000, 5000, step=500)
            c1, c2, c3 = st.columns(3)
            open_lines = c1.number_input("Open credit lines", 0, 60, 8)
            inq6 = c2.number_input("Inquiries (6 mo)", 0, 30, 1)
            delinq7 = c3.number_input("Delinquencies (7 yr)", 0, 60, 0)
            c1, c2, c3 = st.columns(3)
            cur_delinq = c1.number_input("Current delinquencies", 0, 30, 0)
            pub_rec = c2.number_input("Public records (10 yr)", 0, 30, 0)
            amt_delinq = c3.number_input("Amount delinquent ($)", 0, 100000, 0, step=100)

            st.markdown('<div class="w98-group">Prior Prosper history</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            prior_loans = c1.number_input("Prior Prosper loans", 0, 30, 0)
            ontime = c2.number_input("On-time prior payments", 0, 500, 0)

            submitted = st.form_submit_button("▶  Run Risk Assessment")

    with right:
        if submitted:
            code = next(k for k, v in PURPOSE.items() if v == purpose_label)
            form = dict(
                LoanOriginalAmount=loan_amt, Term=term, purpose_code=code,
                StatedMonthlyIncome=income, IncomeRange=income_range, EmploymentStatus=emp,
                EmploymentStatusDuration=emp_dur, IncomeVerifiable=verifiable,
                IsBorrowerHomeowner=homeowner, credit_score_mid=score,
                credit_history_years=hist_years, DebtToIncomeRatio=dti, BankcardUtilization=util,
                RevolvingCreditBalance=revol, AvailableBankcardCredit=avail,
                OpenCreditLines=open_lines, InquiriesLast6Months=inq6,
                DelinquenciesLast7Years=delinq7, CurrentDelinquencies=cur_delinq,
                PublicRecordsLast10Years=pub_rec, AmountDelinquent=amt_delinq,
                TotalProsperLoans=prior_loans, OnTimeProsperPayments=ontime,
            )
            inputs = build_inputs(form)
            with st.spinner("Scoring…"):
                result = get_predictor(family).assess(inputs)
            render_result(family, float(loan_amt), result)
        else:
            st.markdown(
                '<div class="w98-window"><div class="w98-titlebar"><span>📊 Risk Assessment</span>'
                '<span class="controls">_ ▢ ✕</span></div><div class="body">'
                'Enter the application on the left and press <b>Run Risk Assessment</b>.'
                '</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div class="w98-statusbar"><span>Ready</span>'
        f'<span class="fixed">Model: {family}</span>'
        f'<span class="fixed">EL = PD × LGD × EAD</span></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
