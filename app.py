import streamlit as st
import pandas as pd

from inference import (
    CreditScoreInferenceEngine,
    OCCUPATION_OPTIONS,
    PAYMENT_BEHAVIOUR_OPTIONS,
)

# Page Config 
st.set_page_config(
    page_title="Credit Score Classifier",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_resource
def load_engine():
    return CreditScoreInferenceEngine()

engine = load_engine()

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1B3A6B;
        margin-bottom: 0;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        text-align: center;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    .result-desc {
        font-size: 0.95rem;
        margin-top: 0.5rem;
        color: #374151;
    }
    .stButton button {
        width: 100%;
        background-color: #1B3A6B;
        color: white;
        font-weight: 600;
        padding: 0.6rem;
        border-radius: 8px;
        border: none;
    }
    .stButton button:hover {
        background-color: #2E5FAC;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Credit Score Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Prediksi performa kredit nasabah berdasarkan profil finansial — '
    'Model: Random Forest Tuned (Accuracy 73.40%)</p>',
    unsafe_allow_html=True,
)

st.divider()

with st.form("credit_form"):
    st.subheader("Profil Nasabah")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Usia", min_value=10, max_value=100, value=35)
    with col2:
        occupation = st.selectbox("Pekerjaan", OCCUPATION_OPTIONS, index=4)
    with col3:
        credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"], index=1)
    with col4:
        payment_min = st.selectbox("Hanya Bayar Minimum?", ["No", "Yes"], index=0)

    st.subheader("Pendapatan & Tabungan")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        annual_income = st.number_input("Pendapatan Tahunan (USD)", min_value=0.0, value=50000.0, step=1000.0)
    with col6:
        monthly_salary = st.number_input("Gaji Bersih Bulanan (USD)", min_value=0.0, value=4000.0, step=100.0)
    with col7:
        monthly_balance = st.number_input("Saldo Bulanan (USD)", value=500.0, step=50.0)
    with col8:
        amount_invested = st.number_input("Investasi Bulanan (USD)", min_value=0.0, value=100.0, step=10.0)

    st.subheader("Profil Kredit")
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        num_bank_accounts = st.number_input("Jumlah Rekening Bank", min_value=0, value=3)
    with col10:
        num_credit_card = st.number_input("Jumlah Kartu Kredit", min_value=0, value=3)
    with col11:
        interest_rate = st.number_input("Suku Bunga (%)", min_value=0.0, value=15.0, step=0.5)
    with col12:
        num_loan = st.number_input("Jumlah Pinjaman Aktif", min_value=0, value=2)

    col13, col14, col15, col16 = st.columns(4)
    with col13:
        outstanding_debt = st.number_input("Total Utang Outstanding (USD)", min_value=0.0, value=1000.0, step=100.0)
    with col14:
        credit_util = st.number_input("Rasio Penggunaan Kredit (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    with col15:
        total_emi = st.number_input("Total Cicilan/Bulan (USD)", min_value=0.0, value=200.0, step=10.0)
    with col16:
        credit_history = st.number_input("Lama Riwayat Kredit (bulan)", min_value=0, value=60)

    st.subheader("Riwayat Pembayaran")
    col17, col18, col19, col20 = st.columns(4)
    with col17:
        delay_due_date = st.number_input("Rata-rata Keterlambatan (hari)", min_value=0, value=10)
    with col18:
        num_delayed = st.number_input("Jumlah Pembayaran Terlambat", min_value=0, value=5)
    with col19:
        changed_limit = st.number_input("Perubahan Limit Kredit (%)", value=5.0, step=0.5)
    with col20:
        num_inquiries = st.number_input("Jumlah Inquiry Kredit", min_value=0, value=3)

    payment_behaviour = st.selectbox("Perilaku Pembayaran", PAYMENT_BEHAVIOUR_OPTIONS, index=4)

    submitted = st.form_submit_button("🔍 Prediksi Credit Score")

if submitted:
    input_data = {
        "Age": age,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": num_loan,
        "Delay_from_due_date": delay_due_date,
        "Num_of_Delayed_Payment": num_delayed,
        "Changed_Credit_Limit": changed_limit,
        "Num_Credit_Inquiries": num_inquiries,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_util,
        "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": amount_invested,
        "Monthly_Balance": monthly_balance,
        "Credit_History_Months": credit_history,
        "Credit_Mix": credit_mix,
        "Occupation": occupation,
        "Payment_Behaviour": payment_behaviour,
        "Payment_of_Min_Amount": payment_min,
    }

    result = engine.predict(input_data)
    label = result["label"]
    color = result["color"]
    proba = result["probabilities"]

    st.markdown(
        f"""
        <div class="result-card" style="background-color:{color}22; border: 2px solid {color};">
            <p class="result-label" style="color:{color};">{label}</p>
            <p class="result-desc">{result['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Probabilitas per Kelas")
    proba_df = pd.DataFrame({
        "Kelas": list(proba.keys()),
        "Probabilitas (%)": list(proba.values()),
    }).sort_values("Probabilitas (%)", ascending=False)

    st.bar_chart(proba_df.set_index("Kelas"), color="#1B3A6B")

    col_a, col_b, col_c = st.columns(3)
    for col, cls in zip([col_a, col_b, col_c], ["Good", "Standard", "Poor"]):
        with col:
            st.metric(label=cls, value=f"{proba[cls]:.1f}%")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Credit Score Classification — Model Random Forest Tuned (Accuracy 73.40%) | "
    "Data Science — Bina Nusantara University"
)
