import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
from texts2 import TEXTS

if "prob" not in st.session_state:
    st.session_state.prob = None

st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    layout="centered"
)

st.markdown("""
<style>
.badge {
    display:inline-block;
    padding:0.6em 1em;
    border-radius:16px;
    font-weight:600;
    color:white;
    text-align:center;
}
.badge-green { background:#2ecc71; }
.badge-red { background:#e74c3c; }
.badge-orange { background:#f39c12; }

.kpi-box {
    padding:1.2rem;
    border-radius:14px;
    background:#f5f7fa;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)


BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "model" / "hotel_model_streamlit2.pkl")

with open(BASE_DIR / "model" / "metrics2.json") as f:
    metrics = json.load(f)

with open(BASE_DIR / "model" / "confusion_matrix2.json") as f:
    confusion = json.load(f)


with st.sidebar:
    language = st.selectbox("Language / Język", ["Polski", "English"])

lang = "pl" if language == "Polski" else "en"
T = TEXTS[lang]

tab_pred, tab_metrics = st.tabs([T["tab_pred"], T["tab_metrics"]])

with tab_pred:

    st.title(T["title"])
    st.write(T["description"])
    st.subheader(T["booking"])

    # ---------- INPUTS ----------
    lead_time = st.number_input(T["lead_time"], 0, 700, 50)
    adults = st.number_input(T["adults"], 0, 10, 2)
    children = st.number_input(T["children"], 0, 10, 0)
    prev_cancel = st.number_input(T["prev_cancel"], 0, 20, 0)
    prev_not_cancel = st.number_input(T["prev_not_cancel"], 0, 50, 0)
    requests = st.number_input(T["requests"], 0, 10, 0)
    price = st.number_input(T["price"], 0.0, 1000.0, 120.0)
    month_label = st.selectbox(T["month"], T["months"])
    month = T["months"].index(month_label) + 1

    dow_label = st.selectbox(T["dow"], T["dow_labels"])
    dow = T["dow_labels"].index(dow_label)

    segment_label = st.selectbox(
        T["kpi_segment"],
        list(T["segment_map"].keys())
    )
    market_segment = T["segment_map"][segment_label]
    st.caption(T["segment_help"])

    if st.button(T["calc"]):
        input_df = pd.DataFrame([{
            "lead_time": lead_time,
            "no_of_previous_cancellations": prev_cancel,
            "no_of_previous_bookings_not_canceled": prev_not_cancel,
            "no_of_special_requests": requests,
            "avg_price_per_room": price,
            "arrival_month_num": month,
            "arrival_day_of_week": dow,
            "no_of_adults": adults,
            "no_of_children": children,
            "market_segment_type": market_segment
        }])

        st.session_state.prob = model.predict_proba(input_df)[0][1]

    if st.session_state.prob is not None:

        prob = st.session_state.prob
        threshold = metrics["threshold"]

        st.markdown("---")

        st.markdown(
            f"<span class='badge {'badge-red' if prob >= threshold else 'badge-green'}'>"
            f"{T['high'] if prob >= threshold else T['low']}</span>",
            unsafe_allow_html=True
        )

        st.write(f"**{T['prob']}**: {prob:.2%}")

        st.markdown("### 📌 " + T["kpi_title"])
        c1, c2, c3 = st.columns(3)

        c1.markdown(
            f"<div class='kpi-box'><b>{T['kpi_risk']}</b><br>{prob:.0%}</div>",
            unsafe_allow_html=True
        )

        season = T["season_high"] if month in [6, 7, 8] else T["season_low"]
        c2.markdown(
            f"<div class='kpi-box'><b>{T['kpi_season']}</b><br>{season}</div>",
            unsafe_allow_html=True
        )

        c3.markdown(
            f"<div class='kpi-box'><b>{T['kpi_segment']}</b><br>{segment_label}</div>",
            unsafe_allow_html=True
        )

        with st.expander("ℹ️ " + T["explanation"]):
            reasons = []
            if lead_time > 120: reasons.append(T["exp_lead_time"])
            if prev_cancel > 0: reasons.append(T["exp_prev_cancel"])
            if price > 200: reasons.append(T["exp_price"])
            if month in [6, 7, 8]: reasons.append(T["exp_season"])
            if not reasons: reasons.append(T["exp_neutral"])
            for r in reasons:
                st.write(r)

        st.markdown("### 💰 " + T["business_impact_title"])

        nights = st.number_input(
            T["nights"],
            min_value=1,
            max_value=30,
            value=3,
            key="nights_sim"
        )

        loss_if_cancelled = nights * price
        expected_loss = prob * loss_if_cancelled

        st.markdown(
            f"<div class='kpi-box'><b>{T['expected_loss']}</b><br>"
            f"{expected_loss:,.2f} €</div>",
            unsafe_allow_html=True
        )

        st.markdown("### 🧠 " + T["business_recommendation_title"])

        risk_pct = prob * 100

        if risk_pct >= 70:
            st.error(T["rec_high_risk"])
        elif risk_pct >= 50 and expected_loss >= 150:
            st.warning(T["rec_high_risk_loss"])
        elif risk_pct >= 50:
            st.warning(T["rec_high_risk_only"])
        elif expected_loss >= 300:
            st.info(T["rec_low_risk_high_value"])
        else:
            st.success(T["rec_safe"])

with tab_metrics:

    st.title(T["metrics_title"])

    col1, col2 = st.columns(2)
    col1.metric(T["accuracy"], f"{metrics['accuracy']:.0%}")
    col1.metric(T["precision"], f"{metrics['precision']:.0%}")
    col2.metric(T["recall"], f"{metrics['recall']:.0%}")
    col2.metric(T["roc_auc"], f"{metrics['roc_auc']:.2f}")
    col2.metric(T["F1-score"], f"{metrics['f1']:.2f}")

    st.markdown("### " + T["confusion_title"])

    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:600px">
            <div class="badge badge-green">TP<br>{confusion['true_positive']}<br><small>{T['cm_tp']}</small></div>
            <div class="badge badge-red">FN<br>{confusion['false_negative']}<br><small>{T['cm_fn']}</small></div>
            <div class="badge badge-orange">FP<br>{confusion['false_positive']}<br><small>{T['cm_fp']}</small></div>
            <div class="badge badge-green">TN<br>{confusion['true_negative']}<br><small>{T['cm_tn']}</small></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        T["cm_summary"].format(
            fn=confusion["false_negative"],
            fp=confusion["false_positive"]
        )
    )
    st.markdown("### 💼 " + T["business_assumptions_title"])

    st.markdown(f"- {T['business_assumption_1']}")
    st.markdown(f"- {T['business_assumption_2']}")
    st.markdown(f"- {T['business_assumption_3']}")


    avg_booking_value = 3 * 120
    reaction_rate = 0.5
    effective_recall = metrics["recall"]

    estimated_saving_per_booking = avg_booking_value * effective_recall * reaction_rate
    estimated_monthly_saving = estimated_saving_per_booking * 100

    c1, c2 = st.columns(2)

    c1.markdown(
        f"<div class='kpi-box'><b>{T['business_savings_label']}</b><br>"
        f"{estimated_saving_per_booking:,.0f} €</div>",
        unsafe_allow_html=True
    )

    c2.markdown(
        f"<div class='kpi-box'><b>{T['business_savings_month']}</b><br>"
        f"{estimated_monthly_saving:,.0f} €</div>",
        unsafe_allow_html=True
    )

    st.caption(T["business_value_note"])