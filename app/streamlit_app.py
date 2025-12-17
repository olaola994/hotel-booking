import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "model" / "hotel_model_streamlit.pkl")

st.set_page_config(page_title="Hotel Cancellation Predictor")

st.title("Hotel Booking Cancellation Predictor")
st.write("Predicts the probability that a booking will be cancelled.")

st.subheader("Booking details")

lead_time = st.number_input("Lead time (days)", 0, 700, 50)
has_previous_cancellations = st.selectbox(
    "Has the customer cancelled before?",
    ["No", "Yes"]
)

has_previous_cancellations = 1 if has_previous_cancellations == "Yes" else 0
booking_changes = st.number_input("Booking changes", 0, 20, 0)
total_of_special_requests = st.number_input("Special requests", 0, 10, 0)
adr = st.number_input("Average Daily Rate (ADR)", 0.0, 1000.0, 100.0)

deposit_type = st.selectbox(
    "Deposit type",
    ["No Deposit", "Refundable", "Non Refund"]
)

customer_type = st.selectbox(
    "Customer type",
    ["Transient", "Transient-Party", "Contract", "Group"]
)

if st.button("Predict cancellation risk"):

    input_df = pd.DataFrame([{
        "lead_time": lead_time,
        "has_previous_cancellations": has_previous_cancellations,
        "booking_changes": booking_changes,
        "total_of_special_requests": total_of_special_requests,
        "adr": adr,
        "deposit_type": deposit_type,
        "customer_type": customer_type
    }])

    prob = model.predict_proba(input_df)[0][1]
    threshold = 0.35  # bardzo rozsądny
    pred = 1 if prob >= threshold else 0
    st.caption(
        f"Classification threshold set to {threshold:.2f}. "
        "Lower threshold increases sensitivity to cancellations."
    )

    st.subheader("Result")

    if pred == 1:
        st.error(f"High cancellation risk ({prob:.2%})")
    else:
        st.success(f"Low cancellation risk ({prob:.2%})")

    st.caption(
        "The prediction is based on a simplified logistic regression model "
        "trained on selected booking characteristics."
    )
    st.write("Probability:", prob)