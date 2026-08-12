import streamlit as st
from model_utils import load_model, prepare_input_dataframe

# --- Page Setup ---
st.set_page_config(
    page_title="AVM Home Price Predictor", page_icon="🏠", layout="centered"
)

st.title("🏠 California Home Price Estimator")
st.markdown(
    "Provide basic property characteristics to receive an automated valuation model (AVM) price prediction."
)


# --- Load Pipeline ---
@st.cache_resource
def get_pipeline():
    return load_model("data/processed/best_gb_model.joblib")


try:
    model_pipeline = get_pipeline()
    st.success("Tuned Model Pipeline loaded successfully!", icon="✅")
except Exception as e:
    st.error(f"Failed to load model pipeline: {e}")
    st.stop()

st.divider()

# --- User Inputs Form ---
with st.form("prediction_form"):
    st.subheader("Property Features")

    col1, col2 = st.columns(2)

    with col1:
        living_area = st.number_input(
            "Living Area (sq ft)",
            min_value=300,
            max_value=15000,
            value=2000,
            step=50,
        )
        bedrooms = st.number_input(
            "Bedrooms", min_value=1, max_value=10, value=3, step=1
        )
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
        )
        year_built = st.number_input(
            "Year Built", min_value=1800, max_value=2026, value=1998, step=1
        )
        property_type = st.selectbox(
            "Property Type", ["Residential", "Condominium", "Townhouse"]
        )

    with col2:
        lot_size = st.number_input(
            "Lot Size (sq ft)",
            min_value=500,
            max_value=100000,
            value=6000,
            step=100,
        )
        garage_spaces = st.number_input(
            "Garage Spaces", min_value=0, max_value=6, value=2, step=1
        )
        stories = st.number_input(
            "Stories", min_value=1, max_value=4, value=1, step=1
        )
        latitude = st.number_input("Latitude", value=32.7157, format="%.4f")
        longitude = st.number_input("Longitude", value=-117.1611, format="%.4f")

    submit_button = st.form_submit_button(
        "Estimate Price", use_container_width=True
    )

# --- Prediction Execution ---
if submit_button:
    # Prepare row matching exact 58 columns of X_train
    input_df = prepare_input_dataframe(
        living_area=living_area,
        lot_size=lot_size,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        year_built=year_built,
        latitude=latitude,
        longitude=longitude,
        property_type=property_type,
        stories=stories,
        garage_spaces=garage_spaces,
    )

    try:
        predicted_price = model_pipeline.predict(input_df)[0]

        st.divider()
        st.subheader("Valuation Summary")
        st.metric(
            label="Predicted Close Price", value=f"${predicted_price:,.2f}"
        )

    except Exception as e:
        st.error(f"Error executing prediction pipeline: {e}")