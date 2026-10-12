
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "Car_Sell_Dataset.csv"
MODEL_PATH = BASE_DIR / "model.h5"
SCALER_PATH = BASE_DIR / "scaler.pkl"


# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    # Remove surrounding spaces from categorical values
    for column in data.select_dtypes(include="object").columns:
        data[column] = data[column].str.strip()

    return data


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


@st.cache_resource
def load_artifacts():
    artifact_names = [
        "label_encoder_accidental",
        "label_encoder_transmission",
        "onehot_encoder_brand",
        "onehot_encoder_fuel_type",
        "onehot_encoder_model_name",
        "onehot_encoder_model_variant",
        "onehot_encoder_car_type",
        "onehot_encoder_state",
        "onehot_encoder_owner",
    ]

    artifacts = {}

    for name in artifact_names:
        with open(BASE_DIR / f"{name}.pkl", "rb") as file:
            artifacts[name] = pickle.load(file)

    with open(SCALER_PATH, "rb") as file:
        artifacts["scaler"] = pickle.load(file)

    return artifacts


try:
    data = load_data()
    model = load_model()
    artifacts = load_artifacts()

except Exception as error:
    st.error(f"Unable to load model or files: {error}")
    st.stop()


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("🚗 Car Selling Price Prediction")
st.write(
    "Select your car details from the dropdowns "
    "to estimate its selling price."
)

st.divider()


# --------------------------------------------------
# DEPENDENT DROPDOWN OPTIONS
# --------------------------------------------------

brands = sorted(data["Brand"].dropna().unique().tolist())

brand = None
car_model = None
variant = None

col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox(
        "Brand",
        options=brands,
        index=None,
        placeholder="Select brand"
    )

if brand is not None:
    model_options = sorted(
        data.loc[
            data["Brand"] == brand, "Model_Name"
        ].dropna().unique().tolist()
    )
else:
    model_options = []

with col2:
    car_model = st.selectbox(
        "Model",
        options=model_options,
        index=None,
        placeholder="Select model",
        disabled=(brand is None),
        key="car_model"
    )

if brand is not None and car_model is not None:
    variant_options = sorted(
        data.loc[
            (data["Brand"] == brand)
            & (data["Model_Name"] == car_model),
            "Model_Variant"
        ].dropna().unique().tolist()
    )
else:
    variant_options = []

with col3:
    variant = st.selectbox(
        "Variant",
        options=variant_options,
        index=None,
        placeholder="Select variant",
        disabled=(car_model is None),
        key="car_variant"
    )


# --------------------------------------------------
# OTHER DROPDOWN OPTIONS
# --------------------------------------------------

def get_options(column):
    return sorted(data[column].dropna().unique().tolist())


st.subheader("Additional Car Details")

col1, col2, col3 = st.columns(3)

with col1:
    car_type = st.selectbox(
        "Car Type", get_options("Car_Type"),
        index=None, placeholder="Select car type"
    )

    transmission = st.selectbox(
        "Transmission", get_options("Transmission"),
        index=None, placeholder="Select transmission"
    )

with col2:
    fuel_type = st.selectbox(
        "Fuel Type", get_options("Fuel_Type"),
        index=None, placeholder="Select fuel type"
    )

    owner = st.selectbox(
        "Number of Owners", get_options("Owner"),
        index=None, placeholder="Select owner"
    )

with col3:
    state = st.selectbox(
        "Registration State", get_options("State"),
        index=None, placeholder="Select state"
    )

    accidental = st.selectbox(
        "Accidental History", get_options("Accidental"),
        index=None, placeholder="Select history"
    )


col1, col2 = st.columns(2)

with col1:
    year_options = sorted(
        data["Year"].dropna().astype(int).unique().tolist(),
        reverse=True
    )

    year = st.selectbox(
        "Manufacturing Year",
        options=year_options,
        index=None,
        placeholder="Select year"
    )

with col2:
    kilometers = st.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=1000000,
    value=50000,
    step=1000,
    help="Enter the total kilometers driven by the car."
)


# --------------------------------------------------
# PREPROCESS INPUT USING NOTEBOOK ENCODERS
# --------------------------------------------------

def prepare_input():
    input_data = pd.DataFrame([{
        "Brand": brand,
        "Model_Name": car_model,
        "Model_Variant": variant,
        "Car_Type": car_type,
        "Transmission": transmission,
        "Fuel_Type": fuel_type,
        "Year": year,
        "Kilometers": kilometers,
        "Owner": owner,
        "State": state,
        "Accidental": accidental
    }])

    # Keep the same order as the notebook's encoding steps
    transmission_value = artifacts[
        "label_encoder_transmission"
    ].transform(input_data["Transmission"])[0]

    accidental_value = artifacts[
        "label_encoder_accidental"
    ].transform(input_data["Accidental"])[0]

    encoded_parts = [
        pd.DataFrame({
            "Transmission": [transmission_value],
            "Year": [year],
            "Kilometers": [kilometers],
            "Accidental": [accidental_value]
        })
    ]

    encoder_columns = [
        ("onehot_encoder_brand", "Brand"),
        ("onehot_encoder_fuel_type", "Fuel_Type"),
        ("onehot_encoder_model_name", "Model_Name"),
        ("onehot_encoder_model_variant", "Model_Variant"),
        ("onehot_encoder_car_type", "Car_Type"),
        ("onehot_encoder_state", "State"),
        ("onehot_encoder_owner", "Owner")
    ]

    for encoder_name, column in encoder_columns:
        encoder = artifacts[encoder_name]

        values = encoder.transform(input_data[[column]])

        if hasattr(values, "toarray"):
            values = values.toarray()

        encoded_df = pd.DataFrame(
            values,
            columns=encoder.get_feature_names_out([column])
        )

        encoded_parts.append(encoded_df)

    input_encoded = pd.concat(
        encoded_parts, axis=1
    )

    # Confirm preprocessing matches the scaler's input width
    expected_features = artifacts["scaler"].n_features_in_

    if input_encoded.shape[1] != expected_features:
        raise ValueError(
            f"Feature mismatch: input has "
            f"{input_encoded.shape[1]} features, but scaler "
            f"expects {expected_features}. Retrain and save "
            f"the scaler and encoders from the same pipeline."
        )

    with open(BASE_DIR / "feature_columns.pkl", "rb") as file:
        feature_columns = pickle.load(file)

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )


    return input_encoded


# --------------------------------------------------
# PREDICT PRICE
# --------------------------------------------------

if st.button(
    "Predict Car Price",
    type="primary",
    use_container_width=True
):
    selections = [
        brand, car_model, variant, car_type,
        transmission, fuel_type, owner, state,
        accidental, year, kilometers
    ]

    if any(value is None for value in selections):
        st.warning("Please select all car details.")
    else:
        try:
            input_encoded = prepare_input()

            input_scaled = artifacts["scaler"].transform(
                input_encoded
            )

            prediction = model.predict(
                input_scaled, verbose=0
            )

            predicted_price = float(
                np.asarray(prediction).reshape(-1)[0]
            )

            st.divider()
            st.subheader("Estimated Selling Price")

            if np.isfinite(predicted_price) and predicted_price >= 0:
                st.success(
                    f"₹ {predicted_price:,.0f}"
                )
                st.caption(
                    "This is an estimated price from your trained "
                    "ANN model, not a guaranteed selling price."
                )
            else:
                st.error(
                    "The model returned an invalid price. "
                    "Check the training target and preprocessing."
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
