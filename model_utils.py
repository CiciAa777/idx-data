import joblib
import numpy as np
import pandas as pd

# California major centers matching Week 6 feature engineering
MAJOR_CENTERS_CA = {
    "Downtown_LA": (34.0407, -118.2468),
    "Downtown_San_Diego": (32.7157, -117.1611),
    "Downtown_San_Jose": (37.3382, -121.8863),
    "Downtown_Sacramento": (38.5816, -121.4944),
}

# The exact feature order expected by your trained model pipeline
EXACT_COLUMNS = [
    "ViewYN",
    "PoolPrivateYN",
    "ListingKey",
    "Latitude",
    "Longitude",
    "PropertyType",
    "LivingArea",
    "ListingKeyNumeric",
    "MlsStatus",
    "AttachedGarageYN",
    "ParkingTotal",
    "LotSizeAcres",
    "YearBuilt",
    "StreetNumberNumeric",
    "BathroomsTotalInteger",
    "BedroomsTotal",
    "StateOrProvince",
    "FireplaceYN",
    "Stories",
    "LotSizeArea",
    "MainLevelBedrooms",
    "NewConstructionYN",
    "GarageSpaces",
    "AssociationFee",
    "LotSizeSquareFeet",
    "Year",
    "Month",
    "dist_km_to_nearest_major_center",
    "month_sin",
    "month_cos",
    "property_age_at_sale",
    "zip_median_price_per_sqft",
    "Flooring_target_enc",
    "ListAgentEmail_target_enc",
    "CloseDate_target_enc",
    "ListAgentFirstName_target_enc",
    "ListAgentLastName_target_enc",
    "UnparsedAddress_target_enc",
    "ListOfficeName_target_enc",
    "BuyerOfficeName_target_enc",
    "ListAgentFullName_target_enc",
    "BuyerAgentMlsId_target_enc",
    "BuyerAgentFirstName_target_enc",
    "BuyerAgentLastName_target_enc",
    "MLSAreaMajor_target_enc",
    "CountyOrParish_target_enc",
    "PropertySubType_target_enc",
    "BuyerOfficeAOR_target_enc",
    "ListingId_target_enc",
    "City_target_enc",
    "ContractStatusChangeDate_target_enc",
    "ListingContractDate_target_enc",
    "Levels_target_enc",
    "HighSchoolDistrict_target_enc",
    "PostalCode_target_enc",
    "BuyerAgentAOR_target_enc",
    "ListAgentAOR_target_enc",
    "geo_cluster_target_enc",
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculates great-circle distance between coordinates in kilometers."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


def prepare_input_dataframe(
    living_area: float,
    lot_size: float,
    bedrooms: int,
    bathrooms: float,
    year_built: int,
    latitude: float = 32.7157,
    longitude: float = -117.1611,
    property_type: str = "Residential",
    stories: int = 1,
    garage_spaces: int = 2,
    sale_month: int = 8,
    sale_year: int = 2026,
) -> pd.DataFrame:
    """Constructs a single-row DataFrame matching the exact 58 features expected by X_train."""

    # 1. Distance to nearest major CA employment center
    dist_cols = []
    for name, (clat, clon) in MAJOR_CENTERS_CA.items():
        d = haversine_km(latitude, longitude, clat, clon)
        dist_cols.append(d)
    min_dist_km = min(dist_cols)

    # 2. Build dictionary using string representations ("True"/"False") for booleans
    row = {
        # --- User-specified inputs ---
        "LivingArea": float(living_area),
        "LotSizeArea": float(lot_size),
        "LotSizeSquareFeet": float(lot_size),
        "LotSizeAcres": float(lot_size / 43560.0),
        "BedroomsTotal": int(bedrooms),
        "BathroomsTotalInteger": int(bathrooms),
        "YearBuilt": int(year_built),
        "Latitude": float(latitude),
        "Longitude": float(longitude),
        "PropertyType": str(property_type),
        "Stories": int(stories),
        "GarageSpaces": int(garage_spaces),
        "Year": int(sale_year),
        "Month": int(sale_month),
        # --- Engineered features ---
        "dist_km_to_nearest_major_center": float(min_dist_km),
        "month_sin": float(np.sin(2 * np.pi * sale_month / 12)),
        "month_cos": float(np.cos(2 * np.pi * sale_month / 12)),
        "property_age_at_sale": int(max(0, sale_year - year_built)),
        "zip_median_price_per_sqft": 450.0,
        # --- Categoricals formatted strictly as strings ---
        "ViewYN": "False",
        "PoolPrivateYN": "False",
        "AttachedGarageYN": "True",
        "FireplaceYN": "False",
        "NewConstructionYN": "False",
        "MlsStatus": "Closed",
        "StateOrProvince": "CA",
        "ListingKey": "1000",
        "ListingKeyNumeric": 1000,
        "StreetNumberNumeric": 100,
        "ParkingTotal": int(garage_spaces),
        "MainLevelBedrooms": int(bedrooms),
        "AssociationFee": 0.0,
    }

    # 3. Fill target-encoded high-cardinality fields with float placeholders
    target_enc_cols = [c for c in EXACT_COLUMNS if c.endswith("_target_enc")]
    for col in target_enc_cols:
        row[col] = 0.0

    # 4. Construct DataFrame and enforce column order
    df = pd.DataFrame([row])

    for col in EXACT_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    return df[EXACT_COLUMNS]


def load_model(model_path="data/processed/best_gb_model.joblib"):
    """Loads saved model pipeline via joblib."""
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load model at {model_path}. Ensure training script has executed. Error: {e}"
        )