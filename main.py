from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

TRAIN_FILE = "./data/train.csv"
TEST_FILE = "./data/submission.csv"


def main():
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)

    df_train = data_pipeline(df_train)
    df_test = data_pipeline(df_test)

    X_train, X_val, y_train, y_val = train_test_split(
        df_train.drop("is_converted", axis=1),
        df_train["is_converted"],
        test_size=0.2,
        shuffle=True,
        random_state=503,
    )

    model = build_model()
    model.fit(X_train.fillna(0), y_train)

    pred = model.predict(X_val.fillna(0))
    evaluate(y_val, pred)

    X_test = df_test.drop(["is_converted", "id"], axis=1).fillna(0)
    test_pred = model.predict(X_test)

    df_sub = pd.read_csv(TEST_FILE)
    df_sub["is_converted"] = test_pred

    df_sub.to_csv("submission.csv", index=False)


def data_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    features_to_drop = [
        "customer_country",
        "business_subarea",
        "business_area",
        "customer_idx",
        "product_category",
        "product_subcategory",
        "product_modelname",
        "customer_country.1",
        "customer_position",
        "customer_job",
        "customer_type",
        "lead_desc_length",
        "response_corporate",
    ]

    features_to_encode = [
        "enterprise",
        "expected_timeline",
        "business_unit",
        "inquiry_type",
    ]

    df = preprocess_features(df)
    df = engineer_features(df)
    df = drop_features(df, features=features_to_drop)
    df = encode_features(df, features=features_to_encode)
    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    # Expected Timeline
    expected_timeline_columns = [
        "less than 3 months",
        "more than a year",
        "6 months ~ 9 months",
        "3 months ~ 6 months",
        "9 months ~ 1 year",
    ]

    df["expected_timeline"] = [
        (
            np.nan
            if expected_timeline not in expected_timeline_columns
            else expected_timeline
        )
        for expected_timeline in df["expected_timeline"].values
    ]

    # Inquiry Type
    inquiry_type_columns = [
        "Purchase Consultation",
        "Sales Inquiry",
        "Product Information",
        "Usage or technical consultation",
        "Other",
    ]

    inquiry_type_values = df["inquiry_type"].values
    inquiry_type_values = [
        (
            "Purchase Consultation"
            if type(x) == str and x.lower().find("quotation") != -1
            else x
        )
        for x in inquiry_type_values
    ]

    df["inquiry_type"] = [
        ("Other" if inquiry_type not in inquiry_type_columns else inquiry_type)
        for inquiry_type in inquiry_type_values
    ]

    df["inquiry_type"] = df["inquiry_type"].astype(
        pd.CategoricalDtype(categories=inquiry_type_columns)
    )

    # Business Unit
    df["business_unit"] = df["business_unit"].astype(
        pd.CategoricalDtype(categories=["ID", "AS", "IT", "Solution", "CM"])
    )

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    return df


def drop_features(df: pd.DataFrame, features: List) -> pd.DataFrame:
    return df.drop(features, axis=1)


def encode_features(df: pd.DataFrame, features: List) -> pd.DataFrame:
    df_encoded = pd.get_dummies(df[features], columns=features)
    df_encoded = df_encoded.apply(lambda x: x.astype("category").cat.codes)
    df = pd.concat([df, df_encoded], axis=1).drop(features, axis=1)
    return df


def build_model() -> Any:
    return RandomForestClassifier()


def evaluate(y_test, y_pred=None):
    confusion = confusion_matrix(y_test, y_pred, labels=[True, False])
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=[True, False])
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, labels=[True, False])

    print(f"Confusion Matrix:\n{confusion}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")


if __name__ == "__main__":
    main()
