from typing import List

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

    model = RandomForestClassifier()
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
        "expected_timeline",
        "product_category",
        "product_subcategory",
        "product_modelname",
        "customer_country.1",
        "customer_position",
        "customer_job",
        "business_unit",
        "customer_type",
        "inquiry_type",
        "response_corporate",
    ]

    features_to_encode = [
        "enterprise",
    ]

    df = drop_features(df, features=features_to_drop)
    df = encode_features(df, features=features_to_encode)
    return df


def drop_features(df: pd.DataFrame, features: List) -> pd.DataFrame:
    return df.drop(features, axis=1)


def encode_features(df: pd.DataFrame, features: List) -> pd.DataFrame:
    df_encoded = pd.get_dummies(df[features], columns=features)
    df_encoded = df_encoded.apply(lambda x: x.astype("category").cat.codes)
    df = pd.concat([df, df_encoded], axis=1).drop(features, axis=1)
    return df


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
