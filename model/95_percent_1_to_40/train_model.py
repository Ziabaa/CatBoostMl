from pathlib import Path

import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "catboost_vip_model.cbm"


def load_dataset():
    train_df = pd.read_parquet("data/model_train_dataset.parquet")
    test_df = pd.read_parquet("data/model_test_dataset.parquet")

    # Признак наличия даты установки
    for df in (train_df, test_df):
        df["has_install_date"] = (df["install_to_ftd_hours"].notna().astype(int))

    # Категориальные признаки
    cat_features = ["email_domain", "ftd_month_period"]

    for df in (train_df, test_df):
        for col in cat_features:
            df[col] = df[col].fillna("unknown").astype(str)

    # Удаляем целевую переменную из признаков
    X_train = train_df.drop(columns=["is_vip"])
    y_train = train_df["is_vip"].astype(int)

    X_test = test_df.drop(columns=["is_vip"])
    y_test = test_df["is_vip"].astype(int)

    return X_train, X_test, y_train, y_test, cat_features


def train_model():
    X_train, X_test, y_train, y_test, cat_features = load_dataset()

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    print("\nTrain target distribution:")
    print(y_train.value_counts())

    print("\nTest target distribution:")
    print(y_test.value_counts())

    # Создаём модель
    model = CatBoostClassifier(
        iterations=1500,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=100,
    )

    # Обучение
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test),
        use_best_model=True,
    )

    # Предсказания
    y_pred = model.predict(X_test).astype(int).flatten()
    y_proba = model.predict_proba(X_test)[:, 1]

    # Метрики
    print("\n========== MODEL QUALITY ==========")

    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("Precision:", precision_score(y_test, y_pred, zero_division=0))

    print("Recall:", recall_score(y_test, y_pred, zero_division=0))

    print("F1-score:", f1_score(y_test, y_pred, zero_division=0))

    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    print("PR-AUC:", average_precision_score(y_test, y_proba))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        ),
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Важность признаков
    feature_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        },
    )

    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False,
    )

    print("\nFeature importance:")
    print(feature_importance.to_string(index=False))

    # Сохраняем модель
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(MODEL_PATH)

    feature_importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    print(f"\nModel saved: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
