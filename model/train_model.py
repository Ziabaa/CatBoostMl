from pathlib import Path
import json
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix
)

# ========================= CONFIG =========================
CONFIG = {
    "experiment_name": "vip_loyal",
    "auto_class_weights": None,
    "class_weights": [1, 10],       # сильнее поднимает скоры VIP, чем [1,5]
    "negative_ratio": 20,           # часть негативов сохраняем, чтобы не терять ранжирование
    "iterations": 4000,
    "depth": 4,                    # проще модель, лучше обобщение
    "learning_rate": 0.03,
    "l2_leaf_reg": 10,
    "early_stopping_rounds": 300,
    "random_seed": 42,              # затем 43, 44 и усреднить
    "validation_size": 0.2,         # без downsampling на validation, естественное 1:40
    "threshold_rule": "fpr",     # вместо F1
    "target_fpr": 0.28,
    "vip_to_nonvip_ratio": 1 / 40,
    "population_size": 10000,
    "thresholds": [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}
# ==========================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("experiments") / CONFIG["experiment_name"]
CAT_FEATURES = ["email_domain", "ftd_month_period"]


def load_dataset():
    train_df = pd.read_parquet(DATA_DIR / "model_train_dataset.parquet")
    test_df = pd.read_parquet(DATA_DIR / "model_test_dataset.parquet")

    for df in (train_df, test_df):
        df["has_install_date"] = df["install_to_ftd_hours"].notna().astype(int)
        for col in CAT_FEATURES:
            df[col] = df[col].fillna("unknown").astype(str)

    X_train = train_df.drop(columns=["is_vip"])
    y_train = train_df["is_vip"].astype(int)
    X_test = test_df.drop(columns=["is_vip"])
    y_test = test_df["is_vip"].astype(int)

    return X_train, X_test, y_train, y_test


def downsample_negatives(X, y, ratio, seed):
    if ratio is None:
        return X, y

    if ratio <= 0:
        raise ValueError("negative_ratio должен быть положительным числом.")

    rng = np.random.default_rng(seed)
    pos_idx = y[y == 1].index.to_numpy()
    neg_idx = y[y == 0].index.to_numpy()
    max_negatives = int(len(pos_idx) * ratio)

    if max_negatives < 1:
        raise ValueError("Слишком маленький negative_ratio: не осталось не-VIP.")

    if len(neg_idx) > max_negatives:
        neg_idx = rng.choice(neg_idx, size=max_negatives, replace=False)

    selected_idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(selected_idx)
    return X.loc[selected_idx], y.loc[selected_idx]


def threshold_table(y_true, y_proba):
    rows = []
    for threshold in CONFIG["thresholds"]:
        pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        recall = tp / (tp + fn) if tp + fn else 0
        fpr = fp / (fp + tn) if fp + tn else 0
        precision = tp / (tp + fp) if tp + fp else 0
        f1 = f1_score(y_true, pred, zero_division=0)

        rows.append({
            "threshold": threshold,
            "VIP_found_per_1000_real_VIP": round(recall * 1000),
            "false_nonVIP_per_1000_real_nonVIP": round(fpr * 1000),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "predicted_VIP": tp + fp,
            "true_VIP_found": tp,
            "false_VIP": fp,
            "missed_VIP": fn,
        })

    return pd.DataFrame(rows)


def print_population_estimate(y_true, predictions):
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else 0
    fpr = fp / (fp + tn) if fp + tn else 0

    # vip_to_nonvip_ratio = 1/40 означает 1 VIP на 40 не-VIP.
    vip_share = CONFIG["vip_to_nonvip_ratio"] / (1 + CONFIG["vip_to_nonvip_ratio"])
    vip_count = CONFIG["population_size"] * vip_share
    nonvip_count = CONFIG["population_size"] - vip_count

    found_vip = vip_count * recall
    false_vip = nonvip_count * fpr
    selected = found_vip + false_vip

    print(f"\n========== ESTIMATE FOR {CONFIG['population_size']:,} CLIENTS ==========")
    print(f"Assumed VIP share: 1:{round(1 / CONFIG['vip_to_nonvip_ratio'])}")
    print(f"Expected real VIP: {vip_count:.0f}")
    print(f"Expected real non-VIP: {nonvip_count:.0f}")
    print(f"VIP found: {found_vip:.0f}")
    print(f"VIP missed: {vip_count - found_vip:.0f}")
    print(f"False VIP selections: {false_vip:.0f}")
    print(f"Total selected: {selected:.0f}")
    print(f"Precision in selected group: {found_vip / selected:.2%}" if selected else "Precision: 0%")


def train_model():
    if CONFIG["auto_class_weights"] is not None and CONFIG["class_weights"] is not None:
        raise ValueError("Укажи либо auto_class_weights, либо class_weights, но не оба параметра.")

    X_train, X_test, y_train, y_test = load_dataset()

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    print("\nTrain target distribution:\n", y_train.value_counts().sort_index())
    print("\nTest target distribution:\n", y_test.value_counts().sort_index())

    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train, y_train, test_size=CONFIG["validation_size"],
        stratify=y_train, random_state=CONFIG["random_seed"]
    )

    # Downsampling применяется только к subtrain. Validation и test не меняются.
    X_subtrain, y_subtrain = downsample_negatives(
        X_subtrain, y_subtrain, CONFIG["negative_ratio"], CONFIG["random_seed"]
    )

    print("\nSubtrain size after sampling:", len(X_subtrain))
    print("Subtrain target distribution:\n", y_subtrain.value_counts().sort_index())
    print("Validation size:", len(X_val))

    model_params = {
        "iterations": CONFIG["iterations"],
        "depth": CONFIG["depth"],
        "learning_rate": CONFIG["learning_rate"],
        "l2_leaf_reg": CONFIG["l2_leaf_reg"],
        "loss_function": "Logloss",
        "eval_metric": CONFIG.get("eval_metric", "Logloss"),
        "random_seed": CONFIG["random_seed"],
        "verbose": 100,
        "early_stopping_rounds": CONFIG["early_stopping_rounds"],
        "allow_writing_files": False,
    }

    if CONFIG["auto_class_weights"] is not None:
        model_params["auto_class_weights"] = CONFIG["auto_class_weights"]
    if CONFIG["class_weights"] is not None:
        model_params["class_weights"] = CONFIG["class_weights"]

    model = CatBoostClassifier(**model_params)
    model.fit(X_subtrain, y_subtrain, cat_features=CAT_FEATURES, eval_set=(X_val, y_val), use_best_model=True)

    val_proba = model.predict_proba(X_val)[:, 1]
    val_table = threshold_table(y_val, val_proba)
    print("\n========== VALIDATION THRESHOLD ANALYSIS ==========")
    print(val_table.to_string(index=False))

    rule = CONFIG.get("threshold_rule", "f1")

    if rule == "recall":
        # порог = (1 - target_recall)-квантиль скоров VIP на validation
        best_threshold = float(
            np.quantile(
                val_proba[y_val.values == 1],
                1 - CONFIG["target_recall"]
                )
            )
    elif rule == "fpr":
        # порог = (1 - target_fpr)-квантиль скоров не-VIP на validation
        best_threshold = float(
            np.quantile(
                val_proba[y_val.values == 0],
                1 - CONFIG["target_fpr"]
                )
            )
    else:  # f1
        best_idx = val_table["f1"].idxmax()
        best_threshold = float(val_table.loc[best_idx, "threshold"])

    val_pred = (val_proba >= best_threshold).astype(int)
    print(f"\nRule: {rule} | Selected threshold: {best_threshold:.5f}")
    print(f"Validation recall: {recall_score(y_val, val_pred):.4f}")
    print(f"Validation precision: {precision_score(y_val, val_pred, zero_division=0):.4f}")

    test_proba = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_proba >= best_threshold).astype(int)

    print("\n========== FINAL TEST QUALITY ==========")
    print("Threshold:", best_threshold)
    print("Accuracy:", accuracy_score(y_test, test_predictions))
    print("Precision:", precision_score(y_test, test_predictions, zero_division=0))
    print("Recall:", recall_score(y_test, test_predictions, zero_division=0))
    print("F1-score:", f1_score(y_test, test_predictions, zero_division=0))
    print("ROC-AUC:", roc_auc_score(y_test, test_proba))
    print("PR-AUC:", average_precision_score(y_test, test_proba))
    print("\nClassification report:\n", classification_report(y_test, test_predictions, zero_division=0))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, test_predictions))
    print_population_estimate(y_test, test_predictions)

    test_table = threshold_table(y_test, test_proba)
    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(OUTPUT_DIR / "model.cbm")
    val_table.to_csv(OUTPUT_DIR / "validation_thresholds.csv", index=False)
    test_table.to_csv(OUTPUT_DIR / "test_thresholds.csv", index=False)
    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    metrics = {
        "experiment_name": CONFIG["experiment_name"],
        "config": CONFIG,
        "best_iteration": model.get_best_iteration(),
        "selected_threshold": best_threshold,
        "accuracy": accuracy_score(y_test, test_predictions),
        "precision": precision_score(y_test, test_predictions, zero_division=0),
        "recall": recall_score(y_test, test_predictions, zero_division=0),
        "f1": f1_score(y_test, test_predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, test_proba),
        "pr_auc": average_precision_score(y_test, test_proba),
        "confusion_matrix": confusion_matrix(y_test, test_predictions).tolist(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "subtrain_size_after_sampling": len(X_subtrain),
        "validation_size": len(X_val),
    }

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    train_model()
