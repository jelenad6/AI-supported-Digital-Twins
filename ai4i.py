import os
import random
import warnings
import json
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. Podešavanja
# ------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = "ai4i2020.csv"

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001


# ------------------------------------------------------------
# 2. Učitavanje podataka
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)

print(
    "\nMachine failure distribution:"
)

print(
    df["Machine failure"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 3. Izbor karakteristika
# ------------------------------------------------------------

TARGET = "Machine failure"

DROP_COLUMNS = [
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]

X = df.drop(
    columns=DROP_COLUMNS
).copy()

y = df[TARGET].astype(int).copy()


# ------------------------------------------------------------
# 4. One-hot encoding za Type
# ------------------------------------------------------------

X = pd.get_dummies(
    X,
    columns=["Type"],
    drop_first=False,
    dtype=np.float32
)


X.columns = (
    X.columns
    .str.replace("[", "(", regex=False)
    .str.replace("]", ")", regex=False)
    .str.replace("<", "less_than", regex=False)
)

if not X.columns.is_unique:
    raise ValueError(
        "Normalizacija naziva karakteristika napravila je duplikate."
    )

print(
    "\nUlazne karakteristike:"
)

print(
    X.columns.tolist()
)

print(
    "\nBroj karakteristika:",
    X.shape[1]
)


# ------------------------------------------------------------
# 5. Train / Validation / Test = 70 / 15 / 15
# ------------------------------------------------------------

X_train, X_temp, y_train, y_temp = (
    train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=SEED,
        stratify=y
    )
)

X_validation, X_test, y_validation, y_test = (
    train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=y_temp
    )
)


print(
    "\nTrain:",
    len(X_train)
)

print(
    "Validation:",
    len(X_validation)
)

print(
    "Test:",
    len(X_test)
)


print(
    "\nTrain failures:",
    y_train.sum()
)

print(
    "Validation failures:",
    y_validation.sum()
)

print(
    "Test failures:",
    y_test.sum()
)


# ------------------------------------------------------------
# 6. Skaliranje
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
).astype(np.float32)

X_validation_scaled = scaler.transform(
    X_validation
).astype(np.float32)

X_test_scaled = scaler.transform(
    X_test
).astype(np.float32)


y_train_array = (
    y_train.to_numpy(
        dtype=np.float32
    )
)

y_validation_array = (
    y_validation.to_numpy(
        dtype=np.float32
    )
)

y_test_array = (
    y_test.to_numpy(
        dtype=np.float32
    )
)


# ------------------------------------------------------------
# 7. Class weights
# ------------------------------------------------------------

CLASS_WEIGHTS = {
    0: 1.0,
    1: 5.0
}

print(
    "\nClass weights:",
    CLASS_WEIGHTS
)


# ------------------------------------------------------------
# 8. MLP model
# ------------------------------------------------------------

model = keras.Sequential([
    layers.Input(
        shape=(
            X_train_scaled.shape[1],
        )
    ),

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Dropout(
        0.20
    ),

    layers.Dense(
        32,
        activation="relu"
    ),

    layers.Dropout(
        0.20
    ),

    layers.Dense(
        16,
        activation="relu"
    ),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="binary_crossentropy",

    metrics=[
        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        keras.metrics.Precision(
            name="precision"
        ),

        keras.metrics.Recall(
            name="recall"
        ),

        keras.metrics.AUC(
            name="roc_auc",
            curve="ROC"
        ),

        keras.metrics.AUC(
            name="pr_auc",
            curve="PR"
        )
    ]
)

model.summary()


# ------------------------------------------------------------
# 9. Callbacks
# ------------------------------------------------------------

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=12,
        restore_best_weights=True
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_pr_auc",
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
]


# ------------------------------------------------------------
# 10. Treniranje
# ------------------------------------------------------------

history = model.fit(
    X_train_scaled,
    y_train_array,

    validation_data=(
        X_validation_scaled,
        y_validation_array
    ),

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,

    class_weight=CLASS_WEIGHTS,

    callbacks=callbacks,
    verbose=2
)


# ------------------------------------------------------------
# 11. Validation probabilities
# ------------------------------------------------------------

validation_probabilities = (
    model.predict(
        X_validation_scaled,
        batch_size=BATCH_SIZE,
        verbose=0
    )
    .reshape(-1)
)


# ------------------------------------------------------------
# 12. Threshold tuning na VALIDATION skupu
# ------------------------------------------------------------

threshold_candidates = np.arange(
    0.05,
    0.96,
    0.01
)

threshold_results = []

for threshold in threshold_candidates:

    predictions = (
        validation_probabilities
        >= threshold
    ).astype(int)

    threshold_results.append({
        "threshold": threshold,

        "accuracy": accuracy_score(
            y_validation_array,
            predictions
        ),

        "precision": precision_score(
            y_validation_array,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_validation_array,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_validation_array,
            predictions,
            zero_division=0
        )
    })


threshold_results = pd.DataFrame(
    threshold_results
)


best_threshold_row = (
    threshold_results
    .sort_values(
        [
            "f1",
            "recall"
        ],
        ascending=False
    )
    .iloc[0]
)


BEST_THRESHOLD = float(
    best_threshold_row[
        "threshold"
    ]
)


print(
    f"\nBest validation threshold: "
    f"{BEST_THRESHOLD:.2f}"
)

print(
    f"Validation precision: "
    f"{best_threshold_row['precision']:.4f}"
)

print(
    f"Validation recall: "
    f"{best_threshold_row['recall']:.4f}"
)

print(
    f"Validation F1: "
    f"{best_threshold_row['f1']:.4f}"
)


# ------------------------------------------------------------
# 13. MLP test predictions
# ------------------------------------------------------------

test_probabilities = (
    model.predict(
        X_test_scaled,
        batch_size=BATCH_SIZE,
        verbose=0
    )
    .reshape(-1)
)


y_pred_mlp = (
    test_probabilities
    >= BEST_THRESHOLD
).astype(int)


# ------------------------------------------------------------
# 14. Funkcija za evaluaciju
# ------------------------------------------------------------

def evaluate_model(
    model_name,
    y_true,
    y_pred,
    probabilities
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )


    print(
        f"\n{'=' * 50}"
    )

    print(
        model_name
    )

    print(
        f"{'=' * 50}"
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1-score:  {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:    {pr_auc:.4f}"
    )


    print(
        "\nClassification report:\n"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "NORMAL",
                "FAILURE"
            ],
            digits=4,
            zero_division=0
        )
    )


    confusion = confusion_matrix(
        y_true,
        y_pred
    )

    print(
        "Confusion matrix:"
    )

    print(
        confusion
    )


    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    }


# ------------------------------------------------------------
# 15. Evaluacija MLP-a
# ------------------------------------------------------------

mlp_results = evaluate_model(
    model_name="Deep Neural Network (MLP)",
    y_true=y_test_array,
    y_pred=y_pred_mlp,
    probabilities=test_probabilities
)


# ------------------------------------------------------------
# 16. Confusion Matrix - MLP
# ------------------------------------------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test_array,
    y_pred_mlp,
    display_labels=[
        "NORMAL",
        "FAILURE"
    ],
    values_format="d"
)

plt.title(
    "Confusion Matrix – MLP"
)

plt.show()


# ------------------------------------------------------------
# 17. ROC Curve - MLP
# ------------------------------------------------------------

RocCurveDisplay.from_predictions(
    y_test_array,
    test_probabilities
)

plt.title(
    "ROC Curve – MLP"
)

plt.show()


# ------------------------------------------------------------
# 18. Precision-Recall Curve - MLP
# ------------------------------------------------------------

PrecisionRecallDisplay.from_predictions(
    y_test_array,
    test_probabilities
)

plt.title(
    "Precision-Recall Curve – MLP"
)

plt.show()


# ============================================================
# RANDOM FOREST BENCHMARK
# ============================================================


# ------------------------------------------------------------
# 19. Random Forest
# ------------------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=500,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=SEED,
    n_jobs=-1
)

random_forest.fit(
    X_train_scaled,
    y_train
)


# ------------------------------------------------------------
# 20. RF threshold tuning na validation skupu
# ------------------------------------------------------------

rf_validation_probabilities = (
    random_forest.predict_proba(
        X_validation_scaled
    )[:, 1]
)


rf_threshold_results = []

for threshold in threshold_candidates:

    predictions = (
        rf_validation_probabilities
        >= threshold
    ).astype(int)

    rf_threshold_results.append({
        "threshold": threshold,

        "f1": f1_score(
            y_validation_array,
            predictions,
            zero_division=0
        ),

        "precision": precision_score(
            y_validation_array,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_validation_array,
            predictions,
            zero_division=0
        )
    })


rf_threshold_results = pd.DataFrame(
    rf_threshold_results
)


best_rf_threshold_row = (
    rf_threshold_results
    .sort_values(
        [
            "f1",
            "recall"
        ],
        ascending=False
    )
    .iloc[0]
)


BEST_RF_THRESHOLD = float(
    best_rf_threshold_row[
        "threshold"
    ]
)


print(
    f"\nBest Random Forest validation threshold: "
    f"{BEST_RF_THRESHOLD:.2f}"
)


# ------------------------------------------------------------
# 21. Random Forest test
# ------------------------------------------------------------

rf_test_probabilities = (
    random_forest.predict_proba(
        X_test_scaled
    )[:, 1]
)


y_pred_rf = (
    rf_test_probabilities
    >= BEST_RF_THRESHOLD
).astype(int)


rf_results = evaluate_model(
    model_name="Random Forest",
    y_true=y_test_array,
    y_pred=y_pred_rf,
    probabilities=rf_test_probabilities
)

# ============================================================
# XGBOOST BENCHMARK
# ============================================================


# ------------------------------------------------------------
# 22. XGBoost model
# ------------------------------------------------------------

number_of_negative_samples = int(
    (y_train == 0).sum()
)

number_of_positive_samples = int(
    (y_train == 1).sum()
)

scale_pos_weight = (
    number_of_negative_samples
    / number_of_positive_samples
)

print(
    "\nXGBoost scale_pos_weight:",
    round(scale_pos_weight, 4)
)


xgboost_model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,

    subsample=0.80,
    colsample_bytree=0.80,

    min_child_weight=2,

    objective="binary:logistic",
    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=SEED,
    n_jobs=-1
)


xgboost_model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# 23. XGBoost threshold tuning na validation skupu
# ------------------------------------------------------------

xgb_validation_probabilities = (
    xgboost_model.predict_proba(
        X_validation
    )[:, 1]
)


xgb_threshold_results = []

for threshold in threshold_candidates:

    predictions = (
        xgb_validation_probabilities
        >= threshold
    ).astype(int)

    xgb_threshold_results.append({
        "threshold": threshold,

        "accuracy": accuracy_score(
            y_validation_array,
            predictions
        ),

        "precision": precision_score(
            y_validation_array,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_validation_array,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_validation_array,
            predictions,
            zero_division=0
        )
    })


xgb_threshold_results = pd.DataFrame(
    xgb_threshold_results
)


best_xgb_threshold_row = (
    xgb_threshold_results
    .sort_values(
        [
            "f1",
            "recall"
        ],
        ascending=False
    )
    .iloc[0]
)


BEST_XGB_THRESHOLD = float(
    best_xgb_threshold_row[
        "threshold"
    ]
)


print(
    f"\nBest XGBoost validation threshold: "
    f"{BEST_XGB_THRESHOLD:.2f}"
)

print(
    f"Validation precision: "
    f"{best_xgb_threshold_row['precision']:.4f}"
)

print(
    f"Validation recall: "
    f"{best_xgb_threshold_row['recall']:.4f}"
)

print(
    f"Validation F1: "
    f"{best_xgb_threshold_row['f1']:.4f}"
)


# ------------------------------------------------------------
# 24. XGBoost test
# ------------------------------------------------------------

xgb_test_probabilities = (
    xgboost_model.predict_proba(
        X_test
    )[:, 1]
)


y_pred_xgb = (
    xgb_test_probabilities
    >= BEST_XGB_THRESHOLD
).astype(int)


xgb_results = evaluate_model(
    model_name="XGBoost",
    y_true=y_test_array,
    y_pred=y_pred_xgb,
    probabilities=xgb_test_probabilities
)


# ------------------------------------------------------------
# 25. Finalno poređenje
# ------------------------------------------------------------

comparison = pd.DataFrame([
    mlp_results,
    rf_results,
    xgb_results
])

print(
    "\nFINAL MODEL COMPARISON"
)

print(
    comparison
    .set_index("Model")
    .round(4)
)


# ------------------------------------------------------------
# 26. Cuvanje MLP artefakata za Digital Twin prototip
# ------------------------------------------------------------

MLP_MODEL_PATH = "ai4i_mlp_model.keras"
SCALER_PATH = "ai4i_scaler.joblib"
MODEL_METADATA_PATH = "ai4i_model_metadata.json"

model.save(
    MLP_MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)

model_metadata = {
    "feature_columns": X.columns.tolist(),
    "decision_threshold": float(BEST_THRESHOLD),
    "target": TARGET,
    "model_description": "AI4I MLP from the 70/15/15 experiment"
}

with open(
    MODEL_METADATA_PATH,
    "w",
    encoding="utf-8"
) as metadata_file:
    json.dump(
        model_metadata,
        metadata_file,
        indent=4,
        ensure_ascii=False
    )

print(
    "\nDigital Twin artefacts saved:"
)
print(f"- {MLP_MODEL_PATH}")
print(f"- {SCALER_PATH}")
print(f"- {MODEL_METADATA_PATH}")


# ------------------------------------------------------------
# 27. 5-fold Stratified Cross-Validation
# ------------------------------------------------------------

cross_validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

cv_threshold_candidates = np.arange(
    0.05,
    0.96,
    0.01
)

cv_fold_results = []


def select_cv_threshold(y_true, probabilities):
    """Izbor praga samo na unutrasnjem validation skupu."""

    threshold_scores = []

    for threshold in cv_threshold_candidates:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        threshold_scores.append({
            "Threshold": float(threshold),
            "F1": f1_score(
                y_true,
                predictions,
                zero_division=0
            ),
            "Recall": recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        })

    best_result = max(
        threshold_scores,
        key=lambda result: (
            result["F1"],
            result["Recall"]
        )
    )

    return best_result["Threshold"]


def save_and_print_cv_result(
    model_name,
    fold_number,
    threshold,
    y_true,
    probabilities
):
    """Evaluacija na potpuno nevidjenom spoljasnjem test foldu."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    result = {
        "Model": model_name,
        "Fold": fold_number,
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

    cv_fold_results.append(result)

    print(
        f"\nFold {fold_number}/5 - {model_name}"
    )
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(
        f"TN / FP / FN / TP: "
        f"{tn} / {fp} / {fn} / {tp}"
    )


# ------------------------------------------------------------
# 28. Treniranje i evaluacija po spoljasnjim foldovima
# ------------------------------------------------------------

for fold_number, (
    outer_train_indices,
    outer_test_indices
) in enumerate(
    cross_validation.split(X, y),
    start=1
):
    print(
        f"\n{'=' * 60}\n"
        f"CROSS-VALIDATION FOLD {fold_number}/5\n"
        f"{'=' * 60}"
    )

    X_outer_train = X.iloc[
        outer_train_indices
    ].copy()
    y_outer_train = y.iloc[
        outer_train_indices
    ].copy()

    X_outer_test = X.iloc[
        outer_test_indices
    ].copy()
    y_outer_test = y.iloc[
        outer_test_indices
    ].copy()

    (
        X_inner_train,
        X_inner_validation,
        y_inner_train,
        y_inner_validation
    ) = train_test_split(
        X_outer_train,
        y_outer_train,
        test_size=0.15,
        stratify=y_outer_train,
        random_state=SEED
    )

    # Scaler se fituje iskljucivo na unutrasnjem training skupu.
    fold_scaler = StandardScaler()

    X_inner_train_scaled = fold_scaler.fit_transform(
        X_inner_train
    ).astype(np.float32)

    X_inner_validation_scaled = fold_scaler.transform(
        X_inner_validation
    ).astype(np.float32)

    X_outer_test_scaled = fold_scaler.transform(
        X_outer_test
    ).astype(np.float32)

    y_inner_train_array = y_inner_train.to_numpy(
        dtype=np.float32
    )
    y_inner_validation_array = y_inner_validation.to_numpy(
        dtype=np.float32
    )
    y_outer_test_array = y_outer_test.to_numpy(
        dtype=np.float32
    )

    # Potpuno nov MLP i novi skup tezina za svaki fold.
    keras.backend.clear_session()
    tf.random.set_seed(SEED)

    cv_mlp_model = keras.Sequential([
        layers.Input(
            shape=(X_inner_train_scaled.shape[1],)
        ),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.20),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.20),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    cv_mlp_model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            keras.metrics.Precision(
                name="precision"
            ),
            keras.metrics.Recall(
                name="recall"
            ),
            keras.metrics.AUC(
                name="roc_auc",
                curve="ROC"
            ),
            keras.metrics.AUC(
                name="pr_auc",
                curve="PR"
            )
        ]
    )

    cv_mlp_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=12,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc",
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    cv_mlp_model.fit(
        X_inner_train_scaled,
        y_inner_train_array,
        validation_data=(
            X_inner_validation_scaled,
            y_inner_validation_array
        ),
        epochs=100,
        batch_size=128,
        class_weight={0: 1.0, 1: 5.0},
        callbacks=cv_mlp_callbacks,
        verbose=2
    )

    cv_mlp_validation_probabilities = (
        cv_mlp_model.predict(
            X_inner_validation_scaled,
            batch_size=128,
            verbose=0
        ).reshape(-1)
    )

    cv_mlp_threshold = select_cv_threshold(
        y_inner_validation_array,
        cv_mlp_validation_probabilities
    )

    cv_mlp_test_probabilities = (
        cv_mlp_model.predict(
            X_outer_test_scaled,
            batch_size=128,
            verbose=0
        ).reshape(-1)
    )

    save_and_print_cv_result(
        "Deep Neural Network (MLP)",
        fold_number,
        cv_mlp_threshold,
        y_outer_test_array,
        cv_mlp_test_probabilities
    )


    cv_random_forest = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    cv_random_forest.fit(
        X_inner_train,
        y_inner_train
    )

    cv_rf_validation_probabilities = (
        cv_random_forest.predict_proba(
            X_inner_validation
        )[:, 1]
    )

    cv_rf_threshold = select_cv_threshold(
        y_inner_validation,
        cv_rf_validation_probabilities
    )

    cv_rf_test_probabilities = (
        cv_random_forest.predict_proba(
            X_outer_test
        )[:, 1]
    )

    save_and_print_cv_result(
        "Random Forest",
        fold_number,
        cv_rf_threshold,
        y_outer_test,
        cv_rf_test_probabilities
    )


    cv_number_negative = int(
        (y_inner_train == 0).sum()
    )
    cv_number_positive = int(
        (y_inner_train == 1).sum()
    )
    cv_scale_pos_weight = (
        cv_number_negative / cv_number_positive
    )

    cv_xgboost_model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=cv_scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    cv_xgboost_model.fit(
        X_inner_train,
        y_inner_train
    )

    cv_xgb_validation_probabilities = (
        cv_xgboost_model.predict_proba(
            X_inner_validation
        )[:, 1]
    )

    cv_xgb_threshold = select_cv_threshold(
        y_inner_validation,
        cv_xgb_validation_probabilities
    )

    cv_xgb_test_probabilities = (
        cv_xgboost_model.predict_proba(
            X_outer_test
        )[:, 1]
    )

    save_and_print_cv_result(
        "XGBoost",
        fold_number,
        cv_xgb_threshold,
        y_outer_test,
        cv_xgb_test_probabilities
    )


# ------------------------------------------------------------
# 29. Rezultati po foldu i zavrsni CV pregled
# ------------------------------------------------------------

cv_results = pd.DataFrame(
    cv_fold_results
)

cv_compact_columns = [
    "Model",
    "Fold",
    "Threshold",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "PR-AUC",
    "TN",
    "FP",
    "FN",
    "TP"
]

print(
    "\n5-FOLD CROSS-VALIDATION RESULTS BY FOLD"
)
print(
    cv_results[
        cv_compact_columns
    ].round(4).to_string(index=False)
)

cv_summary = (
    cv_results
    .groupby("Model", sort=False)
    .agg(
        **{
            "Accuracy Mean": ("Accuracy", "mean"),
            "Accuracy Std": ("Accuracy", "std"),
            "Precision Mean": ("Precision", "mean"),
            "Precision Std": ("Precision", "std"),
            "Recall Mean": ("Recall", "mean"),
            "Recall Std": ("Recall", "std"),
            "F1 Mean": ("F1", "mean"),
            "F1 Std": ("F1", "std"),
            "ROC-AUC Mean": ("ROC-AUC", "mean"),
            "ROC-AUC Std": ("ROC-AUC", "std"),
            "PR-AUC Mean": ("PR-AUC", "mean"),
            "PR-AUC Std": ("PR-AUC", "std"),
            "Threshold Mean": ("Threshold", "mean"),
            "Threshold Std": ("Threshold", "std")
        }
    )
    .reset_index()
)

print(
    "\n5-FOLD STRATIFIED CROSS-VALIDATION SUMMARY"
)
print(
    cv_summary.round(4).to_string(index=False)
)
