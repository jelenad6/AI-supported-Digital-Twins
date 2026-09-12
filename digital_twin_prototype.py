"""AI-supported Digital Twin proof of concept for the AI4I dataset.

The replay represents a simulated stream of independent operational machine
snapshots. Consecutive AI4I rows are not treated as the real temporal history
of one physical machine.
"""

import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras


# ------------------------------------------------------------
# 30. Digital Twin configuration
# ------------------------------------------------------------

DATA_PATH = "ai4i2020.csv"
MLP_MODEL_PATH = "ai4i_mlp_model.keras"
SCALER_PATH = "ai4i_scaler.joblib"
MODEL_METADATA_PATH = "ai4i_model_metadata.json"

HISTORY_PATH = "digital_twin_history.csv"
ALERT_LOG_PATH = "digital_twin_alert_log.csv"

REPLAY_DELAY = 1.0
NUMBER_OF_REPLAY_RECORDS = 20

# "sequential" is the default realistic demonstration mode.
# "failure_examples" is only a presentation filter, never model evaluation.
# "mixed_demo" mixes known normal/failure examples for visualization only.
#REPLAY_MODE = "sequential"
REPLAY_MODE = "mixed_demo"

def load_digital_twin_artifacts():
    """Load the trained MLP and its original preprocessing artefacts."""

    trained_model = keras.models.load_model(
        MLP_MODEL_PATH
    )
    fitted_scaler = joblib.load(
        SCALER_PATH
    )

    with open(
        MODEL_METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as metadata_file:
        metadata = json.load(metadata_file)

    feature_columns = metadata["feature_columns"]
    decision_threshold = float(
        metadata["decision_threshold"]
    )

    if fitted_scaler.n_features_in_ != len(feature_columns):
        raise ValueError(
            "Scaler and saved feature-column metadata are incompatible."
        )

    return (
        trained_model,
        fitted_scaler,
        feature_columns,
        decision_threshold
    )


# ------------------------------------------------------------
# 31. Digital Twin class
# ------------------------------------------------------------

class DigitalTwin:
    """Virtual machine state supported by predictions from the trained MLP."""

    def __init__(
        self,
        trained_model,
        fitted_scaler,
        feature_columns,
        decision_threshold
    ):
        self.model = trained_model
        self.scaler = fitted_scaler
        self.feature_columns = list(feature_columns)
        self.decision_threshold = float(
            decision_threshold
        )
        self.high_risk_threshold = min(
            self.decision_threshold + 0.20,
            0.90
        )

        self.replay_step = None
        self.product_type = None
        self.air_temperature = None
        self.process_temperature = None
        self.rotational_speed = None
        self.torque = None
        self.tool_wear = None
        self.actual_machine_failure = None

        self.failure_probability = None
        self.risk_state = None
        self.maintenance_recommendation = None
        self.history = []

    def update_state(self, record, replay_step):
        """Update the virtual state from one operational snapshot."""

        self.replay_step = int(replay_step)
        self.product_type = str(record["Type"])
        self.air_temperature = float(
            record["Air temperature [K]"]
        )
        self.process_temperature = float(
            record["Process temperature [K]"]
        )
        self.rotational_speed = float(
            record["Rotational speed [rpm]"]
        )
        self.torque = float(record["Torque [Nm]"])
        self.tool_wear = float(
            record["Tool wear [min]"]
        )

        if "Machine failure" in record.index:
            self.actual_machine_failure = int(
                record["Machine failure"]
            )
        else:
            self.actual_machine_failure = None

        self.predict_failure()
        self.determine_risk_state()
        self.generate_maintenance_recommendation()
        self.save_history()

    def prepare_model_input(self):
        """Apply training-time encoding, ordering and fitted scaling."""

        snapshot = pd.DataFrame([{
            "Type": self.product_type,
            "Air temperature [K]": self.air_temperature,
            "Process temperature [K]": self.process_temperature,
            "Rotational speed [rpm]": self.rotational_speed,
            "Torque [Nm]": self.torque,
            "Tool wear [min]": self.tool_wear
        }])

        snapshot = pd.get_dummies(
            snapshot,
            columns=["Type"],
            drop_first=False,
            dtype=np.float32
        )

        # Identical name normalization to the training script.
        snapshot.columns = (
            snapshot.columns
            .str.replace("[", "(", regex=False)
            .str.replace("]", ")", regex=False)
            .str.replace("<", "less_than", regex=False)
        )

        unexpected_columns = set(snapshot.columns).difference(
            self.feature_columns
        )
        if unexpected_columns:
            raise ValueError(
                "Snapshot contains an unsupported product type or feature: "
                f"{sorted(unexpected_columns)}"
            )

        snapshot = snapshot.reindex(
            columns=self.feature_columns,
            fill_value=0.0
        )

        return self.scaler.transform(
            snapshot
        ).astype(np.float32)

    def predict_failure(self):
        model_input = self.prepare_model_input()
        self.failure_probability = float(
            self.model.predict(
                model_input,
                verbose=0
            ).reshape(-1)[0]
        )
        return self.failure_probability

    def determine_risk_state(self):
        if self.failure_probability < self.decision_threshold:
            self.risk_state = "NORMAL"
        elif self.failure_probability < self.high_risk_threshold:
            self.risk_state = "ELEVATED FAILURE RISK"
        else:
            self.risk_state = "HIGH FAILURE RISK"

        return self.risk_state

    def generate_maintenance_recommendation(self):
        recommendations = {
            "NORMAL": (
                "No immediate maintenance action required."
            ),
            "ELEVATED FAILURE RISK": (
                "Additional inspection and increased monitoring are "
                "recommended."
            ),
            "HIGH FAILURE RISK": (
                "Preventive inspection and maintenance should be "
                "prioritized."
            )
        }

        # Decision support depends only on AI risk, never on the true label.
        self.maintenance_recommendation = recommendations[
            self.risk_state
        ]
        return self.maintenance_recommendation

    def save_history(self):
        self.history.append({
            "Replay step": self.replay_step,
            "Type": self.product_type,
            "Air temperature": self.air_temperature,
            "Process temperature": self.process_temperature,
            "Rotational speed": self.rotational_speed,
            "Torque": self.torque,
            "Tool wear": self.tool_wear,
            "Failure probability": self.failure_probability,
            "Risk state": self.risk_state,
            "Maintenance recommendation": (
                self.maintenance_recommendation
            ),
            "Actual Machine failure": self.actual_machine_failure
        })

    def display_state(self):
        actual_failure = (
            str(self.actual_machine_failure)
            if self.actual_machine_failure is not None
            else "Not available"
        )

        print("\n" + "=" * 60)
        print("AI-SUPPORTED DIGITAL TWIN - CURRENT MACHINE STATE")
        print("=" * 60)
        print(f"Replay step:              {self.replay_step}")
        print(f"Product type:             {self.product_type}")
        print(
            f"Air temperature:          "
            f"{self.air_temperature:.1f} K"
        )
        print(
            f"Process temperature:      "
            f"{self.process_temperature:.1f} K"
        )
        print(
            f"Rotational speed:         "
            f"{self.rotational_speed:.0f} rpm"
        )
        print(f"Torque:                   {self.torque:.1f} Nm")
        print(f"Tool wear:                {self.tool_wear:.0f} min")
        print(
            f"\nAI failure probability:   "
            f"{self.failure_probability:.4f}"
        )
        print(
            f"Decision threshold:       "
            f"{self.decision_threshold:.4f}"
        )
        print(
            f"High-risk threshold:      "
            f"{self.high_risk_threshold:.4f}"
        )
        print(f"\nAI predicted risk state:  {self.risk_state}")
        print("\nMaintenance recommendation (decision support):")
        print(self.maintenance_recommendation)
        print(f"\nActual Machine failure:   {actual_failure}")
        print("=" * 60)


# ------------------------------------------------------------
# 32. Digital Twin preprocessing and inference
# ------------------------------------------------------------

def create_digital_twin():
    (
        trained_model,
        fitted_scaler,
        feature_columns,
        decision_threshold
    ) = load_digital_twin_artifacts()

    digital_twin = DigitalTwin(
        trained_model=trained_model,
        fitted_scaler=fitted_scaler,
        feature_columns=feature_columns,
        decision_threshold=decision_threshold
    )

    print("\nLoaded the trained AI4I MLP and preprocessing artefacts.")
    print(
        f"Decision threshold:  "
        f"{digital_twin.decision_threshold:.4f}"
    )
    print(
        f"High-risk threshold: "
        f"{digital_twin.high_risk_threshold:.4f}"
    )

    return digital_twin


# ------------------------------------------------------------
# 33. Simulated operational data replay
# ------------------------------------------------------------

def select_replay_records(dataset):
    if REPLAY_MODE == "sequential":
        replay_records = dataset.head(
            NUMBER_OF_REPLAY_RECORDS
        )
    elif REPLAY_MODE == "failure_examples":
        print(
            "\nDEMONSTRATION FILTER ACTIVE: showing records with "
            "Machine failure = 1. This mode is not model evaluation."
        )
        replay_records = (
            dataset.loc[dataset["Machine failure"] == 1]
            .head(NUMBER_OF_REPLAY_RECORDS)
        )
    elif REPLAY_MODE == "mixed_demo":
        print(
            "\nDEMONSTRATION MODE: mixed normal/failure examples "
            "selected using known labels. This mode is for visualization "
            "only and is not used for model evaluation."
        )

        normal_examples = (
            dataset.loc[dataset["Machine failure"] == 0]
            .sample(n=10, random_state=42)
        )
        failure_examples = (
            dataset.loc[dataset["Machine failure"] == 1]
            .sample(n=10, random_state=42)
        )

        replay_records = (
            pd.concat(
                [normal_examples, failure_examples],
                ignore_index=True
            )
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
    else:
        raise ValueError(
            "REPLAY_MODE must be 'sequential', 'failure_examples' "
            "or 'mixed_demo'."
        )

    if replay_records.empty:
        raise ValueError(
            "No records are available for the selected replay mode."
        )

    return replay_records


def run_digital_twin_replay(digital_twin, dataset):
    """Replay independent AI4I snapshots as a simulated real-time stream."""

    print(
        "\nSIMULATED REAL-TIME REPLAY OF OPERATIONAL MACHINE SNAPSHOTS"
    )
    print(
        "Consecutive rows are not the real temporal evolution of one "
        "physical machine."
    )
    print(f"Replay mode: {REPLAY_MODE}")

    replay_records = select_replay_records(
        dataset
    )

    for replay_step, (_, record) in enumerate(
        replay_records.iterrows(),
        start=1
    ):
        digital_twin.update_state(
            record,
            replay_step
        )
        digital_twin.display_state()

        if replay_step < len(replay_records):
            time.sleep(REPLAY_DELAY)


# ------------------------------------------------------------
# 34. Digital Twin history and alert log
# ------------------------------------------------------------

def save_replay_outputs(digital_twin):
    history = pd.DataFrame(
        digital_twin.history
    )
    history.to_csv(
        HISTORY_PATH,
        index=False
    )

    alert_columns = [
        "Replay step",
        "Failure probability",
        "Risk state",
        "Maintenance recommendation",
        "Actual Machine failure"
    ]

    alert_log = history.loc[
        history["Risk state"] != "NORMAL",
        alert_columns
    ].copy()

    alert_log.to_csv(
        ALERT_LOG_PATH,
        index=False
    )

    print("\nDIGITAL TWIN STATE HISTORY")
    print(history.round(4).to_string(index=False))
    print(f"\nHistory saved to: {HISTORY_PATH}")
    print(f"Alert log saved to: {ALERT_LOG_PATH}")

    return history, alert_log


# ------------------------------------------------------------
# 35. Risk history visualization
# ------------------------------------------------------------

def plot_risk_history(
    history,
    decision_threshold,
    high_risk_threshold
):
    plt.figure(figsize=(12, 6))
    plt.plot(
        history["Replay step"],
        history["Failure probability"],
        marker="o",
        linewidth=2,
        label="AI failure probability"
    )
    plt.axhline(
        decision_threshold,
        color="orange",
        linestyle="--",
        label=f"Decision threshold ({decision_threshold:.4f})"
    )
    plt.axhline(
        high_risk_threshold,
        color="red",
        linestyle="--",
        label=f"High-risk threshold ({high_risk_threshold:.4f})"
    )
    plt.xlabel("Replay step")
    plt.ylabel("AI failure probability")
    plt.title(
        "Digital Twin - AI Failure Risk During Simulated Data Replay"
    )
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    dataset = pd.read_csv(
        DATA_PATH
    )
    digital_twin = create_digital_twin()

    run_digital_twin_replay(
        digital_twin,
        dataset
    )

    history, _ = save_replay_outputs(
        digital_twin
    )

    plot_risk_history(
        history,
        digital_twin.decision_threshold,
        digital_twin.high_risk_threshold
    )


if __name__ == "__main__":
    main()
