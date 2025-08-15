import re
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, fbeta_score

warnings.filterwarnings("ignore")


class LightGBMTrainer:
    def __init__(self, drop_cols=None, smote_k=5):
        self.drop_cols = drop_cols or []
        self.smote_k = smote_k
        self.model = None
        self.feature_columns = None

    def _preprocess(self, df):
        if df is None or df.empty:
            raise ValueError("Training dataframe is empty.")
        if "Label" not in df.columns:
            raise ValueError("Missing 'Label' column in dataset.")

        # Drop chỉ các cột thực sự tồn tại
        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns])

        # Map label -> {0,1}
        label_map = {"False": 0, "True": 1, False: 0, True: 1}
        df["Label"] = df["Label"].map(label_map)
        if df["Label"].isnull().any():
            bad = df.loc[df["Label"].isnull(), "Label"]
            raise ValueError(
                f"Invalid Label values that cannot be mapped to 0/1. Examples: {bad.head(3).tolist()}"
            )
        df["Label"] = df["Label"].astype(int)
        return df

    def _prepare_features(self, df, is_train=False):
        # Bóc y, one-hot các feature
        if "Label" not in df.columns:
            raise ValueError("Missing 'Label' column after preprocessing.")
        X_raw = df.drop(columns=["Label"])
        if X_raw.shape[1] == 0:
            raise ValueError(
                "No feature columns left after dropping; check drop_cols and input CSVs."
            )

        X = pd.get_dummies(X_raw, drop_first=True).astype(float)
        if X.shape[1] == 0:
            raise ValueError(
                "No features after get_dummies; check categorical columns."
            )
        y = df["Label"]

        if is_train:
            self.feature_columns = X.columns
        else:
            if self.feature_columns is None:
                raise RuntimeError("feature_columns is not set. Call fit() first.")
            X = X.reindex(columns=self.feature_columns, fill_value=0)

        # Tên cột “sạch” cho LightGBM
        X.columns = [re.sub(r"[^\w]", "_", col) for col in X.columns]
        return X, y

    def fit(self, train_file):
        df = pd.read_csv(train_file)
        if df is None or df.empty:
            raise ValueError(
                f"Training file '{train_file}' is empty or cannot be read."
            )
        df = self._preprocess(df)
        X, y = self._prepare_features(df, is_train=True)

        # Lưu lại objective để dùng ở evaluate
        self.objective = "multiclass"

        cls_counts = y.value_counts()
        X_resampled, y_resampled = X, y

        if len(cls_counts) >= 2:
            minority_count = int(cls_counts.min())
            k = min(self.smote_k, max(1, minority_count - 1))
            if minority_count >= 2 and k >= 1:
                try:
                    smote = SMOTE(k_neighbors=k, random_state=42)
                    X_resampled, y_resampled = smote.fit_resample(X, y)
                except Exception as e:
                    print(f"SMOTE skipped due to: {e}. Proceeding without resampling.")

            params = {
                "objective": "multiclass",
                "metric": "multi_logloss",
                "num_classes": 2,
                "learning_rate": 0.05,
                "device": "gpu",  # sẽ fallback nếu GPU không có
                "verbosity": -1,
                "boosting_type": "gbdt",
                "num_leaves": 200,
                "feature_fraction": 0.4,
                "max_depth": 45,
            }
        else:
            # Chỉ 1 class -> chuyển binary
            print(
                f"[WARNING] Training set has a single class {cls_counts.to_dict()}. Switching to binary objective."
            )
            self.objective = "binary"
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": 0.05,
                "device": "gpu",  # sẽ fallback nếu GPU không có
                "verbosity": -1,
                "boosting_type": "gbdt",
                "num_leaves": 200,
                "feature_fraction": 0.4,
                "max_depth": 45,
            }

        train_data = lgb.Dataset(X_resampled, label=y_resampled)

        # Try GPU -> fallback CPU nếu không có OpenCL/GPU
        try:
            self.model = lgb.train(params, train_data, num_boost_round=250)
        except lgb.basic.LightGBMError as e:
            msg = str(e)
            if "No OpenCL device found" in msg or "GPU" in msg or "OpenCL" in msg:
                print("[WARNING] No GPU/OpenCL available. Falling back to CPU.")
                params["device"] = "cpu"
                self.model = lgb.train(params, train_data, num_boost_round=250)
            else:
                raise
        return self.model

    def evaluate(self, test_file):
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call .fit() first.")

        df = pd.read_csv(test_file)
        if df is None or df.empty:
            raise ValueError(f"Test file '{test_file}' is empty or cannot be read.")
        df = self._preprocess(df)
        X_test, y_test = self._prepare_features(df)

        y_pred = self.model.predict(X_test)

        # Xác định nhị phân / đa lớp theo shape (và theo self.objective nếu có)
        is_multiclass = False
        if getattr(self, "objective", None) == "multiclass":
            # vẫn kiểm tra shape đề phòng model trả 1D
            is_multiclass = (
                hasattr(y_pred, "ndim") and y_pred.ndim == 2 and y_pred.shape[1] > 1
            )
        else:
            # nếu chưa đặt self.objective, suy ra từ shape
            is_multiclass = (
                hasattr(y_pred, "ndim") and y_pred.ndim == 2 and y_pred.shape[1] > 2
            )

        if is_multiclass:
            # y_pred: (n_samples, n_classes)
            y_pred_label = np.argmax(y_pred, axis=1)
        else:
            # Binary: y_pred có thể là:
            #  - 1D: prob lớp 1
            #  - 2D với 2 cột: [:, 1] là prob lớp 1
            if hasattr(y_pred, "ndim") and y_pred.ndim == 2:
                if y_pred.shape[1] == 1:
                    y_prob = y_pred.ravel()
                else:
                    y_prob = y_pred[:, 1]
            else:
                y_prob = np.asarray(y_pred).ravel()

            y_pred_label = (y_prob >= 0.5).astype(int)

        f1 = fbeta_score(y_test, y_pred_label, beta=0.5)
        acc = accuracy_score(y_test, y_pred_label)
        report = classification_report(y_test, y_pred_label, output_dict=True)

        print(f"F1 Score (beta=0.5): {f1:.4f}")
        print(f"Accuracy: {acc:.4f}")
        return {"f1": f1, "accuracy": acc, "classification_report": report}

    def get_model(self):
        return self.model

    def train_and_evaluate(self, train_file, test_file):
        self.fit(train_file)
        results = self.evaluate(test_file)
        return self.model, results
