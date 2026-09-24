import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class MaintenanceMLPipeline:
    """
    Evaluates and trains ML models for block duration prediction and disruption risk scoring.
    """
    def __init__(self):
        self.duration_model = None
        self.risk_model = None
        self.duration_metrics = {}
        self.risk_metrics = {}
        self.is_trained = False

    def train_and_evaluate(self, df_processed, feature_cols):
        """
        Trains and compares models for both regression (duration) and classification (risk).
        """
        X = df_processed[feature_cols]
        y_duration = df_processed['actual_duration_predicted_target']
        y_risk = df_processed['disruption_risk_target']

        # Split train/test or use full dataset if small demo sample size
        n_samples = len(X)
        if n_samples < 8:
            X_train, X_test = X, X
            y_dur_train, y_dur_test = y_duration, y_duration
            y_risk_train, y_risk_test = y_risk, y_risk
        else:
            X_train, X_test, y_dur_train, y_dur_test, y_risk_train, y_risk_test = train_test_split(
                X, y_duration, y_risk, test_size=0.25, random_state=42
            )

        # --- Task A: Duration Regression ---
        reg_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        reg_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)

        reg_rf.fit(X_train, y_dur_train)
        pred_dur_rf = reg_rf.predict(X_test)

        reg_gb.fit(X_train, y_dur_train)
        pred_dur_gb = reg_gb.predict(X_test)

        rf_mae = mean_absolute_error(y_dur_test, pred_dur_rf)
        rf_rmse = np.sqrt(mean_squared_error(y_dur_test, pred_dur_rf))
        rf_r2 = r2_score(y_dur_test, pred_dur_rf)

        gb_mae = mean_absolute_error(y_dur_test, pred_dur_gb)
        gb_rmse = np.sqrt(mean_squared_error(y_dur_test, pred_dur_gb))
        gb_r2 = r2_score(y_dur_test, pred_dur_gb)

        self.duration_metrics = {
            'Random Forest': {'MAE': round(rf_mae, 4), 'RMSE': round(rf_rmse, 4), 'R2': round(rf_r2, 4)},
            'Gradient Boosting': {'MAE': round(gb_mae, 4), 'RMSE': round(gb_rmse, 4), 'R2': round(gb_r2, 4)}
        }
        
        # Select best regression model
        self.duration_model = reg_gb if gb_r2 >= rf_r2 else reg_rf

        # --- Task B: Disruption Risk Classification ---
        clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

        clf_rf.fit(X_train, y_risk_train)
        pred_risk_rf = clf_rf.predict(X_test)

        clf_gb.fit(X_train, y_risk_train)
        pred_risk_gb = clf_gb.predict(X_test)

        acc_rf = accuracy_score(y_risk_test, pred_risk_rf)
        prec_rf, rec_rf, f1_rf, _ = precision_recall_fscore_support(y_risk_test, pred_risk_rf, average='macro', zero_division=0)

        acc_gb = accuracy_score(y_risk_test, pred_risk_gb)
        prec_gb, rec_gb, f1_gb, _ = precision_recall_fscore_support(y_risk_test, pred_risk_gb, average='macro', zero_division=0)

        self.risk_metrics = {
            'Random Forest': {
                'Accuracy': round(acc_rf, 4),
                'Precision': round(prec_rf, 4),
                'Recall': round(rec_rf, 4),
                'F1-Score': round(f1_rf, 4),
                'Confusion Matrix': confusion_matrix(y_risk_test, pred_risk_rf).tolist()
            },
            'Gradient Boosting': {
                'Accuracy': round(acc_gb, 4),
                'Precision': round(prec_gb, 4),
                'Recall': round(rec_gb, 4),
                'F1-Score': round(f1_gb, 4),
                'Confusion Matrix': confusion_matrix(y_risk_test, pred_risk_gb).tolist()
            }
        }

        self.risk_model = clf_gb if f1_gb >= f1_rf else clf_rf
        self.is_trained = True

        return {
            'best_duration_model': self.duration_model.__class__.__name__,
            'duration_metrics': self.duration_metrics,
            'best_risk_model': self.risk_model.__class__.__name__,
            'risk_metrics': self.risk_metrics
        }

    def predict(self, X_input):
        """Generates duration predictions and disruption risk classifications for new requests."""
        if not self.is_trained or self.duration_model is None:
            raise ValueError("ML Pipeline has not been trained yet.")

        pred_durations = self.duration_model.predict(X_input)
        pred_risks = self.risk_model.predict(X_input)
        
        # Calculate feature importances
        if hasattr(self.risk_model, 'feature_importances_'):
            importances = dict(zip(X_input.columns, self.risk_model.feature_importances_.round(4)))
        else:
            importances = {}

        return pred_durations, pred_risks, importances
