import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Extracts features for ML duration prediction and disruption risk scoring.
    """
    def __init__(self):
        self.priority_map = {'High': 3, 'Medium': 2, 'Low': 1}
        self.dept_map = {'Engineering': 1, 'Traction': 2, 'S&T': 3}
        self.line_map = {'UP Line': 1, 'DOWN Line': 1, 'Yard Line': 0.5, 'Both Lines': 2.0}

    def fit_transform(self, df_requests):
        """Converts raw request dataframe into ML feature matrix and targets."""
        df = df_requests.copy()

        # 1. Temporal features
        if 'start_hour' not in df.columns:
            if 'preferred_start_time' in df.columns:
                df['start_hour'] = pd.to_datetime(df['preferred_start_time']).dt.hour
            else:
                df['start_hour'] = 10

        df['is_peak_hour'] = df['start_hour'].apply(
            lambda h: 1 if (7 <= h <= 10 or 17 <= h <= 20) else 0
        )

        # 2. Priority numeric mapping
        df['priority_val'] = df['priority'].map(self.priority_map).fillna(2)

        # 3. Department mapping
        df['dept_code'] = df['department'].map(self.dept_map).fillna(1)

        # 4. Line impact weight
        df['line_impact'] = df['track_line'].map(self.line_map).fillna(1.0)

        # 5. Density & Power block weights
        df['train_density_score'] = pd.to_numeric(df.get('train_density_score', 10), errors='coerce').fillna(10)
        df['power_block_flag'] = df['requires_power_block'].astype(int) if 'requires_power_block' in df.columns else 0

        # 6. Work complexity index
        df['complexity_index'] = (
            df['dept_code'] * 0.3 +
            df['priority_val'] * 0.4 +
            df['line_impact'] * 0.5 +
            df['power_block_flag'] * 0.8
        )

        # Target 1: Estimated realistic duration (for ML regression task)
        # Real duration fluctuates based on complexity, traffic density delays, and peak hours
        if 'requested_duration_hours' in df.columns:
            df['actual_duration_predicted_target'] = (
                df['requested_duration_hours'] * (1.0 + 0.15 * df['is_peak_hour'] + 0.05 * (df['train_density_score'] > 20).astype(int))
            ).round(2)
        else:
            df['actual_duration_predicted_target'] = 3.0

        # Target 2: Disruption Risk Category (Low, Medium, High)
        risk_score = df['complexity_index'] * (1 + 0.5 * df['is_peak_hour']) * (1 + df['train_density_score'] / 50.0)
        df['disruption_risk_target'] = pd.qcut(risk_score, q=3, labels=['Low', 'Medium', 'High']).astype(str)

        feature_cols = [
            'dept_code', 'priority_val', 'start_hour', 'is_peak_hour',
            'line_impact', 'train_density_score', 'power_block_flag', 'complexity_index'
        ]

        return df, feature_cols
