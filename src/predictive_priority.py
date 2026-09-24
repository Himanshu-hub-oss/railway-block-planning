import pandas as pd
import numpy as np

class PredictivePriorityEngine:
    """
    Calculates maintenance priority scores (0 - 100) and risk levels based on multi-criteria operational inputs.
    Clearly labels simulated asset condition / failure history parameters when user inputs are synthetic.
    """
    def __init__(self):
        self.criticality_weights = {
            'High': 35,
            'Medium': 20,
            'Low': 10
        }
        self.asset_condition_weights = {
            'Critical Failure / Hotspot': 30,
            'Poor (Wear > 70%)': 25,
            'Moderate (Wear 40-70%)': 15,
            'Good (Routine Monitoring)': 5
        }
        self.dept_priority_weights = {
            'Engineering': 20,
            'Traction': 18,
            'S&T': 22
        }

    def calculate_priority_score(self, request_data):
        """
        Calculates maintenance priority score (0-100) for a given request.
        """
        req = request_data if isinstance(request_data, dict) else request_data.to_dict()

        prio_str = req.get('priority', 'Medium')
        dept_str = req.get('department', 'Engineering')
        crit_score = self.criticality_weights.get(prio_str, 20)
        dept_score = self.dept_priority_weights.get(dept_str, 20)

        # Asset condition score (simulated/provided)
        asset_cond = req.get('asset_condition', 'Moderate (Wear 40-70%)')
        asset_score = self.asset_condition_weights.get(asset_cond, 15)

        # Maintenance frequency penalty (older maintenance date = higher urgency)
        days_since_last_maint = req.get('days_since_last_maintenance', 45)
        freq_score = min(15, int(days_since_last_maint / 10))

        # Traffic density impact
        density = float(req.get('train_density_score', 15))
        density_score = min(10, int(density / 5))

        total_score = min(100, crit_score + dept_score + asset_score + freq_score + density_score)

        if total_score >= 80:
            risk_level = "CRITICAL / HIGH"
            rec_action = "🚨 Immediate Schedule Priority: Assign to next available shadow maintenance window."
        elif total_score >= 60:
            risk_level = "HIGH"
            rec_action = "⚠️ Schedule within 48 Hours: Coordinate with adjacent department requests."
        elif total_score >= 40:
            risk_level = "MEDIUM"
            rec_action = "📅 Routine Window: Group with regular weekend traffic disconnection block."
        else:
            risk_level = "LOW"
            rec_action = "💤 Low Urgency: Shadow during scheduled major overhaul block."

        return {
            'priority_score': total_score,
            'risk_level': risk_level,
            'recommended_action': rec_action,
            'breakdown': {
                'Safety Criticality Score': crit_score,
                'Department Priority Weight': dept_score,
                'Asset Health Score': asset_score,
                'Maintenance Backlog Weight': freq_score,
                'Traffic Density Weight': density_score
            },
            'is_simulated_scenario': True  # Clear label for asset history metrics
        }
