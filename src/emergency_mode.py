import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class EmergencyBlockEngine:
    """
    Handles fast-path Emergency Maintenance Block requests (🚨 EMERGENCY BLOCK).
    Scans schedules, existing requests, calculates immediate risk, and recommends the least-conflicting window.
    """
    def __init__(self, train_impact_engine=None):
        self.train_impact_engine = train_impact_engine

    def process_emergency_block(self, station_code, section, failure_type, severity, required_duration_h, df_existing_requests=None):
        """
        Processes an emergency block request and evaluates window options across 24 hours.
        """
        st_code = str(station_code).upper().strip()
        dur_h = float(required_duration_h)

        # 1. Search candidate start hours across the day (0 to 22)
        candidate_windows = []
        
        for start_h in [1, 3, 5, 11, 13, 15, 22, 23]:  # Typical low density off-peak windows
            if self.train_impact_engine:
                impact = self.train_impact_engine.analyze_block_impact(st_code, start_hour=start_h, duration_hours=dur_h)
                delay_mins = impact['total_estimated_delay_minutes']
                crit_trains = impact['critical_impact_count']
                affected_count = impact['affected_trains_count']
            else:
                delay_mins = int(np.random.randint(15, 120))
                crit_trains = 0 if start_h in [1, 23] else 1
                affected_count = 2

            # Check overlap with existing scheduled requests
            existing_conflicts = 0
            if df_existing_requests is not None and not df_existing_requests.empty:
                for _, req in df_existing_requests.iterrows():
                    if req.get('station_code') == st_code or req.get('section') == section:
                        req_start = float(req.get('start_hour', 10))
                        req_end = req_start + float(req.get('requested_duration_hours', 3))
                        if max(0, min(req_start + req_end, start_h + dur_h) - max(req_start, start_h)) > 0:
                            existing_conflicts += 1

            # Penalty score (lower is better)
            penalty_score = delay_mins * 1.5 + crit_trains * 100 + existing_conflicts * 50

            candidate_windows.append({
                'start_hour': start_h,
                'end_hour': start_h + dur_h,
                'window_str': f"{start_h:02d}:00 - {int(start_h + dur_h):02d}:00",
                'estimated_delay_mins': delay_mins,
                'critical_trains': crit_trains,
                'affected_trains_count': affected_count,
                'existing_conflicts': existing_conflicts,
                'penalty_score': penalty_score
            })

        # Sort candidate windows by penalty score
        candidate_windows.sort(key=lambda x: x['penalty_score'])
        best_window = candidate_windows[0]

        # Get detailed train impact for best window
        if self.train_impact_engine:
            best_impact = self.train_impact_engine.analyze_block_impact(st_code, start_hour=best_window['start_hour'], duration_hours=dur_h)
        else:
            best_impact = {'affected_trains': []}

        req_id = f"EMG-{severity[:3].upper()}-{np.random.randint(100, 999)}"

        return {
            'emergency_request_id': req_id,
            'station_code': st_code,
            'section': section,
            'failure_type': failure_type,
            'severity': severity,
            'required_duration_hours': dur_h,
            'recommended_window': best_window['window_str'],
            'recommended_start_hour': best_window['start_hour'],
            'recommended_end_hour': best_window['end_hour'],
            'estimated_delay_minutes': best_window['estimated_delay_mins'],
            'affected_trains_count': best_window['affected_trains_count'],
            'affected_trains_detail': best_impact.get('affected_trains', []),
            'candidate_windows_evaluated': candidate_windows,
            'risk_level': 'HIGH' if severity in ['CRITICAL', 'HIGH'] else 'MEDIUM',
            'explanation': (
                f"Emergency block at {st_code} ({section}) due to {failure_type} [{severity}] requires {dur_h}h window. "
                f"Recommended least-conflicting window is {best_window['window_str']} with estimated train delay of "
                f"{best_window['estimated_delay_mins']} mins and {best_window['existing_conflicts']} request collisions."
            ),
            'is_simulated_scenario': True  # Clear label for emergency demo
        }
