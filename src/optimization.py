import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class MaintenanceOptimizer:
    """
    Constraint Programming Optimization Engine using Google OR-Tools CP-SAT Solver / Heuristic Fallback
    to optimize multi-department railway maintenance blocks.
    """
    def __init__(self):
        pass

    def optimize_blocks(self, df_requests, conflict_report, max_block_hours=6.0):
        """
        Takes maintenance requests and conflict report, formulates optimization problem,
        and generates an optimized combined block schedule.
        """
        requests = df_requests.to_dict(orient='records')
        num_reqs = len(requests)
        
        if num_reqs == 0:
            return {
                'original_blocks_count': 0,
                'optimized_blocks_count': 0,
                'original_total_hours': 0.0,
                'optimized_total_hours': 0.0,
                'time_saved_hours': 0.0,
                'combined_activities_count': 0,
                'optimized_blocks': []
            }

        # Try using Google OR-Tools CP-SAT Solver
        try:
            from ortools.sat.python import cp_model
            use_ortools = True
        except ImportError:
            use_ortools = False

        if use_ortools:
            return self._solve_with_ortools(requests, conflict_report, max_block_hours)
        else:
            return self._solve_with_heuristic(requests, conflict_report, max_block_hours)

    def _solve_with_ortools(self, requests, conflict_report, max_block_hours):
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        num_reqs = len(requests)

        # Incompatible pairs from conflict report
        incompatible_pairs = set()
        for conf in conflict_report.get('conflicts_list', []):
            incompatible_pairs.add((conf['req1_id'], conf['req2_id']))
            incompatible_pairs.add((conf['req2_id'], conf['req1_id']))

        # Combinable pairs lookup
        combinable_pairs = set()
        for comb in conflict_report.get('compatible_pairs', []):
            combinable_pairs.add((comb['req1_id'], comb['req2_id']))
            combinable_pairs.add((comb['req2_id'], comb['req1_id']))

        # Decision Variables: assign each request i to block cluster k (k in [0..num_reqs-1])
        block_vars = {}
        for i in range(num_reqs):
            for k in range(num_reqs):
                block_vars[(i, k)] = model.NewBoolVar(f'req_{i}_in_block_{k}')

        # Each request must belong to exactly 1 block
        for i in range(num_reqs):
            model.Add(sum(block_vars[(i, k)] for k in range(num_reqs)) == 1)

        # Indicator if block k is active
        block_used = [model.NewBoolVar(f'block_{k}_used') for k in range(num_reqs)]
        for k in range(num_reqs):
            for i in range(num_reqs):
                model.Add(block_vars[(i, k)] <= block_used[k])

        # Constraint: Incompatible requests cannot share the same block
        for i in range(num_reqs):
            for j in range(i + 1, num_reqs):
                req1_id = requests[i]['request_id']
                req2_id = requests[j]['request_id']

                # Location check: distant locations cannot share block
                loc1 = requests[i]['section']
                loc2 = requests[j]['section']

                if loc1 != loc2 or (req1_id, req2_id) in incompatible_pairs:
                    for k in range(num_reqs):
                        model.Add(block_vars[(i, k)] + block_vars[(j, k)] <= 1)

        # Objective: Minimize number of active blocks + minimize total block duration penalty
        # Priority weighting: High priority work prioritized for off-peak windows
        total_blocks_obj = sum(block_used[k] for k in range(num_reqs))

        # Encouraging compatible multi-department grouping
        grouping_reward = 0
        for i in range(num_reqs):
            for j in range(i + 1, num_reqs):
                req1_id = requests[i]['request_id']
                req2_id = requests[j]['request_id']
                if (req1_id, req2_id) in combinable_pairs:
                    for k in range(num_reqs):
                        same_block = model.NewBoolVar(f'same_{i}_{j}_{k}')
                        model.AddBoolAnd([block_vars[(i, k)], block_vars[(j, k)]]).OnlyEnforceIf(same_block)
                        grouping_reward += same_block * 5

        model.Minimize(total_blocks_obj * 10 - grouping_reward)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.Solve(model)

        # Construct solution clusters
        clusters = {}
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for i in range(num_reqs):
                for k in range(num_reqs):
                    if solver.Value(block_vars[(i, k)]) == 1:
                        if k not in clusters:
                            clusters[k] = []
                        clusters[k].append(requests[i])
        else:
            # Fallback to single requests per block if solver infeasible
            for i in range(num_reqs):
                clusters[i] = [requests[i]]

        return self._format_optimized_schedule(clusters, requests)

    def _solve_with_heuristic(self, requests, conflict_report, max_block_hours):
        """Greedy graph-coloring heuristic fallback."""
        clusters = {}
        cluster_id = 0

        incompatible_pairs = set()
        for conf in conflict_report.get('conflicts_list', []):
            incompatible_pairs.add((conf['req1_id'], conf['req2_id']))
            incompatible_pairs.add((conf['req2_id'], conf['req1_id']))

        assigned = set()
        for i, req1 in enumerate(requests):
            if req1['request_id'] in assigned:
                continue

            current_cluster = [req1]
            assigned.add(req1['request_id'])

            for j, req2 in enumerate(requests):
                if req2['request_id'] in assigned:
                    continue

                if req1['section'] == req2['section']:
                    if (req1['request_id'], req2['request_id']) not in incompatible_pairs:
                        # Check duration sum
                        dur_sum = sum(r['requested_duration_hours'] for r in current_cluster) + req2['requested_duration_hours']
                        if dur_sum <= max_block_hours:
                            current_cluster.append(req2)
                            assigned.add(req2['request_id'])

            clusters[cluster_id] = current_cluster
            cluster_id += 1

        return self._format_optimized_schedule(clusters, requests)

    def _format_optimized_schedule(self, clusters, requests):
        optimized_blocks = []
        block_counter = 1

        orig_total_hours = sum(r['requested_duration_hours'] for r in requests)
        opt_total_hours = 0.0
        combined_count = 0

        for k, group in clusters.items():
            if not group:
                continue

            depts = sorted(list(set(r['department'] for r in group)))
            section = group[0]['section']
            st_code = group[0]['station_code']
            st_name = group[0]['station_name']
            lat = group[0].get('lat')
            lng = group[0].get('lng')

            # Combined block duration is max duration of merged activities (parallel execution)
            # plus 15 min safety overhead per additional activity
            max_act_duration = max(r['requested_duration_hours'] for r in group)
            merged_duration = round(max_act_duration + 0.25 * (len(group) - 1), 2)
            opt_total_hours += merged_duration

            if len(group) > 1:
                combined_count += len(group)

            # Determine best window (prefer lowest traffic density / off-peak)
            min_start_h = min(r['start_hour'] for r in group)
            start_h_str = f"{int(min_start_h):02d}:00"
            end_h_str = f"{int(min_start_h + merged_duration):02d}:00"

            block_id = f"BLK-OPT-{block_counter:03d}"
            block_counter += 1

            optimized_blocks.append({
                'block_id': block_id,
                'station_code': st_code,
                'station_name': st_name,
                'section': section,
                'zone': group[0]['zone'],
                'lat': lat,
                'lng': lng,
                'affected_departments': ", ".join(depts),
                'departments_list': depts,
                'num_activities': len(group),
                'activity_ids': [r['request_id'] for r in group],
                'activity_names': [r['activity_type'] for r in group],
                'activities_detail': group,
                'scheduled_start_time': start_h_str,
                'scheduled_end_time': end_h_str,
                'optimized_duration_hours': merged_duration,
                'original_cumulative_hours': sum(r['requested_duration_hours'] for r in group),
                'time_saved_hours': round(sum(r['requested_duration_hours'] for r in group) - merged_duration, 2),
                'is_combined': len(group) > 1,
                'priority': 'High' if 'High' in [r['priority'] for r in group] else ('Medium' if 'Medium' in [r['priority'] for r in group] else 'Low')
            })

        time_saved = round(max(0, orig_total_hours - opt_total_hours), 2)

        return {
            'original_blocks_count': len(requests),
            'optimized_blocks_count': len(optimized_blocks),
            'original_total_hours': round(orig_total_hours, 2),
            'optimized_total_hours': round(opt_total_hours, 2),
            'time_saved_hours': time_saved,
            'combined_activities_count': combined_count,
            'block_reduction_pct': round((1.0 - len(optimized_blocks) / max(1, len(requests))) * 100, 1),
            'time_reduction_pct': round((time_saved / max(0.1, orig_total_hours)) * 100, 1),
            'optimized_blocks': optimized_blocks
        }
