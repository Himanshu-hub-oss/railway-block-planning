import pandas as pd
import numpy as np

class ConflictDetector:
    """
    Detects spatial, temporal, departmental, and safety constraint conflicts among maintenance requests.
    Generates conflict heatmap matrices and detailed 'Why Not Combine?' incompatibility logic.
    """
    def __init__(self):
        pass

    def analyze_conflicts(self, df_requests):
        """
        Analyzes a set of maintenance requests and identifies all conflicting request pairs.
        Returns a detailed conflict report, combinable pairs, incompatibility explanations, and matrix counts.
        """
        requests = df_requests.to_dict(orient='records')
        conflicts = []
        compatible_pairs = []
        why_not_combine = []

        # Department conflict matrix counters
        depts = ['Engineering', 'Traction', 'S&T']
        matrix_counts = {d1: {d2: 0 for d2 in depts} for d1 in depts}

        for i in range(len(requests)):
            for j in range(i + 1, len(requests)):
                req1 = requests[i]
                req2 = requests[j]
                dept1, dept2 = req1['department'], req2['department']

                # Check spatial overlap
                same_location = (
                    req1['station_code'] == req2['station_code'] or
                    req1['section'] == req2['section'] or
                    (req1['lat'] and req2['lat'] and abs(req1['lat'] - req2['lat']) < 0.05 and abs(req1['lng'] - req2['lng']) < 0.05)
                )

                # Check temporal overlap
                start1, end1 = float(req1['start_hour']), float(req1['start_hour']) + float(req1['requested_duration_hours'])
                start2, end2 = float(req2['start_hour']), float(req2['start_hour']) + float(req2['requested_duration_hours'])
                overlap_time = max(0, min(end1, end2) - max(start1, start2))

                is_time_overlapping = overlap_time > 0.5  # Overlap greater than 30 mins

                if same_location and is_time_overlapping:
                    dept_set = {dept1, dept2}

                    # Rule 1: Multi-department track block coordination
                    if dept_set == {'Engineering', 'Traction'}:
                        if req1['track_line'] == req2['track_line'] or 'Both Lines' in [req1['track_line'], req2['track_line']]:
                            compatible_pairs.append({
                                'req1_id': req1['request_id'],
                                'req2_id': req2['request_id'],
                                'type': 'Combinable Shadow Block',
                                'reason': f"Engineering ({req1['activity_type']}) and Traction ({req2['activity_type']}) share section {req1['section']} on line {req1['track_line']} and can be merged into a Joint Power+Traffic Block.",
                                'location': req1['section'],
                                'overlap_hours': round(overlap_time, 2)
                            })
                        else:
                            conflicts.append({
                                'conflict_id': f"CONF-{len(conflicts)+1:03d}",
                                'req1_id': req1['request_id'],
                                'req2_id': req2['request_id'],
                                'req1_dept': dept1,
                                'req2_dept': dept2,
                                'severity': 'High',
                                'type': 'Track Line Mismatch',
                                'description': f"Requests operate on conflicting lines ({req1['track_line']} vs {req2['track_line']}) at {req1['section']}.",
                                'location': req1['section']
                            })
                            matrix_counts[dept1][dept2] += 1
                            matrix_counts[dept2][dept1] += 1
                            why_not_combine.append({
                                'pair': f"{req1['request_id']} ({dept1}) & {req2['request_id']} ({dept2})",
                                'section': req1['section'],
                                'reason': "Track Line Mismatch: One targets UP Line while the other targets DOWN Line.",
                                'category': 'Track Line Mismatch'
                            })

                    elif dept_set == {'Engineering', 'S&T'} or dept_set == {'Traction', 'S&T'}:
                        compatible_pairs.append({
                            'req1_id': req1['request_id'],
                            'req2_id': req2['request_id'],
                            'type': 'Combinable Shadow Block',
                            'reason': f"Signal & Telecom work ({req2['activity_type'] if dept2=='S&T' else req1['activity_type']}) can shadow {dept1 if dept2 == 'S&T' else dept2} block at {req1['section']}.",
                            'location': req1['section'],
                            'overlap_hours': round(overlap_time, 2)
                        })

                    elif dept1 == dept2:
                        # Same department conflict - competing for same track machine/resources
                        conflicts.append({
                            'conflict_id': f"CONF-{len(conflicts)+1:03d}",
                            'req1_id': req1['request_id'],
                            'req2_id': req2['request_id'],
                            'req1_dept': dept1,
                            'req2_dept': dept2,
                            'severity': 'Critical',
                            'type': 'Resource / Machine Collision',
                            'description': f"Same department ({dept1}) requests competing for section {req1['section']} resources simultaneously.",
                            'location': req1['section']
                        })
                        matrix_counts[dept1][dept2] += 1
                        why_not_combine.append({
                            'pair': f"{req1['request_id']} ({dept1}) & {req2['request_id']} ({dept2})",
                            'section': req1['section'],
                            'reason': f"Equipment & Machine Conflict: Both requests require heavy {req1['required_resource']} on section {req1['section']}.",
                            'category': 'Equipment & Resource Collision'
                        })

                elif same_location and not is_time_overlapping:
                    # Adjacent time windows on same section
                    gap = min(abs(start2 - end1), abs(start1 - end2))
                    if gap <= 1.0:
                        compatible_pairs.append({
                            'req1_id': req1['request_id'],
                            'req2_id': req2['request_id'],
                            'type': 'Sequential Grouping',
                            'reason': f"Adjacent requests at {req1['section']} with only {gap:.1f}h gap. Combining minimizes line setup overhead.",
                            'location': req1['section'],
                            'overlap_hours': 0
                        })
                    else:
                        why_not_combine.append({
                            'pair': f"{req1['request_id']} ({dept1}) & {req2['request_id']} ({dept2})",
                            'section': req1['section'],
                            'reason': f"Time Gap Too Wide: Requested windows are separated by {gap:.1f} hours, exceeding max shadow gap threshold.",
                            'category': 'Time Gap Exceeded'
                        })

                elif not same_location and is_time_overlapping:
                    why_not_combine.append({
                        'pair': f"{req1['request_id']} ({dept1}) & {req2['request_id']} ({dept2})",
                        'section': f"{req1['section']} vs {req2['section']}",
                        'reason': f"Different Geographical Sections: {req1['section']} is geographically separate from {req2['section']}.",
                        'category': 'Different Railway Sections'
                    })

        # Convert matrix_counts to DataFrame
        df_heatmap = pd.DataFrame(matrix_counts)

        return {
            'total_conflicts': len(conflicts),
            'conflicts_list': conflicts,
            'compatible_pairs': compatible_pairs,
            'why_not_combine': why_not_combine,
            'conflict_matrix': df_heatmap
        }
