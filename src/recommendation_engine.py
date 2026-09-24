try:
    import networkx as nx
except ImportError:
    nx = None
import pandas as pd

class AIRecommendationEngine:
    """
    Generates explainable AI recommendations, graph network visualizations,
    and structured 'Explain This Decision' breakdowns.
    """
    def __init__(self):
        pass

    def generate_recommendations(self, optimization_results, conflict_report):
        """Generates structured, human-readable explanations for all block recommendations."""
        recommendations = []

        for blk in optimization_results.get('optimized_blocks', []):
            req_count = blk['num_activities']
            depts = blk['affected_departments']
            loc = blk['section']
            dur = blk['optimized_duration_hours']
            time_saved = blk['time_saved_hours']
            act_ids = blk['activity_ids']

            if req_count > 1:
                rec_text = f"RECOMMENDED: Combine {req_count} requests ({', '.join(act_ids)}) into a single joint block."
                reason = (
                    f"Engineering and Traction/S&T requests target the same section ({loc}) during overlapping time windows. "
                    f"Combining them into a joint shadow block eliminates separate line shutdowns, saving {time_saved:.1f} hours."
                )
                status = "JOINT BLOCK"
                confidence = 0.95
                benefit = f"Saves {time_saved:.1f} hours of disconnection time and avoids {req_count - 1} separate line possessions."
                alt_window = f"{int(int(blk['scheduled_start_time'].split(':')[0]) + 4):02d}:00 - {int(int(blk['scheduled_start_time'].split(':')[0]) + 4 + dur):02d}:00"
            else:
                rec_text = f"STANDALONE: Execute request {act_ids[0]} ({blk['activity_names'][0]}) independently."
                reason = f"No compatible adjacent requests found at section {loc} within the allowable time window."
                status = "STANDALONE BLOCK"
                confidence = 0.88
                benefit = "Ensures dedicated track machine access and zero interference with parallel lines."
                alt_window = f"{int(int(blk['scheduled_start_time'].split(':')[0]) + 2):02d}:00 - {int(int(blk['scheduled_start_time'].split(':')[0]) + 2 + dur):02d}:00"

            # Structure full 'Explain This Decision' card
            explain_decision = {
                'decision': rec_text,
                'why': reason,
                'data_considered': f"Section {loc}, Train density ({blk.get('station_code')}), Requested windows, Line impacts",
                'conflicts_detected': f"{len(conflict_report.get('conflicts_list', []))} global conflicts evaluated",
                'constraints': f"Max block window <= 6.0h, Track line compatibility, Machine availability",
                'alternative_considered': f"Execute as individual blocks during off-peak window {alt_window}",
                'expected_impact': benefit
            }

            recommendations.append({
                'block_id': blk['block_id'],
                'status': status,
                'recommendation': rec_text,
                'reason': reason,
                'confidence_score': confidence,
                'affected_departments': depts,
                'affected_location': loc,
                'estimated_duration_hours': dur,
                'time_saved_hours': time_saved,
                'expected_benefit': benefit,
                'alternative_window': alt_window,
                'activities_detail': blk['activities_detail'],
                'explain_decision': explain_decision
            })

        return recommendations

    def build_network_graph(self, df_requests, conflict_report, optimization_results):
        """
        Builds NetworkX graph:
        Nodes: Requests, Stations, Departments
        Edges: Conflicts, Combinations, Same Station
        """
        G = nx.Graph()

        # Add station nodes
        stations = df_requests['station_code'].unique() if 'station_code' in df_requests.columns else []
        for st in stations:
            G.add_node(f"STATION_{st}", label=f"Station {st}", node_type='station')

        # Add request nodes
        for _, req in df_requests.iterrows():
            req_id = req['request_id']
            G.add_node(req_id, label=f"{req_id} ({req['department']})", node_type='request', department=req['department'], section=req['section'])
            if 'station_code' in req:
                G.add_edge(req_id, f"STATION_{req['station_code']}", edge_type='located_at')

        # Add conflict edges
        for conf in conflict_report.get('conflicts_list', []):
            G.add_edge(conf['req1_id'], conf['req2_id'], edge_type='conflict', severity=conf['severity'], label='Conflict')

        # Add combination edges
        for comb in conflict_report.get('compatible_pairs', []):
            G.add_edge(comb['req1_id'], comb['req2_id'], edge_type='combined', label='Combinable')

        return G
