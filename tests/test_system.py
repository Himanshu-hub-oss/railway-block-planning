import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add src path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from ml_models import MaintenanceMLPipeline
from conflict_detection import ConflictDetector
from optimization import MaintenanceOptimizer
from recommendation_engine import AIRecommendationEngine

def test_data_processor():
    processor = DataProcessor()
    summary = processor.load_dataset()
    assert isinstance(summary, dict)
    assert 'stations_count' in summary

    df_reqs = processor.generate_demo_maintenance_requests(num_requests=10, seed=42)
    assert not df_reqs.empty
    assert len(df_reqs) == 10
    assert 'department' in df_reqs.columns
    assert 'station_code' in df_reqs.columns
    assert df_reqs['is_demo'].all()

def test_feature_engineering():
    processor = DataProcessor()
    processor.load_dataset()
    df_reqs = processor.generate_demo_maintenance_requests(num_requests=10, seed=42)

    fe = FeatureEngineer()
    df_processed, feature_cols = fe.fit_transform(df_reqs)
    assert len(feature_cols) > 0
    assert 'actual_duration_predicted_target' in df_processed.columns
    assert 'disruption_risk_target' in df_processed.columns

def test_ml_pipeline():
    processor = DataProcessor()
    processor.load_dataset()
    df_reqs = processor.generate_demo_maintenance_requests(num_requests=20, seed=42)

    fe = FeatureEngineer()
    df_processed, feature_cols = fe.fit_transform(df_reqs)

    ml = MaintenanceMLPipeline()
    results = ml.train_and_evaluate(df_processed, feature_cols)
    assert 'best_duration_model' in results
    assert 'best_risk_model' in results

    X_test = df_processed[feature_cols].iloc[:3]
    preds_dur, preds_risk, importances = ml.predict(X_test)
    assert len(preds_dur) == 3
    assert len(preds_risk) == 3

def test_conflict_detection():
    processor = DataProcessor()
    processor.load_dataset()
    df_reqs = processor.generate_demo_maintenance_requests(num_requests=15, seed=42)

    cd = ConflictDetector()
    report = cd.analyze_conflicts(df_reqs)
    assert 'total_conflicts' in report
    assert 'compatible_pairs' in report

def test_optimization_engine():
    processor = DataProcessor()
    processor.load_dataset()
    df_reqs = processor.generate_demo_maintenance_requests(num_requests=15, seed=42)

    cd = ConflictDetector()
    conflict_report = cd.analyze_conflicts(df_reqs)

    opt = MaintenanceOptimizer()
    results = opt.optimize_blocks(df_reqs, conflict_report)

    assert 'optimized_blocks_count' in results
    assert results['optimized_blocks_count'] <= len(df_reqs)
    assert results['time_saved_hours'] >= 0

def test_edge_case_empty_requests():
    opt = MaintenanceOptimizer()
    df_empty = pd.DataFrame()
    results = opt.optimize_blocks(df_empty, {'conflicts_list': [], 'compatible_pairs': []})
    assert results['optimized_blocks_count'] == 0
    assert results['time_saved_hours'] == 0.0
