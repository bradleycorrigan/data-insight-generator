# tests/test_analyser.py
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analyser import DataAnalyser

def test_analyser_basic_info():
    """Test basic dataset information extraction"""
    # Create sample data
    df = pd.DataFrame({
        'numeric_col': [1, 2, 3, 4, 5],
        'categorical_col': ['A', 'B', 'A', 'C', 'B'],
        'missing_col': [1, None, 3, None, 5]
    })
    
    analyser = DataAnalyser(df)
    info = analyser.basic_info()
    
    assert info['shape'] == (5, 3)
    assert len(info['columns']) == 3
    assert info['missing_values']['missing_col'] == 2
    assert info['duplicate_rows'] == 0

def test_numerical_summary():
    """Test numerical column analysis"""
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [10, 20, 30, 40, 50]
    })
    
    analyser = DataAnalyser(df)
    summary = analyser.numerical_summary()
    
    assert summary is not None
    assert 'col1' in summary.columns
    assert 'col2' in summary.columns
    assert 'mean' in summary.index
    assert summary.loc['mean', 'col1'] == 3.0

def test_categorical_summary():
    """Test categorical column analysis"""
    df = pd.DataFrame({
        'category': ['A', 'B', 'A', 'C', 'B', 'A']
    })
    
    analyser = DataAnalyser(df)
    summary = analyser.categorical_summary()
    
    assert 'category' in summary
    assert summary['category']['unique_count'] == 3
    assert summary['category']['top_values']['A'] == 3

def test_outlier_detection():
    """Test outlier detection using IQR method"""
    # Create data with clear outliers
    df = pd.DataFrame({
        'normal_data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'with_outliers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is an outlier
    })
    
    analyser = DataAnalyser(df)
    outliers = analyser.detect_outliers()
    
    # Should detect outliers in the second column
    assert outliers['with_outliers']['count'] > 0
    assert outliers['normal_data']['count'] == 0

def test_column_type_detection():
    """Test that columns are correctly categorised by type"""
    df = pd.DataFrame({
        'numeric_int': [1, 2, 3, 4, 5],
        'numeric_float': [1.1, 2.2, 3.3, 4.4, 5.5],
        'categorical': ['A', 'B', 'C', 'D', 'E'],
        'datetime': pd.date_range('2023-01-01', periods=5)
    })
    
    analyser = DataAnalyser(df)
    
    assert 'numeric_int' in analyser.numeric_cols
    assert 'numeric_float' in analyser.numeric_cols
    assert 'categorical' in analyser.categorical_cols
    assert 'datetime' in analyser.datetime_cols

def test_empty_dataframe():
    """Test behaviour with empty dataframe"""
    df = pd.DataFrame()
    analyser = DataAnalyser(df)
    info = analyser.basic_info()
    
    assert info['shape'] == (0, 0)
    assert len(info['columns']) == 0

def test_single_column():
    """Test with single column dataframe"""
    df = pd.DataFrame({'single_col': [1, 2, 3, 4, 5]})
    analyser = DataAnalyser(df)
    
    assert len(analyser.numeric_cols) == 1
    assert len(analyser.categorical_cols) == 0
    
    summary = analyser.numerical_summary()
    assert summary is not None
    assert 'single_col' in summary.columns

def test_missing_values_handling():
    """Test proper handling of missing values"""
    df = pd.DataFrame({
        'with_nulls': [1, 2, None, 4, None, 6],
        'complete': [10, 20, 30, 40, 50, 60]
    })
    
    analyser = DataAnalyser(df)
    info = analyser.basic_info()
    
    assert info['missing_values']['with_nulls'] == 2
    assert info['missing_values']['complete'] == 0

def test_duplicate_detection():
    """Test duplicate row detection"""
    df = pd.DataFrame({
        'col1': [1, 2, 3, 1, 2],  # First two rows are duplicated
        'col2': ['A', 'B', 'C', 'A', 'B']
    })
    
    analyser = DataAnalyser(df)
    info = analyser.basic_info()
    
    assert info['duplicate_rows'] == 2  # Two duplicate rows

if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])