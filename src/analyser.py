import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataAnalyser:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
    def basic_info(self):
        """Get basic dataset information"""
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicate_rows': self.df.duplicated().sum()
        }
    
    def numerical_summary(self):
        """Statistical summary of numerical columns"""
        if not self.numeric_cols:
            return None
        
        summary = self.df[self.numeric_cols].describe()
        # Add additional statistics
        summary.loc['skewness'] = self.df[self.numeric_cols].skew()
        summary.loc['kurtosis'] = self.df[self.numeric_cols].kurtosis()
        
        return summary
    
    def categorical_summary(self):
        """Summary of categorical columns"""
        if not self.categorical_cols:
            return None
            
        cat_summary = {}
        for col in self.categorical_cols:
            cat_summary[col] = {
                'unique_count': self.df[col].nunique(),
                'top_values': self.df[col].value_counts().head(10).to_dict(),
                'missing_percentage': (self.df[col].isnull().sum() / len(self.df)) * 100
            }
        return cat_summary
    
    def detect_outliers(self):
        """Detect outliers using IQR method"""
        outliers = {}
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            outliers[col] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(self.df)) * 100
            }
        return outliers