# src/report_generator.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from io import BytesIO
import os

class ReportGenerator:
    def __init__(self, analyser):
        self.analyser = analyser
        self.df = analyser.df
        
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string for HTML embedding"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png)
        return graphic.decode('utf-8')
    
    def create_visualizations(self):
        """Create all visualizations and return as base64 strings"""
        plots = {}
        # Correlation heatmap for numeric columns
        if len(self.analyser.numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = self.df[self.analyser.numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Matrix')
            plots['correlation'] = self._fig_to_base64(fig)
            plt.close(fig)
        # Distribution plots for numeric columns
        if self.analyser.numeric_cols:
            n_cols = min(3, len(self.analyser.numeric_cols))
            n_rows = (len(self.analyser.numeric_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            for i, col in enumerate(self.analyser.numeric_cols):
                ax = axes[i] if len(self.analyser.numeric_cols) > 1 else axes
                self.df[col].hist(bins=30, ax=ax, alpha=0.7)
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
            # Hide empty subplots
            for j in range(len(self.analyser.numeric_cols), len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            plots['distributions'] = self._fig_to_base64(fig)
            plt.close(fig)
        return plots
    
    def generate_html_report(self, output_path=None):
        """Generate comprehensive HTML report"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"outputs/eda_report_{timestamp}.html"
        # Ensure outputs directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Get analysis results
        basic_info = self.analyser.basic_info()
        numeric_summary = self.analyser.numerical_summary()
        categorical_summary = self.analyser.categorical_summary()
        outliers = self.analyser.detect_outliers()
        plots = self.create_visualizations()
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>EDA Report - {datetime.now().strftime("%Y-%m-%d %H:%M")}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1, h2, h3 {{ color: #333; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #e8f4f8; border-radius: 5px; min-width: 150px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .plot-container {{ text-align: center; margin: 30px 0; }}
                .plot-container img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Automated EDA Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <h2>Dataset Overview</h2>
                <div>
                    <div class="metric">
                        <div class="metric-value">{basic_info['shape'][0]:,}</div>
                        <div class="metric-label">Rows</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{basic_info['shape'][1]}</div>
                        <div class="metric-label">Columns</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{sum(basic_info['missing_values'].values()):,}</div>
                        <div class="metric-label">Missing Values</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{basic_info['duplicate_rows']:,}</div>
                        <div class="metric-label">Duplicates</div>
                    </div>
                </div>
        """
        
        # Add missing values warning if any
        if sum(basic_info['missing_values'].values()) > 0:
            html_content += """
                <div class="warning">
                    <strong>Warning:</strong> This dataset contains missing values. Consider data cleaning strategies.
                </div>
            """
        
        # Numerical summary
        if numeric_summary is not None:
            html_content += """
                <h2>Numerical Columns Summary</h2>
                <div style="overflow-x: auto;">
            """ + numeric_summary.to_html(classes="") + "</div>"
        
        # Categorical summary
        if categorical_summary:
            html_content += "<h2>Categorical Columns Summary</h2>"
            for col, summary in categorical_summary.items():
                html_content += f"""
                    <h3>{col}</h3>
                    <p>Unique values: {summary['unique_count']} | Missing: {summary['missing_percentage']:.1f}%</p>
                    <p>Top values:</p>
                    <ul>
                """
                for value, count in list(summary['top_values'].items())[:5]:
                    html_content += f"<li>{value}: {count}</li>"
                html_content += "</ul>"
        
        # Outliers section
        if outliers:
            html_content += "<h2>Outlier Detection</h2><table><tr><th>Column</th><th>Outlier Count</th><th>Percentage</th></tr>"
            for col, outlier_info in outliers.items():
                html_content += f"""
                    <tr>
                        <td>{col}</td>
                        <td>{outlier_info['count']}</td>
                        <td>{outlier_info['percentage']:.2f}%</td>
                    </tr>
                """
            html_content += "</table>"
        
        # Add visualizations
        if 'correlation' in plots:
            html_content += f"""
                <h2>Correlation Analysis</h2>
                <div class="plot-container">
                    <img src="data:image/png;base64,{plots['correlation']}" alt="Correlation Matrix">
                </div>
            """
        
        if 'distributions' in plots:
            html_content += f"""
                <h2>Data Distributions</h2>
                <div class="plot-container">
                    <img src="data:image/png;base64,{plots['distributions']}" alt="Data Distributions">
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path