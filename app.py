# app.py - Updated with report generation
import streamlit as st
import pandas as pd
from src.analyser import DataAnalyser
from src.report_generator import ReportGenerator
import os

st.title("🔍 Automated EDA Tool")
st.write("Upload a CSV file to generate an automated exploratory data analysis report")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📊 Dataset Preview")
        st.write(df.head())

        # Initialize analyser
        analyser = DataAnalyser(df)

        # Basic info
        st.subheader("📈 Basic Information")
        info = analyser.basic_info()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{info['shape'][0]:,}")
        with col2:
            st.metric("Columns", info['shape'][1])
        with col3:
            st.metric("Missing Values", f"{sum(info['missing_values'].values()):,}")
        with col4:
            st.metric("Duplicates", f"{info['duplicate_rows']:,}")
        
        # Show data types
        st.subheader("🏷️ Data Types")
        dtype_df = pd.DataFrame({
            'Column': list(info['dtypes'].keys()),
            'Data Type': list(info['dtypes'].values()),
            'Missing Count': [info['missing_values'][col] for col in info['dtypes'].keys()]
        })
        st.dataframe(dtype_df, use_container_width=True)
        
        # Numerical summary
        if analyser.numeric_cols:
            st.subheader("🔢 Numerical Summary")
            numeric_summary = analyser.numerical_summary()
            st.dataframe(numeric_summary.round(3), use_container_width=True)
            
            # Outliers
            st.subheader("🎯 Outlier Detection")
            outliers = analyser .detect_outliers()
            outlier_df = pd.DataFrame([
                {'Column': col, 'Outlier Count': info['count'], 'Percentage': f"{info['percentage']:.2f}%"}
                for col, info in outliers.items()
            ])
            st.dataframe(outlier_df, use_container_width=True)
        
        # Categorical summary
        if analyser.categorical_cols:
            st.subheader("📝 Categorical Summary")
            categorical_summary = analyser.categorical_summary()

            for col, summary in categorical_summary.items():
                with st.expander(f"📊 {col}"):
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.metric("Unique Values", summary['unique_count'])
                    with col_info2:
                        st.metric("Missing %", f"{summary['missing_percentage']:.1f}%")
                    
                    st.write("**Top Values:**")
                    top_values_df = pd.DataFrame(
                        list(summary['top_values'].items())[:10], 
                        columns=['Value', 'Count']
                    )
                    st.dataframe(top_values_df, use_container_width=True)
        
        # Generate Report Button
        st.subheader("📄 Generate HTML Report")
        if st.button("Generate Comprehensive Report", type="primary"):
            with st.spinner("Generating report..."):
                report_gen = ReportGenerator(analyser)
                report_path = report_gen.generate_html_report()
            
            st.success(f"✅ Report generated successfully!")
            st.info(f"📁 Report saved to: `{report_path}`")
            
            # Provide download link
            with open(report_path, 'rb') as file:
                st.download_button(
                    label="📥 Download HTML Report",
                    data=file.read(),
                    file_name=f"eda_report_{uploaded_file.name.replace('.csv', '')}.html",
                    mime="text/html"
                )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please make sure your CSV file is properly formatted.")

else:
    st.info("👆 Please upload a CSV file to get started!")
    
    # Show example
    with st.expander("📋 See Example"):
        st.write("Try uploading a CSV with columns like:")
        example_df = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'City': ['New York', 'London', 'Paris'],
            'Salary': [50000, 60000, 70000]
        })
        st.dataframe(example_df)