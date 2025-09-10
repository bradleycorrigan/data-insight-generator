import streamlit as st
import pandas as pd
from src.analyser import DataAnalyser

st.title("Automated EDA Tool")
st.write("Upload a CSV file to generate an automated exploratory data analysis report")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.write(df.head())
    
    # Initialize analyser
    analyser = DataAnalyser(df)

    # Basic info
    st.subheader("Basic Information")
    info = analyser.basic_info()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", info['shape'][0])
    with col2:
        st.metric("Columns", info['shape'][1])
    with col3:
        st.metric("Missing Values", sum(info['missing_values'].values()))
    
    # Numerical summary
    if analyser.numeric_cols:
        st.subheader("Numerical Summary")
        st.write(analyser.numerical_summary())

    # Categorical summary
    if analyser.categorical_cols:
        st.subheader("Categorical Summary")
        cat_summary = analyser.categorical_summary()
        for col, summary in cat_summary.items():
            st.write(f"**{col}**")
            st.write(f"Unique values: {summary['unique_count']}")
            st.bar_chart(pd.Series(summary['top_values']))
