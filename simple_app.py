import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Hierarchical Revenue Analyzer",
    page_icon="🌳",
    layout="wide"
)

st.title("🌳 Hierarchical Revenue Analysis")

# Sample data upload
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)
        
        # Get financial year columns (assuming they start with 'FY')
        fy_cols = [col for col in df.columns if str(col).startswith('FY')]
        
        if not fy_cols:
            st.error("No financial year columns found. Please ensure your file has columns starting with 'FY'.")
        else:
            # Clean data
            required_columns = ['Department', 'Tax/Non-Tax']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}. Please ensure your file has these columns.")
            else:
                # Clean and prepare data
                df = df.dropna(subset=required_columns + fy_cols, how='all')
                
                # Convert FY columns to numeric
                for col in fy_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove any rows where all financial year values are missing
                df = df.dropna(subset=fy_cols, how='all')
                
                # Sidebar controls
                st.sidebar.header("Chart Controls")
                
                # Financial year selection
                selected_fy = st.sidebar.selectbox(
                    "Select Financial Year",
                    options=fy_cols[::-1],  # Show most recent years first
                    index=0
                )
                
                # Depth level selection
                depth_level = st.sidebar.slider(
                    "Hierarchy Depth",
                    min_value=1,
                    max_value=3,
                    value=2,
                    help="Adjust the depth of the hierarchy to show more or less detail"
                )
                
                # Department filter
                all_depts = df['Department'].unique()
                selected_depts = st.sidebar.multiselect(
                    "Filter Departments",
                    options=all_depts,
                    default=all_depts[:min(5, len(all_depts))],  # Default to first 5 departments
                    help="Select specific departments to include in the analysis"
                )
                
                # Apply department filter if any are selected
                if selected_depts:
                    filtered_df = df[df['Department'].isin(selected_depts)]
                else:
                    filtered_df = df
                
                # Create the sunburst chart
                fig = px.sunburst(
                    filtered_df,
                    path=['Tax/Non-Tax', 'Department'],
                    values=selected_fy,
                    color='Tax/Non-Tax',
                    color_discrete_map={
                        'Tax': '#4B78B2',  # Blue for Tax
                        'Non-Tax': '#F6A756'  # Orange for Non-Tax
                    },
                    maxdepth=depth_level,
                    title=f"Revenue Composition - {selected_fy}",
                    height=800
                )
                
                # Customize the layout
                fig.update_layout(
                    margin=dict(t=50, l=0, r=0, b=0),
                    hovermode='closest',
                    uniformtext=dict(minsize=10, mode='hide')
                )
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data summary
                st.subheader("Data Summary")
                
                # Summary statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Total Revenue",
                        f"₹{filtered_df[selected_fy].sum():,.2f} Cr",
                        delta=f"{(filtered_df[selected_fy].sum() / df[selected_fy].sum() - 1) * 100:.1f}% of total"
                    )
                
                with col2:
                    tax_ratio = filtered_df[filtered_df['Tax/Non-Tax'] == 'Tax'][selected_fy].sum() / filtered_df[selected_fy].sum()
                    st.metric(
                        "Tax/Non-Tax Ratio",
                        f"{tax_ratio:.1%} Tax",
                        f"{1 - tax_ratio:.1%} Non-Tax"
                    )
                
                # Data table
                st.subheader("Detailed View")
                st.dataframe(
                    filtered_df[['Department', 'Tax/Non-Tax', selected_fy]].sort_values(
                        by=selected_fy, ascending=False
                    ).reset_index(drop=True),
                    column_config={
                        selected_fy: st.column_config.NumberColumn(
                            format="₹%.2f Cr"
                        )
                    },
                    height=400
                )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please ensure your Excel file is in the correct format.")
else:
    st.info("Please upload an Excel file to begin analysis.")
    st.markdown("""
    ### Expected File Format:
    - One row per department
    - Columns for each financial year (e.g., FY2020, FY2021, FY2022)
    - A 'Department' column
    - A 'Tax/Non-Tax' column with values 'Tax' or 'Non-Tax'
    - Numeric values in the financial year columns
    
    ### Example Structure:
    | Department | Tax/Non-Tax | FY2020 | FY2021 | FY2022 |
    |------------|-------------|--------|--------|--------|
    | Sales Tax  | Tax         | 100    | 120    | 150    |
    | Grants     | Non-Tax     | 50     | 60     | 70     |
    """)
