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
            # Clean and prepare data
            required_columns = ['Department', 'Tax/Non-Tax']
            has_state = 'State' in df.columns
                    
            # Update required columns based on state presence
            if has_state:
                required_columns.append('State')
                    
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
                
                # State filter (if state column exists)
                if has_state:
                    all_states = sorted(df['State'].unique())
                    
                    # Add option to select single state for pie chart
                    show_pie = st.sidebar.checkbox(
                        "Show Pie Chart for Single State",
                        value=False,
                        help="Check to show a pie chart for a single state"
                    )
                    
                    if show_pie and len(all_states) > 0:
                        # Show single state selector for pie chart
                        selected_state = st.sidebar.selectbox(
                            "Select State for Pie Chart",
                            options=all_states,
                            index=0
                        )
                        selected_states = [selected_state]
                        
                        # Create a pie chart for the selected state
                        state_data = df[df['State'] == selected_state]
                        if not state_data.empty:
                            st.subheader(f"Revenue Distribution - {selected_state}")
                            
                            # Aggregate by department for the selected state
                            pie_data = state_data.groupby('Department')[selected_fy].sum().reset_index()
                            pie_data = pie_data.sort_values(by=selected_fy, ascending=False)
                            
                            # Create two columns for layout
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Create pie chart
                                fig_pie = px.pie(
                                    pie_data,
                                    names='Department',
                                    values=selected_fy,
                                    title=f"{selected_fy} Revenue by Department",
                                    height=500
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                            
                            with col2:
                                # Show data table
                                st.subheader("Department-wise Revenue")
                                st.dataframe(
                                    pie_data.rename(columns={selected_fy: 'Revenue'}).sort_values(
                                        'Revenue', ascending=False
                                    ),
                                    column_config={
                                        'Revenue': st.column_config.NumberColumn(
                                            format="₹%.2f Cr"
                                        )
                                    },
                                    height=500
                                )
                    else:
                        # Show multi-select for states
                        selected_states = st.sidebar.multiselect(
                            "Select States",
                            options=all_states,
                            default=all_states[:min(5, len(all_states))],  # Default to first 5 states
                            help="Filter data by state"
                        )
                    
                    # Apply state filter if any states are selected
                    if selected_states:
                        df = df[df['State'].isin(selected_states)]
                
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
                filtered_df = df[df['Department'].isin(selected_depts)] if selected_depts else df
                
                # Prepare path for sunburst based on available columns
                path_columns = []
                if has_state and len(df['State'].unique()) > 1:
                    path_columns.append('State')
                path_columns.extend(['Tax/Non-Tax', 'Department'])
                
                # Create a new column for consistent coloring (Tax/Non-Tax)
                filtered_df['Revenue_Type'] = filtered_df['Tax/Non-Tax'].apply(
                    lambda x: 'Tax' if 'tax' in str(x).lower() or 'gst' in str(x).lower() else 'Non-Tax'
                )
                
                # Only show sunburst if not in single-state pie chart mode or if no states are selected
                if not has_state or not show_pie or len(selected_states) != 1:
                    # Create the sunburst chart
                    fig = px.sunburst(
                        filtered_df,
                        path=path_columns,
                        values=selected_fy,
                        color='Revenue_Type',  # Use the new column for coloring
                        color_discrete_map={
                            'Tax': '#4B78B2',    # Blue for all tax types
                            'Non-Tax': '#F6A756'  # Orange for Non-Tax
                        },
                        maxdepth=depth_level,
                        title=f"Revenue Composition - {selected_fy}",
                        height=800,
                        hover_data={'Revenue_Type': False}  # Hide the Revenue_Type from hover
                    )
                    
                    # Display the sunburst chart
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.subheader("Sunburst Chart")
                    st.info("Sunburst chart is hidden when viewing a single state's pie chart. Uncheck 'Show Pie Chart for Single State' to see the sunburst chart.")
                    
                    # Add a summary of the selected state's data
                    state_summary = filtered_df[filtered_df['State'].isin(selected_states)]
                    if not state_summary.empty:
                        st.subheader(f"Summary for {selected_states[0]}")
                        
                        # Calculate totals
                        total_revenue = state_summary[selected_fy].sum()
                        tax_revenue = state_summary[state_summary['Revenue_Type'] == 'Tax'][selected_fy].sum()
                        non_tax_revenue = total_revenue - tax_revenue
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Revenue", f"₹{total_revenue:,.2f} Cr")
                        with col2:
                            st.metric("Tax Revenue", f"₹{tax_revenue:,.2f} Cr", 
                                    f"{tax_revenue/total_revenue*100:.1f}%" if total_revenue > 0 else "0%")
                        with col3:
                            st.metric("Non-Tax Revenue", f"₹{non_tax_revenue:,.2f} Cr", 
                                    f"{non_tax_revenue/total_revenue*100:.1f}%" if total_revenue > 0 else "0%")
                        
                        # Show top departments
                        st.subheader("Top Departments by Revenue")
                        top_depts = state_summary.groupby('Department')[selected_fy].sum().nlargest(5)
                        st.bar_chart(top_depts)
                
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
                
                # Prepare columns for display
                display_columns = ['Department', 'Tax/Non-Tax']
                if has_state:
                    display_columns.insert(0, 'State')
                display_columns.append(selected_fy)
                
                st.dataframe(
                    filtered_df[display_columns].sort_values(
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
    - One row per department (and state, if applicable)
    - Columns for each financial year (e.g., FY2020, FY2021, FY2022)
    - A 'Department' column
    - A 'Tax/Non-Tax' column with values 'Tax' or 'Non-Tax'
    - (Optional) A 'State' column for state-wise analysis
    - Numeric values in the financial year columns
    
    ### Example Structure (with State):
    | State | Department | Tax/Non-Tax | FY2020 | FY2021 | FY2022 |
    |-------|------------|-------------|--------|--------|--------|
    | MH   | Sales Tax  | Tax         | 100    | 120    | 150    |
    | MH   | Grants     | Non-Tax     | 50     | 60     | 70     |
    | KA   | Sales Tax  | Tax         | 80     | 90     | 110    |
    
    ### Example Structure (without State):
    | Department | Tax/Non-Tax | FY2020 | FY2021 | FY2022 |
    |------------|-------------|--------|--------|--------|
    | Sales Tax  | Tax         | 180    | 210    | 260    |
    | Grants     | Non-Tax     | 50     | 60     | 70     |
    """)
