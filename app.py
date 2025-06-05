import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from streamlit_extras.app_logo import add_logo

# Configure page
st.set_page_config(
    page_title="Advanced Financial Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom theme
add_logo("https://streamlit.io/images/brand/streamlit-mark-color.png", height=80)
st.markdown("""
<style>
[data-testid=stSidebar] { background-color: #333; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# Main navigation
with st.sidebar:
    st.title("Navigation")
    section = st.radio("Select Analysis Mode", [
        "🏠 Dashboard",
        "📈 Trend Analysis",
        "🌳 Revenue Composition",
        "🔮 Forecasting",
        "⚙️ Advanced Tools"
    ])

# Data processing functions
def load_and_process_data(uploaded_file, selected_sheet):
    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    
    # Clean and process data
    fy_cols = [col for col in df.columns if col.startswith("FY")]
    df = df.dropna(subset=["Department"], how='any')
    df = df[~df["Department"].astype(str).str.contains("Total", na=False)]
    
    for col in fy_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=fy_cols, how='all')
    return df, fy_cols

# File upload
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])
if not uploaded_file:
    st.info("Please upload an Excel file to begin analysis.")
    st.stop()

# Load data
excel_data = pd.ExcelFile(uploaded_file)
sheet_names = excel_data.sheet_names
selected_sheet = st.sidebar.selectbox("Select State Sheet", sheet_names)
df, fy_cols = load_and_process_data(uploaded_file, selected_sheet)

# Main content
if section == "🏠 Dashboard":
    st.title("📊 Financial Dashboard")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Revenue FY19", f"₹{df['FY19'].sum():,.2f} Cr")
    with col2:
        yoy_growth = ((df['FY19'].sum() - df['FY18'].sum()) / df['FY18'].sum()) * 100
        st.metric("YoY Growth", f"{yoy_growth:.1f}%")
    with col3:
        tax_ratio = df[df['Tax/Non-Tax'] == 'Tax']['FY19'].sum() / df['FY19'].sum()
        st.metric("Tax Revenue Ratio", f"{tax_ratio:.1%}")
    
    st.subheader("Revenue Composition")
    tab1, tab2, tab3 = st.tabs(["Treemap", "Sankey Flow", "Waterfall"])
    
    with tab1:
        treemap_df = df.melt(id_vars=["Department", "Tax/Non-Tax"], 
                           value_vars=fy_cols, var_name="FY", value_name="Revenue")
        fig = px.treemap(treemap_df, path=['Tax/Non-Tax', 'Department', 'FY'], 
                        values='Revenue', color='Tax/Non-Tax',
                        color_discrete_map={'Tax':'#4B78B2', 'Non-Tax':'#F6A756'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        sankey_df = df.groupby(['Tax/Non-Tax', 'Department'])['FY19'].sum().reset_index()
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=list(sankey_df['Tax/Non-Tax'].unique()) + list(sankey_df['Department'].unique())
            ),
            link=dict(
                source=[0]*len(sankey_df) + [1]*len(sankey_df),
                target=[i+2 for i in range(len(sankey_df))],
                value=sankey_df['FY19'].tolist()
            )
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        selected_fy = st.selectbox("Select Financial Year", fy_cols[::-1])
        prev_fy = fy_cols[fy_cols.index(selected_fy)-1]
        delta = df[selected_fy] - df[prev_fy]
        
        fig = go.Figure(go.Waterfall(
            x=df['Department'],
            y=delta,
            measure=["relative"]*len(df),
            connector={"line":{"color":"rgb(63, 63, 63)"}},
        ))
        st.plotly_chart(fig, use_container_width=True)

elif section == "📈 Trend Analysis":
    st.title("Historical Trend Analysis")
    
    selected_depts = st.multiselect("Select Departments", df['Department'].unique())
    if not selected_depts:
        st.info("Please select at least one department")
        st.stop()
    
    melt_df = df[df['Department'].isin(selected_depts)].melt(
        id_vars="Department", value_vars=fy_cols,
        var_name="Financial Year", value_name="Revenue"
    )
    
    fig = px.line(melt_df, x="Financial Year", y="Revenue", 
                 color="Department", markers=True,
                 title="Revenue Trend Analysis")
    st.plotly_chart(fig, use_container_width=True)

elif section == "🌳 Revenue Composition":
    st.title("Hierarchical Revenue Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_fy = st.selectbox("Select FY for Composition", fy_cols[::-1])
    with col2:
        depth_level = st.select_slider("Hierarchy Depth", options=[1,2,3])
    
    fig = px.sunburst(df, path=['Tax/Non-Tax', 'Department'], 
                     values=selected_fy, color='Tax/Non-Tax',
                     color_discrete_map={'Tax':'#4B78B2', 'Non-Tax':'#F6A756'},
                     maxdepth=depth_level)
    st.plotly_chart(fig, use_container_width=True)

elif section == "🔮 Forecasting":
    st.title("Revenue Forecasting Model")
    
    selected_dept = st.selectbox("Select Department", df['Department'].unique())
    
    # Prepare the time series data
    dept_data = df[df['Department'] == selected_dept][fy_cols].squeeze()
    if isinstance(dept_data, pd.DataFrame):
        dept_data = dept_data.iloc[0]  # Take first row if multiple rows match
    
    # Convert to time series with proper datetime index
    dates = pd.to_datetime([f'20{x[2:]}-03-31' for x in fy_cols])
    ts_data = pd.Series(dept_data.values, index=dates, name='Revenue')
    
    # ARIMA Forecasting
    st.subheader("ARIMA Model Configuration")
    col1, col2, col3 = st.columns(3)
    with col1: p = st.slider("AR Order (p)", 0, 3, 1)
    with col2: d = st.slider("Differencing (d)", 0, 2, 1)
    with col3: q = st.slider("MA Order (q)", 0, 3, 1)
    
    try:
        # Fit ARIMA model
        model = ARIMA(ts_data, order=(p, d, q))
        model_fit = model.fit()
        
        # Forecast
        forecast_steps = st.slider("Forecast Periods", 1, 5, 3)
        forecast = model_fit.get_forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=ts_data.index[-1] + pd.offsets.DateOffset(years=1), 
                                     periods=forecast_steps, freq='AS')
        
        # Plot results
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=ts_data.index, 
            y=ts_data,
            mode='lines+markers',
            name='Historical',
            line=dict(color='#1f77b4')
        ))
        
        # Forecasted values
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=forecast.predicted_mean,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#ff7f0e', dash='dash')
        ))
        
        # Confidence interval
        if hasattr(forecast, 'conf_int') and forecast.conf_int() is not None:
            conf_int = forecast.conf_int()
            upper = conf_int.iloc[:, 1]
            lower = conf_int.iloc[:, 0]
            
            # Create a single trace for the confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_index.tolist() + forecast_index.tolist()[::-1],
                y=pd.concat([upper, lower[::-1]]),
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name='Confidence Interval'
            ))
        else:
            st.warning("Could not calculate confidence intervals for this model.")
        
        fig.update_layout(
            title=f"{selected_dept} Revenue Forecast",
            xaxis_title="Year",
            yaxis_title="Revenue",
            hovermode="x unified",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display forecast metrics
        st.subheader("Forecast Metrics")
        metrics = {
            'AIC': f"{model_fit.aic:.2f}",
            'BIC': f"{model_fit.bic:.2f}",
            'HQIC': f"{model_fit.hqic:.2f}",
            'Forecast Start': forecast_index[0].strftime('%Y-%m-%d'),
            'Forecast End': forecast_index[-1].strftime('%Y-%m-%d')
        }
        st.table(pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value']))
        
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")
        st.warning("Try different ARIMA parameters or check your data.")
        if 'p' in locals() and 'd' in locals() and 'q' in locals():
            st.info(f"Current parameters - p: {p}, d: {d}, q: {q}")

elif section == "⚙️ Advanced Tools":
    st.title("Advanced Analytical Tools")
    
    tab1, tab2 = st.tabs(["Correlation Analysis", "Statistical Testing"])
    
    with tab1:
        st.subheader("Cross-Department Correlation")
        corr_matrix = df[fy_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, 
                       color_continuous_scale='RdBu_r',
                       labels=dict(x="FY", y="FY", color="Correlation"))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Stationarity Testing")
        selected_series = st.selectbox("Select Data Series", fy_cols)
        result = adfuller(df[selected_series])
        
        st.write(f"ADF Statistic: {result[0]:.4f}")
        st.write(f"p-value: {result[1]:.4f}")
        st.write("Critical Values:")
        for key, value in result[4].items():
            st.write(f"{key}: {value:.4f}")

# Download functionality
st.sidebar.download_button(
    label="📥 Export Current View",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"{selected_sheet}_analysis.csv",
    mime="text/csv"
)
