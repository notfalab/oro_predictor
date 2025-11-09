import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import time
from datetime import datetime

# --- Importar nuestros módulos ---
from modules import data_fetcher, preprocessor, predictor
from modules import multi_timeframe

# --- 0. CSS PERSONALIZADO (¡MODO OSCURO TOTAL!) ---
# Esto inyecta CSS para hacer que todo el fondo sea negro
st.markdown(
    """
    <style>
    /* Cuerpo principal de la app */
    body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #000000; /* Negro Puro */
        color: #FFFFFF; /* Texto blanco para legibilidad */
    }
    
    /* Barra lateral */
    [data-testid="stSidebar"] {
        background-color: #000000; /* Negro Puro */
    }
    
    /* Asegurar que los encabezados (h1, h2, h3) sean blancos */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }
    
    /* Hacer visibles los bordes del st.container */
    [data-testid="stVerticalBlockBorderWrapper"] {
         border-color: #333333; /* Borde gris oscuro para los contenedores */
    }
    
    /* Asegurar que las leyendas (captions) sean visibles */
    [data-testid="stCaptionContainer"] {
        color: #AAAAAA; /* Gris claro para las leyendas */
    }
    
    /* Cambiar el color del texto de "info" (el veredicto) */
    [data-testid="stInfo"] {
        background-color: #222222; /* Fondo gris muy oscuro */
        color: #FFFFFF; /* Texto blanco */
    }
    
    </style>
    """,
    unsafe_allow_html=True
)
# --- FIN DE CSS ---


# --- 1. CONFIGURACIÓN DE CACHÉ ---
@st.cache_data(ttl=600)
def get_cached_macro_data():
    return data_fetcher.get_macro_and_sentiment()

@st.cache_data(ttl=600)
def get_cached_weekly_data():
    return data_fetcher.get_weekly_data()

@st.cache_data(ttl=600)
def get_cached_daily_data():
    return data_fetcher.get_daily_data()

@st.cache_data(ttl=600)
def get_cached_h4_data():
    return data_fetcher.get_h4_data()

# --- 2. Configuración de la Página ---
st.set_page_config(
    page_title="MIDAS", 
)

# --- 3. Inicialización del Estado de la Sesión ---
def initialize_session():
    if 'data' not in st.session_state:
        st.session_state.data = data_fetcher.get_initial_historical_data(limit=500)
    if 'run_live' not in st.session_state:
        st.session_state.run_live = False
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()

initialize_session()


# --- 4. Título ---
# (st.title() ha sido eliminado)
st.caption(f"Live Analysis. Last M1 Update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")


# --- 5. Sidebar Controls ---
with st.sidebar:
    st.image("midas_logo.png", use_container_width=True) # Logo
    st.header("Dashboard Controls")
    st.header("Live Analysis Mode")
    
    if st.session_state.run_live:
        if st.button("⏹️ Stop Live Analysis", type="primary"):
            st.session_state.run_live = False
            st.rerun()
    else:
        if st.button("▶️ Start Live Analysis"):
            st.session_state.run_live = True
            st.rerun()
            
    refresh_rate = st.slider(
        "Refresh Rate (seconds)", 1, 10, 3,
        help="Time between each new M1 data tick."
    )
    
    st.caption("Live Mode updates the M1 chart and the System Verdict.")

# --- 6. Core Data & Prediction Logic ---
macro_data = get_cached_macro_data()
df_w = get_cached_weekly_data()
df_d = get_cached_daily_data()
df_h4 = get_cached_h4_data()
weekly_analysis = multi_timeframe.analyze_weekly(df_w)
daily_h4_analysis = multi_timeframe.analyze_daily_h4(df_d, df_h4)
raw_data_m1 = st.session_state.data
df_clean_m1 = preprocessor.clean_data(raw_data_m1)
df_featured_m1 = preprocessor.create_technical_features(df_clean_m1)
model_input_m1 = preprocessor.prepare_data_for_model(df_featured_m1)
prediction = predictor.make_short_term_prediction(
    model_input_m1, 
    macro_data, 
    weekly_analysis, 
    daily_h4_analysis
)

# --- 7. The Main Dashboard Display ---
if not df_featured_m1.empty:
    last_row_m1 = df_featured_m1.iloc[-1]
    last_price = last_row_m1['close']
    prev_price = df_featured_m1.iloc[-2]['close']
    price_change = last_price - prev_price
    last_rsi_m1 = last_row_m1['RSI_14'] if 'RSI_14' in last_row_m1 else 50
else:
    last_price, price_change, last_rsi_m1 = 0, 0, 50

# --- STEP 3: ENTRY VERDICT (INTRADAY) ---
st.subheader("Step 3: Entry Signal ('Confluence Model' v3)")
fund_go = "Neutral"
if macro_data.get('real_yield', 0) < 0: fund_go = "Bullish"
elif macro_data.get('real_yield', 0) > 1.0: fund_go = "Bearish"
tech_go = "Neutral"
if "Bullish" in weekly_analysis.get('verdict', '') and daily_h4_analysis.get('h4_rsi_status', '') != "Overbought":
    tech_go = "Bullish"
elif "Bearish" in weekly_analysis.get('verdict', '') and daily_h4_analysis.get('h4_rsi_status', '') != "Oversold":
    tech_go = "Bearish"
ml_signal = prediction.get("signal", "N/A")

# --- Final Combined Verdict ---
final_verdict = "HOLD / WAIT"
final_recommendation = "Context (P1/P2) is not aligned with the intraday signal (P3)."
if ml_signal == "Bullish" and fund_go == "Bullish" and tech_go == "Bullish":
    final_verdict = "STRONG BUY (CONFLUENCE)"
    final_recommendation = "P1 (Fundamental), P2 (Technical), and P3 (ML) are all aligned BULLISH."
elif ml_signal == "Bearish" and fund_go == "Bearish" and tech_go == "Bearish":
    final_verdict = "STRONG SELL (CONFLUENCE)"
    final_recommendation = "P1 (Fundamental), P2 (Technical), and P3 (ML) are all aligned BEARISH."
elif ml_signal == "Bullish" and (fund_go == "Bullish" or tech_go == "Bullish"):
    final_verdict = "Buy (Weak)"
    final_recommendation = "Bullish ML Signal with partial support (Technical or Fundamental)."
elif ml_signal == "Bearish" and (fund_go == "Bearish" or tech_go == "Bearish"):
    final_verdict = "Sell (Weak)"
    final_recommendation = "Bearish ML Signal with partial support (Technical or Fundamental)."

st.header(f"System Verdict: {final_verdict}")
st.info(f"**Confluence Analysis:** {final_recommendation}")
cols = st.columns(3)
cols[0].metric("Current XAU/USD Price", f"${last_price:.2f}", f"${price_change:.2f} (tick)")
cols[1].metric("ML Model Signal (M1)", prediction.get("signal", "N/A"), help=f"Confidence: {prediction.get('confidence', 'N/A')}")
cols[2].metric("RSI (M1)", f"{last_rsi_m1:.2f}")
st.divider()

# --- Analysis Containers ---
st.subheader("Step 1: Fundamental Context")
with st.container(border=True):
    st.markdown(f"**Fundamental Verdict (Long Term): {macro_data['verdict']}**")
    m_cols = st.columns(4)
    m_cols[0].metric("Fed Interest Rate", f"{macro_data['fed_interest_rate']:.2f}%")
    m_cols[1].metric("Inflation (CPI)", f"{macro_data['inflation_cpi']:.2f}%")
    m_cols[2].metric("📈 Real Yield (Fed-CPI)", f"{macro_data['real_yield']:.2f}%", help="Yields > 1% are Bearish for Gold. Yields < 0% are Bullish.")
    m_cols[3].metric("💵 Dollar Index (DXY)", f"{macro_data['usd_index']:.2f}", help="A strong DXY (typically > 104) puts pressure on Gold.")

st.subheader("Step 2: Technical Direction")
with st.container(border=True):
    pp = daily_h4_analysis.get('pivot_points', {})
    vpoc = daily_h4_analysis.get('vpoc', 0)
    w_cols = st.columns(3)
    w_cols[0].metric("Retail Resistance (R1)", f"${pp.get('R1', 0):.2f}")
    w_cols[0].metric("Soporte Retail (S1)", f"${pp.get('S1', 0):.2f}")
    w_cols[1].metric("Institutional Zone (VPoC 30D)", f"${vpoc:.2f}", help="Volume Point of Control: The price with the most traded volume.")
    w_cols[2].metric("Weekly Context (SMA 200)", weekly_analysis.get('verdict', 'N/A'))
    w_cols[2].metric("RSI Status (4-Hour)", daily_h4_analysis.get('h4_rsi_status', 'N/A'), help=f"RSI Value: {daily_h4_analysis.get('h4_rsi_value', 0):.2f}")

# --- Interactive Chart (Plotly) ---
st.subheader("Price Chart (1-Minute)")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df_featured_m1.index, open=df_featured_m1['open'],
    high=df_featured_m1['high'], low=df_featured_m1['low'],
    close=df_featured_m1['close'], name='XAU/USD Price'
))
if 'EMA_14' in df_featured_m1.columns:
    fig.add_trace(go.Scatter(x=df_featured_m1.index, y=df_featured_m1['EMA_14'], line=dict(color='cyan', width=1.5), name='EMA 14'))
if 'SMA_200' in df_featured_m1.columns:
    fig.add_trace(go.Scatter(x=df_featured_m1.index, y=df_featured_m1['SMA_200'], line=dict(color='orange', width=2, dash='dot'), name='SMA 200'))
bb_cols_exist = all(col in df_featured_m1.columns for col in ['BBU_20_2.0', 'BBL_20_2.0'])
if bb_cols_exist:
    fig.add_trace(go.Scatter(x=df_featured_m1.index, y=df_featured_m1['BBU_20_2.0'], line=dict(color='gray', width=1, dash='dash'), name='Upper BB'))
    fig.add_trace(go.Scatter(x=df_featured_m1.index, y=df_featured_m1['BBL_20_2.0'], line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', name='Lower BB'))

# --- CAMBIO DE TEMA DEL GRÁFICO ---
# El tema "plotly_dark" se ve bien sobre negro.
fig.update_layout(title="Live Technical Analysis (M1)", yaxis_title="Price (USD)",
                  xaxis_rangeslider_visible=False, height=550, template="plotly_dark",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

# --- 8. LIVE UPDATE LOOP ---
if st.session_state.run_live:
    new_data = data_fetcher.fetch_latest_data(st.session_state.data)
    st.session_state.data = new_data
    st.session_state.last_update = datetime.now()
    time.sleep(refresh_rate)
    st.rerun()

# --- Chart & Raw Data (at the end) ---
st.plotly_chart(fig, use_container_width=True) 
with st.expander("View Raw Data (Last 10 Ticks M1)"):
    st.dataframe(df_featured_m1.tail(10).sort_index(ascending=False))