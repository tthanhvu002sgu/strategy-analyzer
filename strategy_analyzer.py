"""
Strategy Analyzer Dashboard — Streamlit
Phân tích toàn diện chiến lược giao dịch từ file backtest MT5.
Usage: streamlit run strategy_analyzer.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import os, warnings, glob, io

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False

# ============================================================
# GOOGLE DRIVE SYNC
# ============================================================
def get_drive_service():
    if not GOOGLE_DRIVE_AVAILABLE: return None
    try:
        if "gcp_service_account" in st.secrets:
            creds = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"], scopes=['https://www.googleapis.com/auth/drive']
            )
            return build('drive', 'v3', credentials=creds)
    except Exception as e:
        st.sidebar.error(f"Lỗi khởi tạo Google Drive: {e}")
    return None

def sync_drive(service, folder_id, local_dir):
    try:
        # Download from Drive
        results = service.files().list(q=f"'{folder_id}' in parents and trashed=false", fields="files(id, name)").execute()
        drive_files = {item['name']: item['id'] for item in results.get('files', [])}
        for name, file_id in drive_files.items():
            local_path = os.path.join(local_dir, name)
            if not os.path.exists(local_path):
                request = service.files().get_media(fileId=file_id)
                with io.FileIO(local_path, 'wb') as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done: _, done = downloader.next_chunk()
        
        # Upload new local files to Drive
        for f in glob.glob(os.path.join(local_dir, "*.*")):
            name = os.path.basename(f)
            if name not in drive_files:
                media = MediaFileUpload(f, resumable=True)
                service.files().create(body={'name': name, 'parents': [folder_id]}, media_body=media, fields='id').execute()
    except Exception as e:
        st.sidebar.error(f"Lỗi đồng bộ Drive: {e}")# ============================================================
# CONFIG
# ============================================================
BACKTEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest result")

st.set_page_config(page_title="Strategy Analyzer", layout="wide", page_icon="📊")

# ============================================================
# DATA LOADING
# ============================================================
def get_mt5_metric(raw_df, label_str, header_idx):
    limit = min(header_idx, 80)
    for r in range(limit):
        for c in range(len(raw_df.columns)):
            val = str(raw_df.iloc[r, c]).strip()
            if label_str.lower() in val.lower():
                for nc in range(c + 1, len(raw_df.columns)):
                    v = str(raw_df.iloc[r, nc]).strip()
                    if v not in ('nan', 'None', ''):
                        return v
    return None

@st.cache_data
def load_backtest(file_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if file_path.lower().endswith('.csv'):
            try: raw = pd.read_csv(file_path, header=None, encoding='utf-16le', sep='\t')
            except: raw = pd.read_csv(file_path, header=None)
        else:
            raw = pd.read_excel(file_path, engine='openpyxl', header=None)

    # Find Deals table
    deals_mask = raw[0].astype(str).str.strip() == 'Deals'
    if deals_mask.any():
        deals_start = raw[deals_mask].index[0]
    else:
        return None, None, None

    header_idx = deals_start + 1
    df = raw.iloc[header_idx + 1:].copy()
    df.columns = raw.iloc[header_idx].values

    # Clean columns
    col_map = {}
    for c in df.columns:
        cs = str(c).strip().lower()
        if cs == 'time': col_map[c] = 'Time'
        elif cs == 'deal': col_map[c] = 'Deal'
        elif cs == 'symbol': col_map[c] = 'Symbol'
        elif cs == 'type': col_map[c] = 'Type'
        elif cs == 'direction': col_map[c] = 'Direction'
        elif cs == 'volume': col_map[c] = 'Volume'
        elif cs == 'price': col_map[c] = 'Price'
        elif cs == 'profit': col_map[c] = 'Profit'
        elif cs == 'balance': col_map[c] = 'Balance'
        elif cs == 'swap': col_map[c] = 'Swap'
        elif cs == 'commission': col_map[c] = 'Commission'
        elif cs == 'comment': col_map[c] = 'Comment'
        elif cs == 'order': col_map[c] = 'Order'
    df.rename(columns=col_map, inplace=True)

    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df.dropna(subset=['Time'], inplace=True)

    for nc in ['Profit', 'Balance', 'Volume', 'Price', 'Swap', 'Commission']:
        if nc in df.columns:
            df[nc] = pd.to_numeric(df[nc], errors='coerce')

    # Filter only closed trades (direction=out or has profit != 0)
    if 'Direction' in df.columns:
        trades = df[df['Direction'].astype(str).str.strip().str.lower() == 'out'].copy()
    else:
        trades = df[df['Profit'].notna() & (df['Profit'] != 0)].copy()

    # Build entry info: match each "out" deal to its "in" deal via Order
    if 'Direction' in df.columns and 'Order' in df.columns:
        entries = df[df['Direction'].astype(str).str.strip().str.lower() == 'in'].copy()
        entry_map = entries.set_index('Order')[['Time', 'Price', 'Type']].rename(
            columns={'Time': 'OpenTime', 'Price': 'OpenPrice', 'Type': 'TradeType'})
        # Some orders may not exist, use left join
        if 'Order' in trades.columns:
            trades = trades.merge(entry_map, left_on='Order', right_index=True, how='left')

    # Duration
    if 'OpenTime' in trades.columns:
        trades['Duration'] = (trades['Time'] - trades['OpenTime']).dt.total_seconds() / 3600.0
    trades.reset_index(drop=True, inplace=True)

    # Extract header metrics
    metrics = {}
    for label, key in [('Total Net Profit:', 'net_profit'), ('Initial Deposit:', 'init_deposit'),
                        ('Profit Factor:', 'profit_factor'), ('Sharpe Ratio:', 'sharpe'),
                        ('Recovery Factor:', 'recovery_factor'), ('Expected Payoff:', 'expected_payoff'),
                        ('Total Trades:', 'total_trades')]:
        v = get_mt5_metric(raw, label, header_idx)
        if v:
            try: metrics[key] = float(v.replace(' ', '').replace(',', ''))
            except: metrics[key] = v

    # Equity DD
    dd_str = get_mt5_metric(raw, 'Equity Drawdown Maximal:', header_idx)
    if dd_str and '(' in dd_str:
        try: metrics['max_dd_pct'] = float(dd_str.split('(')[1].split('%')[0])
        except: pass

    wr_str = get_mt5_metric(raw, 'Profit Trades (% of total):', header_idx)
    if wr_str and '(' in wr_str:
        try: metrics['win_rate'] = float(wr_str.split('(')[1].split('%')[0])
        except: pass

    return trades, metrics, raw

# ============================================================
# METRICS COMPUTATION
# ============================================================
def compute_metrics(trades, mt5_metrics):
    profits = trades['Profit'].dropna()
    wins = profits[profits > 0]
    losses = profits[profits < 0]

    m = {}
    m['Total Trades'] = len(profits)
    m['Net Profit ($)'] = profits.sum()
    m['Win Rate (%)'] = mt5_metrics.get('win_rate', (len(wins)/len(profits)*100 if len(profits) > 0 else 0))
    m['Profit Factor'] = mt5_metrics.get('profit_factor', (wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0))
    m['Avg Win ($)'] = wins.mean() if len(wins) > 0 else 0
    m['Avg Loss ($)'] = losses.mean() if len(losses) > 0 else 0
    m['Avg R:R'] = abs(m['Avg Win ($)'] / m['Avg Loss ($)']) if m['Avg Loss ($)'] != 0 else 0
    m['Expectancy ($)'] = mt5_metrics.get('expected_payoff', profits.mean() if len(profits) > 0 else 0)
    m['Max DD (%)'] = mt5_metrics.get('max_dd_pct', 0)
    m['Sharpe Ratio'] = mt5_metrics.get('sharpe', 0)
    m['Recovery Factor'] = mt5_metrics.get('recovery_factor', 0)
    return m

# ============================================================
# CHARTS
# ============================================================
def chart_equity_dd(trades):
    if 'Balance' not in trades.columns: return None
    bal = trades[['Time', 'Balance']].dropna().copy()
    bal = bal.sort_values('Time')
    bal['Peak'] = bal['Balance'].cummax()
    bal['DD'] = bal['Balance'] - bal['Peak']
    bal['DD_pct'] = bal['DD'] / bal['Peak'] * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=bal['Time'], y=bal['Balance'], name='Equity',
                             line=dict(color='#00d4aa', width=2), fill='tozeroy',
                             fillcolor='rgba(0,212,170,0.1)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=bal['Time'], y=bal['Peak'], name='Peak',
                             line=dict(color='#555', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=bal['Time'], y=bal['DD_pct'], name='Drawdown %',
                             fill='tozeroy', line=dict(color='#ff4757', width=1),
                             fillcolor='rgba(255,71,87,0.3)'), row=2, col=1)
    fig.update_layout(height=500, template='plotly_dark', showlegend=True,
                      legend=dict(orientation='h', y=1.05),
                      margin=dict(l=50, r=20, t=30, b=30))
    fig.update_yaxes(title_text='Balance ($)', row=1, col=1)
    fig.update_yaxes(title_text='DD %', row=2, col=1)
    return fig

def chart_monthly_heatmap(trades):
    df = trades[['Time', 'Profit']].dropna().copy()
    df['Year'] = df['Time'].dt.year
    df['Month'] = df['Time'].dt.month
    pivot = df.groupby(['Year', 'Month'])['Profit'].sum().unstack(fill_value=0)
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    pivot.columns = [months[m-1] for m in pivot.columns]

    fig = px.imshow(pivot.values, x=pivot.columns, y=pivot.index.astype(str),
                    color_continuous_scale=[[0.0, '#ff4757'], [0.499, '#ffa502'], [0.5, '#00d4aa'], [1.0, '#008c72']],
                    color_continuous_midpoint=0, text_auto='.0f', aspect='auto')
    fig.update_layout(height=350, template='plotly_dark', margin=dict(l=50, r=20, t=30, b=30),
                      coloraxis_colorbar=dict(title='Profit $'))
    return fig

def chart_scatter_rr(trades):
    df = trades[['Profit']].dropna().copy()
    df['Trade #'] = range(1, len(df)+1)
    df['Color'] = np.where(df['Profit'] >= 0, 'Win', 'Loss')
    fig = px.scatter(df, x='Trade #', y='Profit', color='Color',
                     color_discrete_map={'Win': '#00d4aa', 'Loss': '#ff4757'},
                     opacity=0.6)
    fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.3)
    fig.update_layout(height=350, template='plotly_dark', margin=dict(l=50, r=20, t=30, b=30),
                      showlegend=True)
    return fig

def chart_profit_distribution(profits):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=profits, nbinsx=60, name='Profit',
                               marker_color='#00d4aa', opacity=0.7))
    mu, sigma = profits.mean(), profits.std()
    x_range = np.linspace(profits.min(), profits.max(), 200)
    pdf = stats.norm.pdf(x_range, mu, sigma) * len(profits) * (profits.max()-profits.min())/60
    fig.add_trace(go.Scatter(x=x_range, y=pdf, name='Normal Fit',
                             line=dict(color='#ffa502', width=2)))
    fig.add_vline(x=0, line_dash='dash', line_color='white', opacity=0.3)
    fig.update_layout(height=350, template='plotly_dark', margin=dict(l=50, r=20, t=30, b=30))
    return fig

def chart_hourly(trades):
    tcol = 'OpenTime' if 'OpenTime' in trades.columns and trades['OpenTime'].notna().any() else 'Time'
    if tcol not in trades.columns: return None
    df = trades[[tcol, 'Profit']].dropna(subset=[tcol, 'Profit']).copy()
    if df.empty: return None
    df['Hour'] = df[tcol].dt.hour
    grp = df.groupby('Hour')['Profit'].agg(['sum', 'count']).reset_index()
    grp.columns = ['Hour', 'Total Profit', 'Count']
    colors = ['#00d4aa' if x >= 0 else '#ff4757' for x in grp['Total Profit']]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=grp['Hour'], y=grp['Total Profit'], marker_color=colors, name='Profit'))
    fig.update_layout(height=350, template='plotly_dark', margin=dict(l=50, r=20, t=30, b=30),
                      xaxis_title='Hour of Day (Server Time)', yaxis_title='Total Profit ($)')
    return fig

def chart_dow(trades):
    tcol = 'OpenTime' if 'OpenTime' in trades.columns and trades['OpenTime'].notna().any() else 'Time'
    if tcol not in trades.columns: return None
    df = trades[[tcol, 'Profit']].dropna(subset=[tcol, 'Profit']).copy()
    if df.empty: return None
    df['DOW'] = df[tcol].dt.dayofweek
    grp = df.groupby('DOW')['Profit'].agg(['sum', 'count']).reset_index()
    grp.columns = ['DOW', 'Total Profit', 'Count']
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    grp['Day'] = grp['DOW'].map(lambda d: days[d])
    colors = ['#00d4aa' if x >= 0 else '#ff4757' for x in grp['Total Profit']]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=grp['Day'], y=grp['Total Profit'], marker_color=colors))
    fig.update_layout(height=350, template='plotly_dark', margin=dict(l=50, r=20, t=30, b=30),
                      xaxis_title='Day of Week', yaxis_title='Total Profit ($)')
    return fig

def chart_duration(trades):
    if 'Duration' not in trades.columns: return None
    df = trades[['Duration', 'Profit']].dropna()
    df = df[df['Duration'] > 0]
    if df.empty: return None
    df['Color'] = np.where(df['Profit'] >= 0, 'Win', 'Loss')
    fig = px.scatter(df, x='Duration', y='Profit', color='Color',
                     color_discrete_map={'Win': '#00d4aa', 'Loss': '#ff4757'},
                     opacity=0.5, labels={'Duration': 'Duration (hours)'})
    fig.update_layout(height=350, template='plotly_dark', margin=dict(l=50, r=20, t=30, b=30))
    return fig

# ============================================================
# ADVANCED ANALYSIS
# ============================================================
def run_ks_test(profits):
    stat, pval = stats.kstest(profits, 'norm', args=(profits.mean(), profits.std()))
    return stat, pval

def run_monte_carlo(profits, n_sims=10000, init_balance=5000):
    results = []
    profit_arr = profits.values
    n = len(profit_arr)
    for _ in range(n_sims):
        shuffled = np.random.choice(profit_arr, size=n, replace=True)
        equity = init_balance + np.cumsum(shuffled)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = dd.min()
        final = equity[-1]
        results.append({'final_equity': final, 'max_dd_pct': max_dd * 100})
    return pd.DataFrame(results)

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown("""
    <style>
    .metric-card {background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 12px;
        padding: 16px; text-align: center; border: 1px solid #333;}
    .metric-value {font-size: 28px; font-weight: bold; color: #00d4aa;}
    .metric-label {font-size: 13px; color: #888; margin-top: 4px;}
    .neg {color: #ff4757 !important;}
    </style>
    """, unsafe_allow_html=True)

    st.title("📊 Strategy Analyzer Dashboard")
    st.caption("Phân tích toàn diện chiến lược giao dịch từ file backtest MT5")

    # Ensure the directory exists
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    
    # ── GOOGLE DRIVE SYNC ──
    service = get_drive_service()
    drive_folder_id = st.secrets.get("drive_folder_id", "") if "drive_folder_id" in st.secrets else None
    
    if service and drive_folder_id:
        if st.sidebar.button("🔄 Đồng bộ dữ liệu với Drive"):
            with st.spinner("Đang đồng bộ..."):
                sync_drive(service, drive_folder_id, BACKTEST_DIR)
            st.sidebar.success("Đồng bộ hoàn tất!")
            
        # Optional: Auto-sync on startup could be added here, but button is safer
    
    st.sidebar.header("📥 Thêm Dữ Liệu Mới")
    uploaded_file = st.sidebar.file_uploader("Tải lên file Backtest (CSV, XLSX)", type=['csv', 'xlsx', 'xls'])
    if uploaded_file is not None:
        save_path = os.path.join(BACKTEST_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Đã lưu thành công: {uploaded_file.name}")
        # Auto-upload to Drive if available
        if service and drive_folder_id:
            with st.spinner("Đang lưu trữ đám mây..."):
                sync_drive(service, drive_folder_id, BACKTEST_DIR)

    st.sidebar.markdown("---")
    
    # File selector
    files = sorted(glob.glob(os.path.join(BACKTEST_DIR, "*.xlsx")) +
                   glob.glob(os.path.join(BACKTEST_DIR, "*.xls")) +
                   glob.glob(os.path.join(BACKTEST_DIR, "*.csv")), reverse=True)
    
    if not files:
        st.info("👋 Chào mừng bạn! Hệ thống chưa có dữ liệu.\n\n👉 Vui lòng sử dụng thanh công cụ bên trái (Sidebar) để **Tải lên file Backtest** (MT5 Report dạng Excel/CSV) và bắt đầu phân tích.")
        return

    file_names = [os.path.basename(f) for f in files]
    # Default to the newly uploaded file if there is one, else the first file
    default_idx = 0
    if uploaded_file is not None and uploaded_file.name in file_names:
        default_idx = file_names.index(uploaded_file.name)

    selected = st.selectbox("🗂️ Chọn file backtest để phân tích", file_names, index=default_idx)
    file_path = files[file_names.index(selected)]

    trades, mt5_metrics, raw_df = load_backtest(file_path)
    if trades is None or trades.empty:
        st.error("Không đọc được dữ liệu từ file. Kiểm tra lại định dạng.")
        return

    profits = trades['Profit'].dropna()
    m = compute_metrics(trades, mt5_metrics)
    init_bal = mt5_metrics.get('init_deposit', 5000)

    # ── STEP 1: CORE METRICS ─────────────────────────────────
    st.header("1️⃣ Chỉ Số Cơ Bản (Core Metrics)")
    cols = st.columns(6)
    items = [
        ("Net Profit", f"${m['Net Profit ($)']:,.2f}", m['Net Profit ($)'] >= 0),
        ("Win Rate", f"{m['Win Rate (%)']:.1f}%", m['Win Rate (%)'] >= 50),
        ("Profit Factor", f"{m['Profit Factor']:.2f}" if isinstance(m['Profit Factor'], float) else str(m['Profit Factor']), True),
        ("Max DD", f"{m['Max DD (%)']:.2f}%", False),
        ("Sharpe", f"{m['Sharpe Ratio']:.2f}" if isinstance(m['Sharpe Ratio'], float) else str(m['Sharpe Ratio']), True),
        ("Expectancy", f"${m['Expectancy ($)']:.2f}" if isinstance(m['Expectancy ($)'], float) else str(m['Expectancy ($)']), True),
    ]
    for col, (label, value, is_pos) in zip(cols, items):
        css = "" if is_pos else " neg"
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value{css}">{value}</div>
            <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

    cols2 = st.columns(4)
    items2 = [
        ("Total Trades", f"{m['Total Trades']}"),
        ("Avg Win", f"${m['Avg Win ($)']:.2f}"),
        ("Avg Loss", f"${m['Avg Loss ($)']:.2f}"),
        ("Avg R:R", f"{m['Avg R:R']:.2f}"),
    ]
    for col, (label, value) in zip(cols2, items2):
        col.metric(label, value)

    # ── STEP 2: EQUITY & DRAWDOWN ────────────────────────────
    st.header("2️⃣ Đường Cong Vốn & Drawdown")
    fig_eq = chart_equity_dd(trades)
    if fig_eq: st.plotly_chart(fig_eq, use_container_width=True)

    # ── Monthly Heatmap + Scatter ─────────────────────────────
    st.header("3️⃣ Trực Quan Hóa (Visualization)")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📅 Heatmap Lợi Nhuận Tháng/Năm")
        st.plotly_chart(chart_monthly_heatmap(trades), use_container_width=True)
    with c2:
        st.subheader("🎯 Scatter Plot Lệnh")
        st.plotly_chart(chart_scatter_rr(trades), use_container_width=True)

    # ── Time Analysis ─────────────────────────────────────────
    st.header("4️⃣ Phân Tích Thời Gian (Time-Series)")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("⏰ Lợi nhuận theo Giờ")
        fig_h = chart_hourly(trades)
        if fig_h: st.plotly_chart(fig_h, use_container_width=True)
    with c2:
        st.subheader("📆 Lợi nhuận theo Thứ")
        fig_d = chart_dow(trades)
        if fig_d: st.plotly_chart(fig_d, use_container_width=True)

    # ── Duration Analysis ─────────────────────────────────────
    fig_dur = chart_duration(trades)
    if fig_dur:
        st.subheader("⏱️ Thời gian giữ lệnh vs Profit")
        st.plotly_chart(fig_dur, use_container_width=True)

    # ── STEP 3: ADVANCED QUANT ────────────────────────────────
    st.header("5️⃣ Phân Tích Chuyên Sâu (Quant Insights)")

    # Distribution + KS Test
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 Phân Phối Lợi Nhuận & KS Test")
        st.plotly_chart(chart_profit_distribution(profits), use_container_width=True)
        ks_stat, ks_pval = run_ks_test(profits)
        if ks_pval < 0.05:
            st.warning(f"**KS Test**: D={ks_stat:.4f}, p={ks_pval:.4e} → Phân phối **KHÔNG phải Normal**. "
                       f"Chiến lược có thể phụ thuộc vào các lệnh \"đuôi béo\" (fat-tail).")
        else:
            st.success(f"**KS Test**: D={ks_stat:.4f}, p={ks_pval:.4e} → Phân phối gần Normal. "
                       f"Lợi nhuận đều đặn, ít phụ thuộc vào lệnh lớn bất thường.")

    # Monte Carlo
    with c2:
        st.subheader("🎲 Monte Carlo Simulation")
        with st.spinner("Đang chạy 10,000 mô phỏng..."):
            mc = run_monte_carlo(profits, n_sims=10000, init_balance=init_bal)

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(x=mc['final_equity'], nbinsx=80, marker_color='#7c4dff', opacity=0.7))
        fig_mc.add_vline(x=init_bal, line_dash='dash', line_color='#ff4757',
                         annotation_text=f'Vốn ban đầu: ${init_bal:,.0f}')
        fig_mc.update_layout(height=350, template='plotly_dark',
                             xaxis_title='Final Equity ($)', yaxis_title='Count',
                             margin=dict(l=50, r=20, t=30, b=30))
        st.plotly_chart(fig_mc, use_container_width=True)

        risk_of_ruin = (mc['final_equity'] <= init_bal * 0.5).mean() * 100
        median_eq = mc['final_equity'].median()
        p5 = mc['final_equity'].quantile(0.05)
        p95 = mc['final_equity'].quantile(0.95)
        worst_dd = mc['max_dd_pct'].min()

        mc_cols = st.columns(3)
        mc_cols[0].metric("Xác suất cháy TK (< 50% vốn)", f"{risk_of_ruin:.2f}%")
        mc_cols[1].metric("Equity trung vị", f"${median_eq:,.0f}")
        mc_cols[2].metric("Worst DD (MC)", f"{worst_dd:.1f}%")
        st.caption(f"📌 Khoảng tin cậy 90%: **${p5:,.0f}** — **${p95:,.0f}**")

    # ── STEP 6: REGIME & LOSS ATTRIBUTION ─────────────────────────
    st.header("6️⃣ Phân Tích Cấu Trúc Lỗ (Loss Attribution)")
    insights_loss = []
    
    if 'Type' in trades.columns:
        c1, c2 = st.columns(2)
        trades['RawType'] = trades['Type'].astype(str).str.lower().str.strip()
        
        if 'Direction' in trades.columns:
            buy_mask = trades['RawType'].isin(['sell', '1'])
            sell_mask = trades['RawType'].isin(['buy', '0'])
        else:
            buy_mask = trades['RawType'].isin(['buy', '0'])
            sell_mask = trades['RawType'].isin(['sell', '1'])
        
        long_trades = trades[buy_mask]
        short_trades = trades[sell_mask]
        
        long_profit = long_trades['Profit'].sum()
        short_profit = short_trades['Profit'].sum()
        
        fig_dir = go.Figure()
        fig_dir.add_trace(go.Bar(name='Long (Buy)', x=['Lợi nhuận'], y=[long_profit], marker_color='#00d4aa' if long_profit >= 0 else '#ff4757'))
        fig_dir.add_trace(go.Bar(name='Short (Sell)', x=['Lợi nhuận'], y=[short_profit], marker_color='#00d4aa' if short_profit >= 0 else '#ff4757'))
        
        fig_dir.update_layout(height=350, template='plotly_dark', barmode='group',
                              title='Lợi Nhuận Theo Hướng Giao Dịch', margin=dict(l=50, r=20, t=40, b=30))
        with c1:
            st.plotly_chart(fig_dir, use_container_width=True)
            
        # Monthly Regime Analysis
        df_monthly = trades.copy()
        df_monthly['YearMonth'] = df_monthly['Time'].dt.to_period('M')
        
        def regime_stats(g):
            wins = (g['Profit'] > 0).sum()
            total = len(g)
            wr = wins / total * 100 if total > 0 else 0
            
            if 'Direction' in trades.columns:
                buy_pnl = g[g['RawType'].isin(['sell', '1'])]['Profit'].sum()
                sell_pnl = g[g['RawType'].isin(['buy', '0'])]['Profit'].sum()
            else:
                buy_pnl = g[g['RawType'].isin(['buy', '0'])]['Profit'].sum()
                sell_pnl = g[g['RawType'].isin(['sell', '1'])]['Profit'].sum()
                
            return pd.Series({
                'Net Profit': g['Profit'].sum(),
                'Trades': total,
                'Win Rate %': wr,
                'Long PnL': buy_pnl,
                'Short PnL': sell_pnl
            })
            
        monthly_stats = df_monthly.groupby('YearMonth').apply(regime_stats, include_groups=False).reset_index()
        monthly_stats['YearMonth'] = monthly_stats['YearMonth'].astype(str)
        monthly_stats = monthly_stats.sort_values('Net Profit')
        
        with c2:
            st.markdown("**Top 5 Tháng Thua Lỗ Nặng Nhất (Phân Rã Cấu Trúc)**")
            st.dataframe(monthly_stats.head(5).style.background_gradient(cmap='RdYlGn', subset=['Net Profit', 'Win Rate %']), height=280)
            
        losing_months = monthly_stats[monthly_stats['Net Profit'] < 0]
        if len(losing_months) > 0:
            top_losers = losing_months.head(5)
            
            # Aggregate stats
            total_loss = top_losers['Net Profit'].sum()
            long_loss_sum = top_losers['Long PnL'].sum()
            short_loss_sum = top_losers['Short PnL'].sum()
            
            avg_trades_all = monthly_stats['Trades'].mean()
            avg_trades_losers = top_losers['Trades'].mean()
            
            insights_loss.append(f"🔍 **Phân tích tổng quan các tháng rủi ro nhất:** Tổng mức sụt giảm trong {len(top_losers)} tháng tệ nhất là **${total_loss:,.0f}**.")
            
            # Directional Bias Insight
            if long_loss_sum < 0 and short_loss_sum > 0:
                insights_loss.append(f"👉 **Điểm yếu ở chiều Buy**: Gần như toàn bộ thiệt hại đến từ các lệnh Long (Lỗ ${long_loss_sum:,.0f} so với mức Lãi ${short_loss_sum:,.0f} của lệnh Short). Điều này chứng tỏ EA rất nhạy cảm với các đợt sập giá mạnh (Downtrend regime). Khuyến nghị: **Tăng cường bộ lọc xu hướng giảm** (ví dụ cấm Buy khi giá nằm dưới EMA khung lớn).")
            elif short_loss_sum < 0 and long_loss_sum > 0:
                insights_loss.append(f"👉 **Điểm yếu ở chiều Sell**: Hầu hết thiệt hại đến từ các lệnh Short (Lỗ ${short_loss_sum:,.0f} so với mức Lãi ${long_loss_sum:,.0f} của lệnh Buy). EA đang chịu đòn nặng khi thị trường có nhịp tăng phi mã (Uptrend regime). Khuyến nghị: **Tránh bắt đỉnh** khi cấu trúc thị trường đang thể hiện lực nén tăng mạnh.")
            elif long_loss_sum < 0 and short_loss_sum < 0:
                if long_loss_sum < short_loss_sum * 2:
                    insights_loss.append(f"👉 **Điểm yếu đa chiều (Thiên về Buy)**: EA lỗ cả 2 đầu nhưng lệnh Buy mất tiền nhiều hơn gấp đôi lệnh Sell. Hệ thống thường xuyên vào lệnh sai nhịp ở các giai đoạn giảm giá.")
                elif short_loss_sum < long_loss_sum * 2:
                    insights_loss.append(f"👉 **Điểm yếu đa chiều (Thiên về Sell)**: EA lỗ cả 2 đầu nhưng lệnh Sell mất tiền nhiều hơn gấp đôi lệnh Buy.")
                else:
                    insights_loss.append(f"👉 **Điểm yếu đa chiều (Cân bằng)**: Các tháng lỗ phân bổ đều ở cả chiều Buy và Sell. Cấu trúc thị trường lúc này hoàn toàn không phù hợp với logic của EA, cắn Stop Loss cả hai bên.")

            # Volatility / Choppiness Insight
            if avg_trades_losers > avg_trades_all * 1.3:
                insights_loss.append(f"👉 **Nhận diện Regime (Whipsaw/Choppy)**: Số lượng lệnh trong các tháng lỗ cao bất thường (Trung bình {avg_trades_losers:.0f} lệnh/tháng so với mức trung bình {avg_trades_all:.0f} bình thường). Đây là dấu hiệu của thị trường dao động nhiễu, đi ngang biên độ hẹp cắn Stop Loss liên tục. Khuyến nghị: Thêm bộ lọc **ADX < 20** hoặc **ATR hẹp** để ngừng giao dịch.")
            elif avg_trades_losers < avg_trades_all * 0.7:
                insights_loss.append(f"👉 **Nhận diện Regime (Trend Expansion)**: Số lượng lệnh cực kỳ ít nhưng lỗ lại sâu. Nghĩa là thị trường chạy một mạch ngược hướng kỳ vọng, không có nhịp hồi để EA thoát lệnh. Khuyến nghị: Sử dụng bộ lọc động lượng (Momentum) để né các cú breakout giả hoặc cắt lỗ sớm.")
            else:
                insights_loss.append(f"👉 **Nhận diện Regime**: Tần suất giao dịch không thay đổi nhiều so với bình thường. Nguyên nhân lỗ chủ yếu do tỷ lệ Win Rate giảm mạnh trong các tháng này (trung bình chỉ đạt {(top_losers['Win Rate %'].mean()):.1f}%). Cần xem lại khoảng cách cắt lỗ (SL) có đang quá hẹp khiến giá dễ chạm tới hay không.")

    for ins in insights_loss:
        st.info(ins)

    # ── STEP 7: WFE ANALYSIS ─────────────────────────
    st.header("7️⃣ Đánh Giá Walk-Forward Efficiency (WFE)")
    st.markdown("""
    Đánh giá độ ổn định của chiến lược trong tương lai (Out-of-Sample) so với quá trình tối ưu (In-Sample).
    
    Công thức:
    $$WFE = \\frac{\\text{Lợi nhuận thực tế (Out-of-Sample)}}{\\text{Lợi nhuận kỳ vọng từ Backtest (In-Sample)}}$$
    """)
    
    wfe_tab1, wfe_tab2 = st.tabs(["🕒 Chia IS/OOS theo thời gian", "📝 Nhập thủ công In-Sample Profit"])
    
    with wfe_tab1:
        if 'Time' in trades.columns and len(trades) > 0:
            min_date = trades['Time'].min().date()
            max_date = trades['Time'].max().date()
            
            if min_date < max_date:
                split_date = st.slider(
                    "Chọn ngày bắt đầu Out-of-Sample", 
                    min_value=min_date, 
                    max_value=max_date, 
                    value=min_date + (max_date - min_date)//2
                )
                
                is_trades = trades[trades['Time'].dt.date < split_date]
                oos_trades = trades[trades['Time'].dt.date >= split_date]
                
                is_profit = is_trades['Profit'].sum()
                oos_profit = oos_trades['Profit'].sum()
                
                is_days = (split_date - min_date).days
                oos_days = (max_date - split_date).days
                
                wfe = oos_profit / is_profit if is_profit > 0 else 0
                
                wc1, wc2, wc3, wc4 = st.columns(4)
                wc1.metric("Lợi nhuận In-Sample (IS)", f"${is_profit:,.2f}", f"{len(is_trades)} lệnh ({is_days} ngày)")
                wc2.metric("Lợi nhuận Out-of-Sample (OOS)", f"${oos_profit:,.2f}", f"{len(oos_trades)} lệnh ({oos_days} ngày)")
                
                if is_days > 0 and oos_days > 0 and is_profit > 0:
                    is_annual = is_profit / is_days * 365
                    oos_annual = oos_profit / oos_days * 365
                    annual_wfe = oos_annual / is_annual
                    wc3.metric("WFE (Tuyệt đối)", f"{wfe*100:.1f}%")
                    wc4.metric("WFE (Thường niên - Annualized)", f"{annual_wfe*100:.1f}%")
                    final_wfe = annual_wfe
                else:
                    wc3.metric("WFE (Tuyệt đối)", f"{wfe*100:.1f}%")
                    final_wfe = wfe
                
                if is_profit > 0:
                    if final_wfe >= 0.5:
                        st.success("✅ **WFE Khả quan (>= 50%)**: Chiến lược duy trì được lợi thế giao dịch trong tập dữ liệu Out-of-Sample. Ít có rủi ro Overfitting.")
                    elif final_wfe > 0:
                        st.warning("⚠️ **WFE Thấp (< 50%)**: Lợi nhuận OOS sụt giảm mạnh so với IS. Dấu hiệu của việc Curve-fitting (Quá khớp dữ liệu quá khứ).")
                    else:
                        st.error("❌ **WFE Âm**: Chiến lược thua lỗ trong giai đoạn Out-of-Sample. Hệ thống đã phá vỡ hoàn toàn và không nên giao dịch thực tế.")
                else:
                    st.info("Vui lòng đảm bảo Lợi nhuận In-Sample > 0 để tính toán WFE hợp lệ.")
            else:
                st.info("Dữ liệu không đủ số ngày để chia In-Sample và Out-of-Sample.")
                
    with wfe_tab2:
        st.info("Sử dụng lựa chọn này nếu file đang load là file kết quả Out-of-Sample, và bạn đã biết mức lợi nhuận của giai đoạn In-Sample trước đó.")
        expected_is_profit = st.number_input("Lợi nhuận kỳ vọng từ Backtest (In-Sample) ($)", min_value=0.0, value=1000.0, step=100.0)
        actual_oos_profit = m['Net Profit ($)']
        
        manual_wfe = actual_oos_profit / expected_is_profit if expected_is_profit > 0 else 0
        
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("In-Sample Profit (Kỳ vọng)", f"${expected_is_profit:,.2f}")
        mc2.metric("Out-of-Sample Profit (Thực tế từ file)", f"${actual_oos_profit:,.2f}")
        mc3.metric("WFE", f"{manual_wfe*100:.1f}%")
        
        if expected_is_profit > 0:
            if manual_wfe >= 0.5:
                st.success("✅ **WFE Khả quan (>= 50%)**: Chiến lược hoạt động tốt trên tập dữ liệu chưa từng thấy.")
            elif manual_wfe > 0:
                st.warning("⚠️ **WFE Thấp (< 50%)**: Hiệu suất giảm đáng kể. Cần cẩn trọng rủi ro Overfitting.")
            else:
                st.error("❌ **WFE Âm**: Chiến lược thua lỗ trong OOS.")

    # ── INSIGHTS SUMMARY ──────────────────────────────────────
    st.header("8️⃣ 💡 Tổng Kết Hiệu Suất")
    insights = []
    if isinstance(m['Profit Factor'], float) and m['Profit Factor'] > 1.5:
        insights.append("✅ **Profit Factor > 1.5**: Chiến lược có lợi thế rõ ràng.")
    elif isinstance(m['Profit Factor'], float) and m['Profit Factor'] < 1.0:
        insights.append("❌ **Profit Factor < 1.0**: Chiến lược đang THUA ròng. Cần xem lại logic.")
    if m['Max DD (%)'] > 30:
        insights.append("⚠️ **Max DD > 30%**: Rủi ro sụt giảm vốn quá cao. Cân nhắc giảm lot hoặc thêm filter.")
    if m['Avg R:R'] > 1.5:
        insights.append("✅ **R:R trung bình > 1.5**: Chiến lược cho phép win rate thấp mà vẫn có lãi.")
    elif m['Avg R:R'] < 1.0:
        insights.append("⚠️ **R:R < 1.0**: Mỗi lệnh thua lớn hơn lệnh thắng. Cần win rate cao để bù đắp.")
    if risk_of_ruin > 5:
        insights.append(f"🔴 **Risk of Ruin = {risk_of_ruin:.1f}%**: Xác suất cháy tài khoản đáng lo ngại.")
    else:
        insights.append(f"🟢 **Risk of Ruin = {risk_of_ruin:.1f}%**: Xác suất cháy tài khoản thấp.")
    if ks_pval < 0.05:
        insights.append("📊 **Fat-tail detected**: Profit phụ thuộc vào một số lệnh lớn bất thường. "
                        "Nếu mất các lệnh này, hiệu suất sẽ giảm đáng kể.")

    # Time insights
    tcol = 'OpenTime' if 'OpenTime' in trades.columns and trades['OpenTime'].notna().any() else 'Time'
    if tcol in trades.columns:
        tdf = trades[[tcol, 'Profit']].dropna(subset=[tcol, 'Profit']).copy()
        tdf['Hour'] = tdf[tcol].dt.hour
        hour_profit = tdf.groupby('Hour')['Profit'].sum()
        if not hour_profit.empty:
            worst_hour = hour_profit.idxmin()
            best_hour = hour_profit.idxmax()
            if hour_profit[worst_hour] < 0:
                insights.append(f"⏰ **Giờ thua lỗ nhiều nhất**: {worst_hour}:00 (${hour_profit[worst_hour]:,.0f}). "
                               f"Cân nhắc tạo bộ lọc thời gian để tránh phiên này.")
            insights.append(f"⏰ **Giờ lãi nhiều nhất**: {best_hour}:00 (${hour_profit[best_hour]:,.0f}).")

    for ins in insights:
        st.markdown(ins)

if __name__ == '__main__':
    from streamlit.runtime import exists
    if not exists():
        import sys
        import subprocess
        print("Starting Streamlit app...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", sys.argv[0]] + sys.argv[1:])
    else:
        main()
