"""
scripts/create_dashboard.py
FastAPI সার্ভার সহ AI Trading Signals ড্যাশবোর্ড
Render Web Service-এ সরাসরি ডিপ্লয় করার জন্য
✅ API Endpoints + HTML Dashboard + Hugging Face থেকে CSV লোড
"""

import os
import sys
import json
import io
import csv
import requests
from pathlib import Path

# =========================================================
# কনফিগারেশন
# =========================================================
OUTPUT_DIR = "./output/dashboard"
HF_REPO = "ahashanahmed/csv"
CSV_FILENAME = "FINAL_AI_SIGNALS.csv"
HF_CSV_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{CSV_FILENAME}"

PORT = int(os.environ.get("PORT", 8000))

# =========================================================
# FastAPI অ্যাপ
# =========================================================
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI(title="AI Trading Signals API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# CSV থেকে ডেটা লোড (Hugging Face)
# =========================================================
def load_csv_data():
    """Hugging Face থেকে CSV ডাউনলোড করে DataFrame রিটার্ন করে"""
    try:
        response = requests.get(HF_CSV_URL, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        # NaN ফিল্ড ফিল্টার
        df = df.where(pd.notnull(df), None)
        return df
    except Exception as e:
        print(f"⚠️ CSV Load Error: {e}")
        return pd.DataFrame()

# =========================================================
# API ROUTES
# =========================================================

@app.get("/api/health")
async def health_check():
    """হেলথ চেক"""
    return {
        "status": "ok",
        "timestamp": pd.Timestamp.now().isoformat(),
        "csv_source": HF_CSV_URL
    }

@app.get("/api/signals")
async def get_signals(
    signal: str = Query(None, description="Filter by signal (BUY, SELL, HOLD)"),
    symbol: str = Query(None, description="Filter by symbol"),
    min_score: float = Query(0, description="Minimum combined score"),
    limit: int = Query(100, description="Max results"),
    offset: int = Query(0, description="Pagination offset")
):
    """সব সিগন্যাল JSON API"""
    df = load_csv_data()
    
    if df.empty:
        return JSONResponse({"error": "No data available"}, status_code=503)
    
    # ফিল্টার
    if signal:
        df = df[df['final_signal'].str.contains(signal, case=False, na=False)]
    if symbol:
        df = df[df['symbol'].str.contains(symbol, case=False, na=False)]
    if min_score > 0:
        df = df[df['final_combined_score'] >= min_score]
    
    # সর্ট
    df = df.sort_values('final_combined_score', ascending=False)
    
    # পেজিনেশন
    total = len(df)
    df = df.iloc[offset:offset + limit]
    
    return {
        "total": total,
        "count": len(df),
        "offset": offset,
        "limit": limit,
        "data": df.to_dict(orient='records')
    }

@app.get("/api/signals/{symbol}")
async def get_signal_by_symbol(symbol: str):
    """নির্দিষ্ট সিম্বলের সিগন্যাল"""
    df = load_csv_data()
    
    if df.empty:
        return JSONResponse({"error": "No data available"}, status_code=503)
    
    match = df[df['symbol'] == symbol]
    if len(match) == 0:
        return JSONResponse({"error": f"Symbol '{symbol}' not found"}, status_code=404)
    
    return match.iloc[0].to_dict()

@app.get("/api/stats")
async def get_stats():
    """পরিসংখ্যান"""
    df = load_csv_data()
    
    if df.empty:
        return JSONResponse({"error": "No data available"}, status_code=503)
    
    return {
        "total_signals": len(df),
        "buy_signals": len(df[df['final_signal'].str.contains('BUY', na=False)]),
        "sell_signals": len(df[df['final_signal'].str.contains('SELL', na=False)]),
        "hold_signals": len(df[df['final_signal'].str.contains('HOLD', na=False)]),
        "avg_score": round(df['final_combined_score'].mean(), 2) if 'final_combined_score' in df.columns else 0,
        "max_score": round(df['final_combined_score'].max(), 2) if 'final_combined_score' in df.columns else 0,
        "elliott_waves": df['elliott_current_wave'].value_counts().to_dict() if 'elliott_current_wave' in df.columns else {},
        "model_availability": df['model_availability'].value_counts().to_dict() if 'model_availability' in df.columns else {},
        "top_symbols": df.nlargest(5, 'final_combined_score')[['symbol', 'final_combined_score', 'final_signal']].to_dict(orient='records'),
    }

# =========================================================
# HTML ড্যাশবোর্ড
# =========================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 AI Trading Signals Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0a0a0f; color: #e0e0e0; line-height: 1.6; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        
        .header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); padding: 30px 0; text-align: center; border-bottom: 2px solid #00d4ff; }
        .header h1 { font-size: 2.5em; margin-bottom: 5px; background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6b6b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .header p { color: #888; font-size: 1.1em; }
        
        .header-stats { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-top: 20px; }
        .stat-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 15px 25px; text-align: center; min-width: 120px; }
        .stat-card.buy { border-left: 3px solid #00ff88; }
        .stat-card.sell { border-left: 3px solid #ff4757; }
        .stat-label { display: block; font-size: 0.8em; color: #888; margin-bottom: 5px; }
        .stat-value { display: block; font-size: 1.8em; font-weight: bold; color: #fff; }
        
        .filters { background: #111122; padding: 20px 0; border-bottom: 1px solid #222; position: sticky; top: 0; z-index: 100; }
        .filter-row { display: flex; gap: 15px; flex-wrap: wrap; align-items: center; }
        .filter-group { display: flex; align-items: center; gap: 8px; }
        .filter-group label { color: #aaa; font-size: 0.9em; white-space: nowrap; }
        .filter-group input[type="text"], .filter-group select { background: #1a1a2e; border: 1px solid #333; color: #fff; padding: 8px 12px; border-radius: 5px; font-size: 0.9em; }
        .filter-group input[type="text"] { width: 160px; }
        .filter-group input[type="range"] { width: 80px; accent-color: #00d4ff; }
        .btn-reset, .btn-export { background: #333; color: #fff; border: 1px solid #555; padding: 8px 16px; border-radius: 5px; cursor: pointer; font-size: 0.9em; margin-top: 10px; }
        .btn-export { background: #0f3460; border-color: #00d4ff; margin-left: 10px; }
        
        .charts { padding: 30px 0; }
        .chart-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }
        .chart-card { background: #111122; border: 1px solid #222; border-radius: 10px; padding: 20px; }
        .chart-card h3 { margin-bottom: 15px; color: #aaa; font-size: 1em; text-align: center; }
        .chart-card canvas { max-height: 250px; }
        
        .top-picks { padding: 20px 0; }
        .top-picks h2 { margin-bottom: 15px; color: #00ff88; }
        .top-picks-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 15px; }
        .pick-card { background: linear-gradient(135deg, #0a2a0a, #0f1f0f); border: 1px solid #00ff88; border-radius: 10px; padding: 15px; }
        .pick-card strong { color: #00ff88; font-size: 1.2em; }
        .pick-card .pick-details { margin-top: 8px; font-size: 0.85em; color: #aaa; }
        .pick-card .pick-details span { display: inline-block; margin-right: 12px; }
        .pick-card .pick-score { font-size: 2em; font-weight: bold; color: #00d4ff; }
        
        .table-section { padding: 20px 0 40px; }
        .table-container { overflow-x: auto; background: #111122; border: 1px solid #222; border-radius: 10px; }
        #signalTable { width: 100%; border-collapse: collapse; font-size: 0.85em; }
        #signalTable thead { background: #1a1a2e; position: sticky; top: 0; }
        #signalTable th { padding: 12px 8px; text-align: left; color: #00d4ff; font-weight: 600; border-bottom: 2px solid #333; white-space: nowrap; }
        #signalTable td { padding: 10px 8px; border-bottom: 1px solid #1a1a2e; white-space: nowrap; }
        #signalTable tbody tr:hover { background: rgba(0,212,255,0.05); }
        
        .signal-strong-buy { color: #00ff88; font-weight: bold; }
        .signal-buy { color: #00cc66; font-weight: bold; }
        .signal-watch { color: #ffa500; }
        .signal-hold { color: #ffd700; }
        .signal-sell { color: #ff4757; }
        .signal-strong-sell { color: #ff0000; font-weight: bold; }
        
        footer { background: #0a0a0f; text-align: center; padding: 20px; color: #555; font-size: 0.85em; border-top: 1px solid #222; }
        
        @media (max-width: 768px) { .header h1 { font-size: 1.8em; } .filter-row { flex-direction: column; } .chart-row { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <header class="header">
        <div class="container">
            <h1>🤖 AI Trading Signals Dashboard</h1>
            <p>LLM + XGBoost + PPO + Agentic Loop + Elliott Wave Combined</p>
            <div class="header-stats" id="headerStats">
                <div class="stat-card"><span class="stat-label">Total Signals</span><span class="stat-value" id="totalSignals">-</span></div>
                <div class="stat-card buy"><span class="stat-label">🟢 Buy</span><span class="stat-value" id="buySignals">-</span></div>
                <div class="stat-card sell"><span class="stat-label">🔴 Sell</span><span class="stat-value" id="sellSignals">-</span></div>
                <div class="stat-card"><span class="stat-label">📊 Avg Score</span><span class="stat-value" id="avgScore">-</span></div>
                <div class="stat-card"><span class="stat-label">🌊 Elliott</span><span class="stat-value" id="elliottWaves">-</span></div>
            </div>
        </div>
    </header>

    <section class="filters">
        <div class="container">
            <div class="filter-row">
                <div class="filter-group"><label>🔍 Symbol:</label><input type="text" id="searchSymbol" placeholder="Search..." onkeyup="applyFilters()"></div>
                <div class="filter-group"><label>📈 Signal:</label><select id="filterSignal" onchange="applyFilters()"><option value="ALL">All</option><option value="STRONG BUY">🔥 STRONG BUY</option><option value="BUY">✅ BUY</option><option value="WATCH">👀 WATCH</option><option value="HOLD">⏳ HOLD</option><option value="SELL">❌ SELL</option><option value="STRONG SELL">💀 STRONG SELL</option></select></div>
                <div class="filter-group"><label>⭐ Score ≥</label><input type="range" id="minScore" min="0" max="100" value="0" oninput="document.getElementById('minScoreValue').textContent=this.value;applyFilters()"><span id="minScoreValue">0</span></div>
                <button class="btn-reset" onclick="resetFilters()">🔄 Reset</button>
                <button class="btn-export" onclick="exportCSV()">📥 Export CSV</button>
            </div>
        </div>
    </section>

    <section class="charts">
        <div class="container">
            <div class="chart-row">
                <div class="chart-card"><h3>📈 Signal Distribution</h3><canvas id="signalChart"></canvas></div>
                <div class="chart-card"><h3>🤖 Model Availability</h3><canvas id="modelChart"></canvas></div>
                <div class="chart-card"><h3>🌊 Elliott Wave</h3><canvas id="elliottChart"></canvas></div>
            </div>
        </div>
    </section>

    <section class="top-picks">
        <div class="container"><h2>🔥 Top BUY Picks</h2><div class="top-picks-grid" id="topPicksGrid"></div></div>
    </section>

    <section class="table-section">
        <div class="container">
            <h2>📊 All Signals (Total: <span id="tableCount">0</span>)</h2>
            <div class="table-container">
                <table id="signalTable"><thead><tr><th>#</th><th>Symbol</th><th>Price</th><th>Final Signal</th><th>Score</th><th>LLM</th><th>LLM%</th><th>XGB</th><th>AUC</th><th>PPO</th><th>Agentic</th><th>Elliott</th><th>Entry</th><th>SL</th><th>TP</th><th>R:R</th><th>Models</th></tr></thead><tbody id="tableBody"></tbody></table>
            </div>
        </div>
    </section>

    <footer><p>🤖 AI Trading System | Data: {HF_REPO}</p></footer>

    <script>
        const API_BASE = window.location.origin + '/api';
        const signalColors = {{'🔥 STRONG BUY': 'signal-strong-buy','✅ BUY': 'signal-buy','👀 WATCH (Near BUY)': 'signal-watch','⏳ HOLD': 'signal-hold','⚠️ WATCH (Near SELL)': 'signal-watch','❌ SELL': 'signal-sell','💀 STRONG SELL': 'signal-strong-sell'}};
        let allSignals = [];
        let filteredSignals = [];
        let signalChartInstance, modelChartInstance, elliottChartInstance;

        async function loadData() {{
            try {{
                const response = await fetch(`${{API_BASE}}/signals?limit=1000`);
                const json = await response.json();
                allSignals = json.data || [];
                filteredSignals = [...allSignals];
                updateDashboard();
            }} catch(e) {{ console.error(e); }}
        }}

        function updateDashboard() {{
            document.getElementById('totalSignals').textContent = filteredSignals.length;
            document.getElementById('buySignals').textContent = filteredSignals.filter(r => r.final_signal && r.final_signal.includes('BUY')).length;
            document.getElementById('sellSignals').textContent = filteredSignals.filter(r => r.final_signal && r.final_signal.includes('SELL')).length;
            document.getElementById('avgScore').textContent = filteredSignals.length > 0 ? (filteredSignals.reduce((s, r) => s + (r.final_combined_score || 0), 0) / filteredSignals.length).toFixed(1) : 0;
            document.getElementById('elliottWaves').textContent = filteredSignals.filter(r => r.elliott_current_wave && !['Unknown','None','Error'].includes(r.elliott_current_wave)).length;
            document.getElementById('tableCount').textContent = filteredSignals.length;
            updateCharts();
            updateTopPicks();
            renderTable();
        }}

        function updateCharts() {{
            const signalCounts = {{}};
            filteredSignals.forEach(r => {{ const s = r.final_signal || 'Unknown'; signalCounts[s] = (signalCounts[s] || 0) + 1; }});
            const ctx1 = document.getElementById('signalChart').getContext('2d');
            if(signalChartInstance) signalChartInstance.destroy();
            signalChartInstance = new Chart(ctx1, {{type:'doughnut',data:{{labels:Object.keys(signalCounts),datasets:[{{data:Object.values(signalCounts),backgroundColor:['#00ff88','#00cc66','#ffa500','#ffd700','#ff4757','#ff0000']}}]}},options:{{plugins:{{legend:{{position:'bottom',labels:{{color:'#aaa',font:{{size:10}}}}}}}}}}}});

            const modelCounts = {{}};
            filteredSignals.forEach(r => {{ const m = r.model_availability || 'Unknown'; modelCounts[m] = (modelCounts[m] || 0) + 1; }});
            const ctx2 = document.getElementById('modelChart').getContext('2d');
            if(modelChartInstance) modelChartInstance.destroy();
            modelChartInstance = new Chart(ctx2, {{type:'pie',data:{{labels:Object.keys(modelCounts),datasets:[{{data:Object.values(modelCounts),backgroundColor:['#00d4ff','#7b2ff7','#ffa500','#ff4757']}}]}},options:{{plugins:{{legend:{{position:'bottom',labels:{{color:'#aaa',font:{{size:10}}}}}}}}}}}});

            const waveCounts = {{}};
            filteredSignals.forEach(r => {{ const w = r.elliott_current_wave; if(w && !['Unknown','None','Error'].includes(w)) waveCounts[w] = (waveCounts[w] || 0) + 1; }});
            const ctx3 = document.getElementById('elliottChart').getContext('2d');
            if(elliottChartInstance) elliottChartInstance.destroy();
            elliottChartInstance = new Chart(ctx3, {{type:'bar',data:{{labels:Object.keys(waveCounts),datasets:[{{data:Object.values(waveCounts),backgroundColor:['#ff6b6b','#ffa500','#ffd700','#00ff88','#00d4ff','#7b2ff7']}}]}},options:{{plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{color:'#aaa'}}}},y:{{ticks:{{color:'#aaa'}}}}}}}}}});
        }}

        function updateTopPicks() {{
            const picks = filteredSignals.filter(r => r.final_signal && r.final_signal.includes('BUY')).sort((a,b) => (b.final_combined_score||0) - (a.final_combined_score||0)).slice(0,8);
            document.getElementById('topPicksGrid').innerHTML = picks.length > 0 ? picks.map(r => `<div class="pick-card"><strong>${{r.symbol}}</strong><span class="pick-score" style="float:right;">${{(r.final_combined_score||0).toFixed(1)}}</span><div class="pick-details"><span>📈 ${{r.final_signal}}</span><span>💰 ${{(r.entry_price||0).toFixed(2)}}</span><span>🛑 ${{(r.stop_loss||0).toFixed(2)}}</span><span>🎯 ${{(r.target_price||0).toFixed(2)}}</span><span>🌊 ${{r.elliott_current_wave||'N/A'}}</span></div></div>`).join('') : '<p style="color:#888;">No BUY signals</p>';
        }}

        function renderTable() {{
            document.getElementById('tableBody').innerHTML = filteredSignals.length > 0 ? filteredSignals.map((r,i) => `<tr><td>${{i+1}}</td><td><strong>${{r.symbol}}</strong></td><td>${{(r.current_price||0).toFixed(2)}}</td><td class="${{signalColors[r.final_signal]||''}}">${{r.final_signal||'N/A'}}</td><td><strong>${{(r.final_combined_score||0).toFixed(1)}}</strong></td><td>${{r.llm_signal||'N/A'}}</td><td>${{(r.llm_confidence||0).toFixed(0)}}%</td><td>${{r.xgb_signal||'N/A'}}</td><td>${{(r.xgb_auc||0).toFixed(3)}}</td><td>${{r.ppo_signal||'N/A'}}</td><td>${{(r.agentic_score||0).toFixed(1)}}</td><td>${{r.elliott_current_wave||'N/A'}}</td><td>${{(r.entry_price||0).toFixed(2)}}</td><td>${{(r.stop_loss||0).toFixed(2)}}</td><td>${{(r.target_price||0).toFixed(2)}}</td><td>${{r.risk_reward_ratio||0}}</td><td>${{r.model_availability||'N/A'}}</td></tr>`).join('') : '<tr><td colspan="17" style="text-align:center;padding:40px;color:#888;">Loading...</td></tr>';
        }}

        function applyFilters() {{
            const search = document.getElementById('searchSymbol').value.toUpperCase();
            const signalFilter = document.getElementById('filterSignal').value;
            const minScore = parseInt(document.getElementById('minScore').value);
            filteredSignals = allSignals.filter(r => {{
                if(search && !r.symbol.toUpperCase().includes(search)) return false;
                if(signalFilter !== 'ALL' && !(r.final_signal||'').includes(signalFilter)) return false;
                if((r.final_combined_score||0) < minScore) return false;
                return true;
            }});
            updateDashboard();
        }}

        function resetFilters() {{
            document.getElementById('searchSymbol').value = '';
            document.getElementById('filterSignal').value = 'ALL';
            document.getElementById('minScore').value = 0;
            document.getElementById('minScoreValue').textContent = '0';
            filteredSignals = [...allSignals];
            updateDashboard();
        }}

        function exportCSV() {{
            if(!filteredSignals.length) return;
            const headers = Object.keys(filteredSignals[0]);
            const csv = [headers.join(','), ...filteredSignals.map(r => headers.map(h => JSON.stringify(r[h]||'')).join(','))].join('\\n');
            const blob = new Blob([csv], {{type:'text/csv'}});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'signals_export.csv';
            a.click();
        }}

        document.addEventListener('DOMContentLoaded', loadData);
    </script>
</body>
</html>""".replace("{HF_REPO}", HF_REPO)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """মেইন ড্যাশবোর্ড"""
    return HTML_TEMPLATE

# =========================================================
# সরাসরি রান (Python)
# =========================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 AI TRADING SIGNALS DASHBOARD + API")
    print("=" * 60)
    print(f"🌐 Dashboard: http://localhost:{PORT}")
    print(f"📡 API Docs: http://localhost:{PORT}/docs")
    print(f"📊 Signals API: http://localhost:{PORT}/api/signals")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=PORT)