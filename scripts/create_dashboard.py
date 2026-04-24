"""
scripts/create_dashboard.py
সম্পূর্ণ AI Trading Signals ড্যাশবোর্ড তৈরি করে (HTML + CSS + JS)
Hugging Face থেকে CSV লোড করে
"""

import os

# =========================================================
# কনফিগারেশন
# =========================================================
DASHBOARD_DIR = "./output/dashboard"
CSV_FILENAME = "FINAL_AI_SIGNALS.csv"
HF_REPO = "ahashanahmed/csv"
HF_CSV_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{CSV_FILENAME}"

os.makedirs(DASHBOARD_DIR, exist_ok=True)

print("="*70)
print("🎨 CREATING AI TRADING SIGNALS DASHBOARD")
print("="*70)
print(f"📂 Output Directory: {DASHBOARD_DIR}")
print(f"📊 CSV Source: {HF_CSV_URL}")
print("="*70)

# =========================================================
# CSS
# =========================================================
CSS = """
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    line-height: 1.6;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

.header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 30px 0;
    text-align: center;
    border-bottom: 2px solid #00d4ff;
}

.header h1 {
    font-size: 2.5em;
    margin-bottom: 5px;
    background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header p {
    color: #888;
    font-size: 1.1em;
}

.header-stats {
    display: flex;
    justify-content: center;
    gap: 20px;
    flex-wrap: wrap;
    margin-top: 20px;
}

.stat-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 15px 25px;
    text-align: center;
    min-width: 120px;
}

.stat-card.buy { border-left: 3px solid #00ff88; }
.stat-card.sell { border-left: 3px solid #ff4757; }

.stat-label {
    display: block;
    font-size: 0.8em;
    color: #888;
    margin-bottom: 5px;
}

.stat-value {
    display: block;
    font-size: 1.8em;
    font-weight: bold;
    color: #fff;
}

.last-updated {
    margin-top: 10px;
    color: #666;
    font-size: 0.85em;
}

.filters {
    background: #111122;
    padding: 20px 0;
    border-bottom: 1px solid #222;
    position: sticky;
    top: 0;
    z-index: 100;
}

.filter-row {
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
    align-items: center;
}

.filter-group {
    display: flex;
    align-items: center;
    gap: 8px;
}

.filter-group label {
    color: #aaa;
    font-size: 0.9em;
    white-space: nowrap;
}

.filter-group input[type="text"],
.filter-group select {
    background: #1a1a2e;
    border: 1px solid #333;
    color: #fff;
    padding: 8px 12px;
    border-radius: 5px;
    font-size: 0.9em;
}

.filter-group input[type="text"] { width: 160px; }
.filter-group input[type="range"] { width: 80px; accent-color: #00d4ff; }

.btn-reset, .btn-export {
    background: #333;
    color: #fff;
    border: 1px solid #555;
    padding: 8px 16px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 0.9em;
    margin-top: 10px;
}

.btn-reset:hover, .btn-export:hover { background: #444; }
.btn-export { background: #0f3460; border-color: #00d4ff; margin-left: 10px; }

.charts { padding: 30px 0; }

.chart-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 20px;
}

.chart-card {
    background: #111122;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 20px;
}

.chart-card h3 {
    margin-bottom: 15px;
    color: #aaa;
    font-size: 1em;
    text-align: center;
}

.chart-card canvas { max-height: 250px; }

.top-picks { padding: 20px 0; }
.top-picks h2 { margin-bottom: 15px; color: #00ff88; }

.top-picks-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 15px;
}

.pick-card {
    background: linear-gradient(135deg, #0a2a0a, #0f1f0f);
    border: 1px solid #00ff88;
    border-radius: 10px;
    padding: 15px;
}

.pick-card strong { color: #00ff88; font-size: 1.2em; }
.pick-card .pick-details { margin-top: 8px; font-size: 0.85em; color: #aaa; }
.pick-card .pick-details span { display: inline-block; margin-right: 12px; }
.pick-card .pick-score { font-size: 2em; font-weight: bold; color: #00d4ff; }

.table-section { padding: 20px 0 40px; }
.table-section h2 { margin-bottom: 15px; }

.table-container {
    overflow-x: auto;
    background: #111122;
    border: 1px solid #222;
    border-radius: 10px;
}

#signalTable {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85em;
}

#signalTable thead {
    background: #1a1a2e;
    position: sticky;
    top: 0;
}

#signalTable th {
    padding: 12px 8px;
    text-align: left;
    color: #00d4ff;
    font-weight: 600;
    border-bottom: 2px solid #333;
    white-space: nowrap;
}

#signalTable td {
    padding: 10px 8px;
    border-bottom: 1px solid #1a1a2e;
    white-space: nowrap;
}

#signalTable tbody tr:hover { background: rgba(0,212,255,0.05); }

.signal-strong-buy { color: #00ff88; font-weight: bold; }
.signal-buy { color: #00cc66; font-weight: bold; }
.signal-watch { color: #ffa500; }
.signal-hold { color: #ffd700; }
.signal-sell { color: #ff4757; }
.signal-strong-sell { color: #ff0000; font-weight: bold; }

footer {
    background: #0a0a0f;
    text-align: center;
    padding: 20px;
    color: #555;
    font-size: 0.85em;
    border-top: 1px solid #222;
}

#loadingOverlay {
    display: flex;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.85);
    z-index: 9999;
    justify-content: center;
    align-items: center;
    flex-direction: column;
}

.spinner {
    width: 60px;
    height: 60px;
    border: 4px solid #333;
    border-top: 4px solid #00d4ff;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

@media (max-width: 768px) {
    .header h1 { font-size: 1.8em; }
    .header-stats { gap: 10px; }
    .stat-card { padding: 10px 15px; min-width: 90px; }
    .stat-value { font-size: 1.3em; }
    .filter-row { flex-direction: column; }
    .chart-row { grid-template-columns: 1fr; }
}
"""

# =========================================================
# JavaScript
# =========================================================
JS = f"""
// =========================================================
// AI Trading Signals Dashboard - Complete JavaScript
// Hugging Face থেকে CSV লোড করে
// =========================================================

const HF_CSV_URL = '{HF_CSV_URL}';
let allData = [];
let filteredData = [];
let signalChartInstance = null;
let modelChartInstance = null;
let elliottChartInstance = null;

const signalColors = {{
    '🔥 STRONG BUY': 'signal-strong-buy',
    '✅ BUY': 'signal-buy',
    '👀 WATCH (Near BUY)': 'signal-watch',
    '⏳ HOLD': 'signal-hold',
    '⚠️ WATCH (Near SELL)': 'signal-watch',
    '❌ SELL': 'signal-sell',
    '💀 STRONG SELL': 'signal-strong-sell',
}};

// =========================================================
// Load CSV from Hugging Face
// =========================================================
async function loadData() {{
    try {{
        showLoading(true);
        const response = await fetch(HF_CSV_URL);
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const csvText = await response.text();
        
        Papa.parse(csvText, {{
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: function(results) {{
                allData = results.data.filter(row => row.symbol && row.symbol !== 'symbol');
                filteredData = [...allData];
                updateDashboard();
                showLoading(false);
                console.log(`✅ Loaded ${{allData.length}} signals from Hugging Face`);
            }},
            error: function(err) {{
                console.error('CSV Parse Error:', err);
                showError('❌ Error parsing CSV file');
            }}
        }});
    }} catch (e) {{
        console.error('Fetch Error:', e);
        showError('⚠️ Could not load from Hugging Face. Please upload FINAL_AI_SIGNALS.csv to the repository.');
    }}
}}

function showLoading(show) {{
    const el = document.getElementById('loadingOverlay');
    if (el) el.style.display = show ? 'flex' : 'none';
}}

function showError(msg) {{
    showLoading(false);
    const tbody = document.getElementById('tableBody');
    if (tbody) {{
        tbody.innerHTML = `<tr><td colspan="18" style="text-align:center;padding:40px;color:orange;">${{msg}}</td></tr>`;
    }}
}}

// =========================================================
// Update Dashboard
// =========================================================
function updateDashboard() {{
    updateHeaderStats();
    updateCharts();
    updateTopPicks();
    renderTable();
}}

// =========================================================
// Header Stats
// =========================================================
function updateHeaderStats() {{
    const total = filteredData.length;
    const buyCount = filteredData.filter(r => r.final_signal && r.final_signal.includes('BUY')).length;
    const sellCount = filteredData.filter(r => r.final_signal && r.final_signal.includes('SELL')).length;
    const avgScore = filteredData.length > 0 
        ? (filteredData.reduce((sum, r) => sum + (r.final_combined_score || 0), 0) / filteredData.length).toFixed(1) 
        : 0;
    const elliottCount = filteredData.filter(r => r.elliott_current_wave && r.elliott_current_wave !== 'Unknown' && r.elliott_current_wave !== 'None' && r.elliott_current_wave !== 'Error').length;
    
    document.getElementById('totalSignals').textContent = total;
    document.getElementById('buySignals').textContent = buyCount;
    document.getElementById('sellSignals').textContent = sellCount;
    document.getElementById('avgScore').textContent = avgScore;
    document.getElementById('elliottWaves').textContent = elliottCount;
    document.getElementById('lastUpdated').textContent = '📅 Data Source: Hugging Face (' + HF_CSV_URL + ')';
    document.getElementById('tableCount').textContent = total;
}}

// =========================================================
// Charts
// =========================================================
function updateCharts() {{
    // Signal Distribution
    const signalCounts = {{}};
    filteredData.forEach(r => {{
        const sig = r.final_signal || 'Unknown';
        signalCounts[sig] = (signalCounts[sig] || 0) + 1;
    }});
    
    const signalCtx = document.getElementById('signalChart').getContext('2d');
    if (signalChartInstance) signalChartInstance.destroy();
    signalChartInstance = new Chart(signalCtx, {{
        type: 'doughnut',
        data: {{
            labels: Object.keys(signalCounts),
            datasets: [{{
                data: Object.values(signalCounts),
                backgroundColor: ['#00ff88', '#00cc66', '#ffa500', '#ffd700', '#ff4757', '#ff0000'],
                borderColor: '#111122',
                borderWidth: 2,
            }}]
        }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ position: 'bottom', labels: {{ color: '#aaa', font: {{ size: 10 }} }} }} }}
        }}
    }});
    
    // Model Availability
    const modelCounts = {{}};
    filteredData.forEach(r => {{
        const ma = r.model_availability || 'Unknown';
        modelCounts[ma] = (modelCounts[ma] || 0) + 1;
    }});
    
    const modelCtx = document.getElementById('modelChart').getContext('2d');
    if (modelChartInstance) modelChartInstance.destroy();
    modelChartInstance = new Chart(modelCtx, {{
        type: 'pie',
        data: {{
            labels: Object.keys(modelCounts),
            datasets: [{{
                data: Object.values(modelCounts),
                backgroundColor: ['#00d4ff', '#7b2ff7', '#ffa500', '#ff4757'],
                borderColor: '#111122',
                borderWidth: 2,
            }}]
        }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ position: 'bottom', labels: {{ color: '#aaa', font: {{ size: 10 }} }} }} }}
        }}
    }});
    
    // Elliott Wave
    const waveCounts = {{}};
    filteredData.forEach(r => {{
        const wave = r.elliott_current_wave || 'Unknown';
        if (wave !== 'Unknown' && wave !== 'None' && wave !== 'Error') {{
            waveCounts[wave] = (waveCounts[wave] || 0) + 1;
        }}
    }});
    
    const elliottCtx = document.getElementById('elliottChart').getContext('2d');
    if (elliottChartInstance) elliottChartInstance.destroy();
    elliottChartInstance = new Chart(elliottCtx, {{
        type: 'bar',
        data: {{
            labels: Object.keys(waveCounts),
            datasets: [{{
                label: 'Count',
                data: Object.values(waveCounts),
                backgroundColor: ['#ff6b6b', '#ffa500', '#ffd700', '#00ff88', '#00d4ff', '#7b2ff7', '#ff4757'],
                borderColor: '#111122',
                borderWidth: 1,
            }}]
        }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ display: false }} }},
            scales: {{
                x: {{ ticks: {{ color: '#aaa' }} }},
                y: {{ ticks: {{ color: '#aaa' }} }}
            }}
        }}
    }});
}}

// =========================================================
// Top Picks
// =========================================================
function updateTopPicks() {{
    const buyPicks = filteredData
        .filter(r => r.final_signal && r.final_signal.includes('BUY'))
        .sort((a, b) => (b.final_combined_score || 0) - (a.final_combined_score || 0))
        .slice(0, 8);
    
    const grid = document.getElementById('topPicksGrid');
    if (buyPicks.length === 0) {{
        grid.innerHTML = '<p style="color:#888;">No BUY signals available</p>';
        return;
    }}
    
    grid.innerHTML = buyPicks.map(r => `
        <div class="pick-card">
            <strong>${{r.symbol}}</strong>
            <span style="float:right;" class="pick-score">${{(r.final_combined_score || 0).toFixed(1)}}</span>
            <div class="pick-details">
                <span>📈 ${{r.final_signal}}</span>
                <span>💰 ${{(r.entry_price || 0).toFixed(2)}}</span>
                <span>🛑 ${{(r.stop_loss || 0).toFixed(2)}}</span>
                <span>🎯 ${{(r.target_price || 0).toFixed(2)}}</span>
                <span>📊 ${{r.risk_reward_ratio || 0}}</span>
                <span>🌊 ${{r.elliott_current_wave || 'N/A'}}</span>
            </div>
        </div>
    `).join('');
}}

// =========================================================
// Render Table
// =========================================================
function renderTable() {{
    const tbody = document.getElementById('tableBody');
    if (filteredData.length === 0) {{
        tbody.innerHTML = '<tr><td colspan="18" style="text-align:center;padding:40px;color:#888;">No data found</td></tr>';
        return;
    }}
    
    tbody.innerHTML = filteredData.map((r, i) => {{
        const signalClass = signalColors[r.final_signal] || '';
        return `
            <tr>
                <td>${{i + 1}}</td>
                <td><strong>${{r.symbol}}</strong></td>
                <td>${{(r.current_price || 0).toFixed(2)}}</td>
                <td class="${{signalClass}}">${{r.final_signal || 'N/A'}}</td>
                <td><strong>${{(r.final_combined_score || 0).toFixed(1)}}</strong></td>
                <td>${{r.llm_signal || 'N/A'}}</td>
                <td>${{(r.llm_confidence || 0).toFixed(0)}}%</td>
                <td>${{r.xgb_signal || 'N/A'}}</td>
                <td>${{(r.xgb_auc || 0).toFixed(3)}}</td>
                <td>${{r.ppo_signal || 'N/A'}}</td>
                <td>${{(r.agentic_score || 0).toFixed(1)}}</td>
                <td>${{r.elliott_current_wave || 'N/A'}}</td>
                <td style="font-size:0.75em;max-width:200px;overflow:hidden;text-overflow:ellipsis;">${{r.elliott_sub_waves || 'N/A'}}</td>
                <td>${{(r.entry_price || 0).toFixed(2)}}</td>
                <td>${{(r.stop_loss || 0).toFixed(2)}}</td>
                <td>${{(r.target_price || 0).toFixed(2)}}</td>
                <td>${{r.risk_reward_ratio || 0}}</td>
                <td>${{r.model_availability || 'N/A'}}</td>
            </tr>
        `;
    }}).join('');
}}

// =========================================================
// Filters
// =========================================================
function applyFilters() {{
    const search = document.getElementById('searchSymbol').value.toUpperCase();
    const signalFilter = document.getElementById('filterSignal').value;
    const modelFilter = document.getElementById('filterModel').value;
    const elliottFilter = document.getElementById('filterElliott').value;
    const minScore = parseInt(document.getElementById('minScore').value);
    
    filteredData = allData.filter(r => {{
        if (search && !r.symbol.toUpperCase().includes(search)) return false;
        if (signalFilter !== 'ALL') {{
            if (signalFilter === 'STRONG BUY' && !r.final_signal?.includes('STRONG BUY')) return false;
            if (signalFilter === 'BUY' && !(r.final_signal?.includes('BUY') && !r.final_signal?.includes('STRONG') && !r.final_signal?.includes('WATCH'))) return false;
            if (signalFilter === 'WATCH' && !r.final_signal?.includes('WATCH')) return false;
            if (signalFilter === 'HOLD' && !r.final_signal?.includes('HOLD')) return false;
            if (signalFilter === 'SELL' && !(r.final_signal?.includes('SELL') && !r.final_signal?.includes('STRONG') && !r.final_signal?.includes('WATCH'))) return false;
            if (signalFilter === 'STRONG SELL' && !r.final_signal?.includes('STRONG SELL')) return false;
        }}
        if (modelFilter !== 'ALL' && !r.model_availability?.includes(modelFilter.split(' ')[0])) return false;
        if (elliottFilter !== 'ALL' && r.elliott_current_wave !== elliottFilter) return false;
        if ((r.final_combined_score || 0) < minScore) return false;
        return true;
    }});
    
    updateDashboard();
}}

function resetFilters() {{
    document.getElementById('searchSymbol').value = '';
    document.getElementById('filterSignal').value = 'ALL';
    document.getElementById('filterModel').value = 'ALL';
    document.getElementById('filterElliott').value = 'ALL';
    document.getElementById('minScore').value = 0;
    document.getElementById('minScoreValue').textContent = '0';
    filteredData = [...allData];
    updateDashboard();
}}

function exportToCSV() {{
    if (filteredData.length === 0) return;
    const csv = Papa.unparse(filteredData);
    const blob = new Blob([csv], {{ type: 'text/csv' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'AI_Trading_Signals_Export.csv';
    a.click();
    URL.revokeObjectURL(url);
}}

// =========================================================
// Initialize
// =========================================================
document.addEventListener('DOMContentLoaded', loadData);
"""

# =========================================================
# HTML
# =========================================================
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 AI Trading Signals Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js"></script>
    <style>{CSS}</style>
</head>
<body>
    <!-- Loading Overlay -->
    <div id="loadingOverlay">
        <div class="spinner"></div>
        <p style="color:#fff;margin-top:20px;font-size:1.2em;">📂 Loading AI Signals from Hugging Face...</p>
    </div>

    <!-- Header -->
    <header class="header">
        <div class="container">
            <h1>🤖 AI Trading Signals Dashboard</h1>
            <p>LLM + XGBoost + PPO + Agentic Loop + Elliott Wave Combined</p>
            <div class="header-stats" id="headerStats">
                <div class="stat-card">
                    <span class="stat-label">Total Signals</span>
                    <span class="stat-value" id="totalSignals">-</span>
                </div>
                <div class="stat-card buy">
                    <span class="stat-label">🟢 Buy Signals</span>
                    <span class="stat-value" id="buySignals">-</span>
                </div>
                <div class="stat-card sell">
                    <span class="stat-label">🔴 Sell Signals</span>
                    <span class="stat-value" id="sellSignals">-</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">📊 Avg Score</span>
                    <span class="stat-value" id="avgScore">-</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">🌊 Elliott Waves</span>
                    <span class="stat-value" id="elliottWaves">-</span>
                </div>
            </div>
            <div class="last-updated" id="lastUpdated">📅 Data Source: Hugging Face</div>
        </div>
    </header>

    <!-- Filters -->
    <section class="filters">
        <div class="container">
            <div class="filter-row">
                <div class="filter-group">
                    <label>🔍 Symbol:</label>
                    <input type="text" id="searchSymbol" placeholder="Search..." onkeyup="applyFilters()">
                </div>
                <div class="filter-group">
                    <label>📈 Signal:</label>
                    <select id="filterSignal" onchange="applyFilters()">
                        <option value="ALL">All Signals</option>
                        <option value="STRONG BUY">🔥 STRONG BUY</option>
                        <option value="BUY">✅ BUY</option>
                        <option value="WATCH">👀 WATCH</option>
                        <option value="HOLD">⏳ HOLD</option>
                        <option value="SELL">❌ SELL</option>
                        <option value="STRONG SELL">💀 STRONG SELL</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>🤖 Models:</label>
                    <select id="filterModel" onchange="applyFilters()">
                        <option value="ALL">All</option>
                        <option value="FULL">FULL (3/3)</option>
                        <option value="GOOD">GOOD (2/3)</option>
                        <option value="BASIC">BASIC (1/3)</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>🌊 Elliott:</label>
                    <select id="filterElliott" onchange="applyFilters()">
                        <option value="ALL">All</option>
                        <option value="Wave 1">Wave 1</option>
                        <option value="Wave 2">Wave 2</option>
                        <option value="Wave 3">Wave 3</option>
                        <option value="Wave 4">Wave 4</option>
                        <option value="Wave 5">Wave 5</option>
                        <option value="Wave A">Wave A</option>
                        <option value="Wave B">Wave B</option>
                        <option value="Wave C">Wave C</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>⭐ Score ≥</label>
                    <input type="range" id="minScore" min="0" max="100" value="0" oninput="document.getElementById('minScoreValue').textContent=this.value;applyFilters()">
                    <span id="minScoreValue">0</span>
                </div>
            </div>
            <button class="btn-reset" onclick="resetFilters()">🔄 Reset</button>
            <button class="btn-export" onclick="exportToCSV()">📥 Export CSV</button>
        </div>
    </section>

    <!-- Charts -->
    <section class="charts">
        <div class="container">
            <div class="chart-row">
                <div class="chart-card">
                    <h3>📈 Signal Distribution</h3>
                    <canvas id="signalChart"></canvas>
                </div>
                <div class="chart-card">
                    <h3>🤖 Model Availability</h3>
                    <canvas id="modelChart"></canvas>
                </div>
                <div class="chart-card">
                    <h3>🌊 Elliott Wave Distribution</h3>
                    <canvas id="elliottChart"></canvas>
                </div>
            </div>
        </div>
    </section>

    <!-- Top Picks -->
    <section class="top-picks">
        <div class="container">
            <h2>🔥 Top BUY Picks</h2>
            <div class="top-picks-grid" id="topPicksGrid"></div>
        </div>
    </section>

    <!-- Table -->
    <section class="table-section">
        <div class="container">
            <h2>📊 All Signals (Total: <span id="tableCount">0</span>)</h2>
            <div class="table-container">
                <table id="signalTable">
                    <thead>
                        <tr>
                            <th>#</th><th>Symbol</th><th>Price</th><th>Final Signal</th><th>Score</th>
                            <th>LLM</th><th>LLM%</th><th>XGB</th><th>AUC</th><th>PPO</th>
                            <th>Agentic</th><th>Elliott</th><th>Sub-Waves</th>
                            <th>Entry</th><th>SL</th><th>TP</th><th>R:R</th><th>Models</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody"></tbody>
                </table>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer>
        <p>🤖 AI Trading System | LLM + XGBoost + PPO + Agentic Loop + Elliott Wave | Data: {HF_REPO}</p>
    </footer>

    <script>{JS}</script>
</body>
</html>"""

# =========================================================
# ফাইল সেভ করুন
# =========================================================
output_file = os.path.join(DASHBOARD_DIR, "index.html")

with open(output_file, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"\n✅ Dashboard created successfully!")
print(f"📂 File: {output_file}")
print(f"📊 Size: {len(HTML):,} bytes")
print(f"\n🚀 TO DEPLOY ON RENDER:")
print(f"   1. Go to https://render.com")
print(f"   2. New Static Site")
print(f"   3. Publish Directory: output/dashboard")
print(f"   4. Deploy!")
print(f"\n📂 OR OPEN LOCALLY:")
print(f"   {output_file}")
print("="*70)
