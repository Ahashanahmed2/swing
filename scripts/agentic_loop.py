# agentic_loop.py - Multi-Agent Voting System with LLM Integration
# ✅ FIXED: Accuracy persistence, feedback matching, agent_votes tracking
# ✅ FIXED: recent_accuracy saved/loaded properly
# ✅ FIXED: Decision log rolling — recent entry lookup works across runs
# ✅ ADDED: Walk-forward outcome evaluation helper
# ✅ ADDED: record_trade_feedback() alias
# ✅ ADDED: Robust feedback (works even without prediction_date match)

import pandas as pd
import numpy as np
import os
import joblib
import json
import requests
import re
import torch
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available — LLM agents disabled")


STATE_FILE = './csv/agentic_loop_state.json'
DECISION_LOG_FILE = './csv/agentic_loop_log.csv'
PERFORMANCE_LOG_FILE = './csv/agentic_loop_performance.csv'


# =========================
# TELEGRAM
# =========================

def send_telegram_message(message, token=None, chat_id=None):
    token = token or os.getenv("TELEGRAM_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        response = requests.post(url, json=payload, timeout=10)
        return response.json()
    except:
        return False


# =========================
# BASE AGENT
# =========================

class TradingAgent:
    """Base class for all trading agents"""

    def __init__(self, name, weight=1.0):
        self.name = name
        self.weight = weight
        self.performance_history = []
        self.correct_predictions = 0
        self.total_predictions = 0
        self.recent_accuracy = 0.5
        self.outcomes = []  # rolling window of 1/0

    def update_performance(self, was_correct, confidence):
        self.total_predictions += 1
        if was_correct:
            self.correct_predictions += 1

        self.outcomes.append(1 if was_correct else 0)
        if len(self.outcomes) > 100:
            self.outcomes = self.outcomes[-100:]

        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'correct': bool(was_correct),
            'confidence': float(confidence)
        })
        if len(self.performance_history) > 200:
            self.performance_history = self.performance_history[-200:]

        if len(self.outcomes) >= 20:
            recent = self.outcomes[-20:]
            self.recent_accuracy = sum(recent) / len(recent)
        elif len(self.outcomes) > 0:
            self.recent_accuracy = sum(self.outcomes) / len(self.outcomes)

    def get_accuracy(self):
        if self.total_predictions == 0:
            return 0.5
        return self.correct_predictions / self.total_predictions

    def get_dynamic_weight(self):
        base_weight = self.weight
        accuracy = self.recent_accuracy if self.total_predictions >= 10 else self.get_accuracy()

        if accuracy > 0.6:
            return base_weight * (1 + (accuracy - 0.6) * 1.5)
        elif accuracy < 0.4:
            return base_weight * (0.5 + accuracy * 0.5)
        return base_weight

    def to_state(self):
        """✅ সব কিছু save করে — accuracy সহ"""
        return {
            'name': self.name,
            'correct': self.correct_predictions,
            'total': self.total_predictions,
            'recent_accuracy': self.recent_accuracy,
            'weight': self.weight,
            'outcomes': self.outcomes[-50:],  # last 50 outcomes
            'performance_history': [
                {'timestamp': str(p['timestamp']),
                 'correct': p['correct'],
                 'confidence': p['confidence']}
                for p in self.performance_history[-50:]
            ]
        }

    def from_state(self, data):
        """✅ সব কিছু restore করে"""
        self.correct_predictions = data.get('correct', 0)
        self.total_predictions = data.get('total', 0)
        self.recent_accuracy = data.get('recent_accuracy', 0.5)
        if 'weight' in data:
            self.weight = data['weight']
        self.outcomes = data.get('outcomes', [])

        hist = data.get('performance_history', [])
        self.performance_history = []
        for h in hist:
            try:
                ts = pd.to_datetime(h.get('timestamp'))
            except Exception:
                ts = datetime.now()
            self.performance_history.append({
                'timestamp': ts,
                'correct': h.get('correct', False),
                'confidence': h.get('confidence', 0.5)
            })


# =========================
# XGBoost AGENT
# =========================

class XGBoostAgent(TradingAgent):
    def __init__(self, xgb_model_dir):
        super().__init__("XGBoost", weight=0.35)
        self.model_dir = xgb_model_dir
        self.models = {}
        self.current_symbol = None
        self.model_auc_scores = {}
        self.load_models()

    def load_models(self):
        try:
            if os.path.exists(self.model_dir):
                model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.joblib')]
                for file in model_files:
                    symbol = file.replace('.joblib', '')
                    try:
                        self.models[symbol] = joblib.load(os.path.join(self.model_dir, file))
                    except Exception as e:
                        print(f"   ⚠️ Failed to load {symbol}: {e}")
                if self.models:
                    print(f"   ✅ XGBoost Agent loaded {len(self.models)} models")
                    self._load_model_quality()
                else:
                    print(f"   ⚠️ No XGBoost models in {self.model_dir}")
            else:
                print(f"   ⚠️ XGBoost dir not found: {self.model_dir}")
        except Exception as e:
            print(f"   ⚠️ XGBoost init failed: {e}")

    def _load_model_quality(self):
        if os.path.exists('./csv/model_metadata.csv'):
            try:
                metadata = pd.read_csv('./csv/model_metadata.csv')
                for _, row in metadata.iterrows():
                    if row['symbol'] in self.models:
                        self.model_auc_scores[row['symbol']] = row.get('auc', 0.5)
            except:
                pass

    def set_symbol(self, symbol):
        self.current_symbol = symbol

    def predict(self, features):
        if not self.models or self.current_symbol not in self.models:
            return 0.5, 0.3
        try:
            model = self.models[self.current_symbol]
            prob = model.predict_proba(features)[0, 1]
            base_confidence = 0.6
            if self.current_symbol in self.model_auc_scores:
                auc = self.model_auc_scores[self.current_symbol]
                base_confidence = min(0.9, max(0.4, auc))
            return prob, base_confidence
        except:
            return 0.5, 0.3


# =========================
# TECHNICAL AGENT
# =========================

class TechnicalAgent(TradingAgent):
    def __init__(self):
        super().__init__("Technical", weight=0.20)

    def analyze(self, symbol_data):
        if len(symbol_data) < 20:
            return 0.5, 0.3

        signals, confidences = [], []

        if 'rsi' in symbol_data.columns:
            rsi = symbol_data['rsi'].iloc[-1]
            if not pd.isna(rsi):
                if rsi < 30:
                    signals.append(1); confidences.append(min(0.8, (30 - rsi) / 30 + 0.3))
                elif rsi > 70:
                    signals.append(0); confidences.append(min(0.8, (rsi - 70) / 30 + 0.3))
                else:
                    signals.append(0.5); confidences.append(0.4)

        if 'macd' in symbol_data.columns and 'macd_signal' in symbol_data.columns:
            macd = symbol_data['macd'].iloc[-1]
            sig = symbol_data['macd_signal'].iloc[-1]
            hist = symbol_data['macd_hist'].iloc[-1] if 'macd_hist' in symbol_data.columns else 0
            if not pd.isna(macd) and not pd.isna(sig):
                if macd > sig and hist > 0:
                    signals.append(1); confidences.append(0.7)
                elif macd < sig and hist < 0:
                    signals.append(0); confidences.append(0.7)
                else:
                    signals.append(0.5); confidences.append(0.4)

        if 'bb_position' in symbol_data.columns:
            bb = symbol_data['bb_position'].iloc[-1]
            if not pd.isna(bb):
                if bb < 0.2:
                    signals.append(1); confidences.append(0.6)
                elif bb > 0.8:
                    signals.append(0); confidences.append(0.6)

        if 'dist_from_sr' in symbol_data.columns and 'is_support' in symbol_data.columns:
            dist = symbol_data['dist_from_sr'].iloc[-1]
            is_sup = symbol_data['is_support'].iloc[-1]
            if not pd.isna(dist) and not pd.isna(is_sup):
                if is_sup == 1 and abs(dist) < 2:
                    signals.append(1); confidences.append(0.65)
                elif is_sup == 0 and abs(dist) < 2:
                    signals.append(0); confidences.append(0.65)

        if 'is_bullish_div' in symbol_data.columns:
            if symbol_data['is_bullish_div'].iloc[-1] == 1:
                signals.append(1); confidences.append(0.75)
        if 'is_bearish_div' in symbol_data.columns:
            if symbol_data['is_bearish_div'].iloc[-1] == 1:
                signals.append(0); confidences.append(0.75)

        if not signals:
            return 0.5, 0.3

        weighted = sum(s * c for s, c in zip(signals, confidences)) / sum(confidences)
        avg_conf = sum(confidences) / len(confidences)
        return weighted, avg_conf


# =========================
# RISK AGENT
# =========================

class RiskAgent(TradingAgent):
    def __init__(self):
        super().__init__("Risk", weight=0.15)
        self.symbol_history = defaultdict(lambda: {'trades': 0, 'losses': 0, 'consecutive_losses': 0})

    def assess(self, symbol, volatility, market_regime, atr=None, drawdown=0):
        risk_score = 0.5
        if volatility > 0.03: risk_score -= 0.2
        elif volatility < 0.01: risk_score += 0.1
        if market_regime == 'BEAR': risk_score -= 0.25
        elif market_regime == 'BULL': risk_score += 0.15
        if atr is not None:
            atr_ratio = atr / 100
            if atr_ratio > 0.03: risk_score -= 0.15
            elif atr_ratio < 0.01: risk_score += 0.1
        if drawdown > 0.1: risk_score -= 0.2
        elif drawdown > 0.05: risk_score -= 0.1
        hist = self.symbol_history[symbol]
        if hist['consecutive_losses'] >= 2: risk_score -= 0.15
        elif hist['trades'] > 10 and hist['losses'] / hist['trades'] > 0.6: risk_score -= 0.2
        confidence = 0.7 + (abs(risk_score - 0.5) * 0.3)
        return max(0.1, min(0.9, risk_score)), confidence

    def update_history(self, symbol, was_loss):
        hist = self.symbol_history[symbol]
        hist['trades'] += 1
        if was_loss:
            hist['losses'] += 1
            hist['consecutive_losses'] += 1
        else:
            hist['consecutive_losses'] = 0


# =========================
# QWEN3 AGENT
# =========================

class Qwen3Agent(TradingAgent):
    def __init__(self, model_dir="./csv/llm_model_qwen3", weight=0.12):
        super().__init__("Qwen3", weight=weight)
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.available = False
        self.load_model()

    def load_model(self):
        if not TRANSFORMERS_AVAILABLE:
            print(f"   ⚠️ Qwen3: transformers unavailable")
            return
        if not os.path.exists(os.path.join(self.model_dir, "config.json")):
            print(f"   ⚠️ Qwen3: model not at {self.model_dir}")
            return
        try:
            print(f"   🔄 Loading Qwen3...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir, trust_remote_code=True,
                torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.available = True
            print(f"   ✅ Qwen3 loaded on {self.device}")
        except Exception as e:
            print(f"   ⚠️ Qwen3 load failed: {e}")
            self.available = False

    def analyze(self, symbol, symbol_data):
        if not self.available or len(symbol_data) < 5:
            return 0.5, 0.3
        try:
            latest = symbol_data.iloc[-1]
            close = float(latest.get('close', 0))
            rsi = float(latest.get('rsi', 50)) if not pd.isna(latest.get('rsi', 50)) else 50
            macd = float(latest.get('macd', 0)) if not pd.isna(latest.get('macd', 0)) else 0

            prompt = f"""Symbol: {symbol} | Price: {close:.2f}
RSI: {rsi:.1f} | MACD: {macd:.4f}

Respond ONLY in JSON: {{"signal": "BUY"|"SELL"|"HOLD", "confidence": 0-100}}
JSON:"""

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=40,
                    do_sample=False, temperature=0.1,  # ✅ deterministic
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "JSON:" in response:
                response = response.split("JSON:")[-1].strip()

            return self._parse_signal(response)
        except Exception:
            return 0.5, 0.3

    def _parse_signal(self, text):
        # Try JSON first
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end > start:
                parsed = json.loads(text[start:end+1])
                sig = str(parsed.get('signal', 'HOLD')).upper()
                conf = float(parsed.get('confidence', 50)) / 100.0
                conf = max(0.3, min(0.95, conf))
                if sig == 'BUY': return 0.75, conf
                elif sig == 'SELL': return 0.25, conf
                else: return 0.5, 0.3
        except:
            pass
        # Fallback to keyword
        score, confidence = 0.5, 0.4
        if re.search(r'\bBUY\b', text, re.IGNORECASE):
            score, confidence = 0.75, 0.6
        elif re.search(r'\bSELL\b', text, re.IGNORECASE):
            score, confidence = 0.25, 0.6
        m = re.search(r'(\d+)%', text)
        if m:
            v = float(m.group(1)) / 100.0
            if 0.3 <= v <= 0.95:
                confidence = v
        return score, confidence


# =========================
# DEEPSEEK AGENT
# =========================

class DeepSeekAgent(TradingAgent):
    def __init__(self, model_dir="./csv/llm_model_deepseek", weight=0.08):
        super().__init__("DeepSeek", weight=weight)
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.available = False
        self.load_model()

    def load_model(self):
        if not TRANSFORMERS_AVAILABLE:
            print(f"   ⚠️ DeepSeek: transformers unavailable")
            return
        if not os.path.exists(os.path.join(self.model_dir, "config.json")):
            print(f"   ⚠️ DeepSeek: model not at {self.model_dir}")
            return
        try:
            print(f"   🔄 Loading DeepSeek...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir, trust_remote_code=True,
                torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.available = True
            print(f"   ✅ DeepSeek loaded on {self.device}")
        except Exception as e:
            print(f"   ⚠️ DeepSeek load failed: {e}")
            self.available = False

    def analyze(self, symbol, symbol_data):
        if not self.available or len(symbol_data) < 5:
            return 0.5, 0.3
        try:
            latest = symbol_data.iloc[-1]
            close = float(latest.get('close', 0))
            rsi = float(latest.get('rsi', 50)) if not pd.isna(latest.get('rsi', 50)) else 50
            macd = float(latest.get('macd', 0)) if not pd.isna(latest.get('macd', 0)) else 0

            prompt = f"""Symbol: {symbol} | Price: {close:.2f}
RSI: {rsi:.1f} | MACD: {macd:.4f}

Respond ONLY in JSON: {{"signal": "BUY"|"SELL"|"HOLD", "confidence": 0-100}}
JSON:"""

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=40,
                    do_sample=False, temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "JSON:" in response:
                response = response.split("JSON:")[-1].strip()

            return self._parse_signal(response)
        except Exception:
            return 0.5, 0.3

    def _parse_signal(self, text):
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end > start:
                parsed = json.loads(text[start:end+1])
                sig = str(parsed.get('signal', 'HOLD')).upper()
                conf = float(parsed.get('confidence', 50)) / 100.0
                conf = max(0.3, min(0.95, conf))
                if sig == 'BUY': return 0.75, conf
                elif sig == 'SELL': return 0.25, conf
                else: return 0.5, 0.3
        except:
            pass
        score, confidence = 0.5, 0.4
        if re.search(r'\bBUY\b', text, re.IGNORECASE):
            score, confidence = 0.75, 0.6
        elif re.search(r'\bSELL\b', text, re.IGNORECASE):
            score, confidence = 0.25, 0.6
        return score, confidence


# =========================
# MEMORY AGENT
# =========================

class MemoryAgent(TradingAgent):
    def __init__(self):
        super().__init__("Memory", weight=0.10)
        self.mistake_memory = []
        self.success_memory = []
        self.pattern_memory = defaultdict(list)

    def remember_trade(self, trade_result):
        trade_result['timestamp'] = datetime.now()
        if trade_result.get('pnl', 0) < 0:
            self.mistake_memory.append(trade_result)
        else:
            self.success_memory.append(trade_result)
        if 'features' in trade_result:
            symbol = trade_result.get('symbol', 'UNKNOWN')
            self.pattern_memory[symbol].append({
                'features': trade_result['features'],
                'pnl': trade_result['pnl'],
                'timestamp': datetime.now()
            })
            if len(self.pattern_memory[symbol]) > 50:
                self.pattern_memory[symbol] = self.pattern_memory[symbol][-50:]

    def get_similar_pattern(self, symbol, current_features):
        if symbol not in self.pattern_memory or len(self.pattern_memory[symbol]) < 5:
            return 0.5, 0.3
        patterns = self.pattern_memory[symbol]
        wins = sum(1 for p in patterns if p['pnl'] > 0)
        total = len(patterns)
        if total > 0:
            win_rate = wins / total
            confidence = 0.4 + (abs(win_rate - 0.5) * 0.4)
            if win_rate > 0.55: return 0.65, confidence
            elif win_rate < 0.45: return 0.35, confidence
        return 0.5, 0.3


# =========================
# MAIN AGENTIC LOOP
# =========================

class AgenticLoop:
    def __init__(self, xgb_model_dir='./csv/xgboost/',
                 qwen3_model_dir='./csv/llm_model_qwen3',
                 deepseek_model_dir='./csv/llm_model_deepseek'):
        self.agents = []
        self.vote_history = []
        self.decision_log = []
        self.performance_log = []
        self.ensemble_correct = 0
        self.ensemble_total = 0

        self.agents.append(XGBoostAgent(xgb_model_dir))
        self.agents.append(TechnicalAgent())
        self.agents.append(RiskAgent())
        self.agents.append(MemoryAgent())
        self.agents.append(Qwen3Agent(model_dir=qwen3_model_dir))
        self.agents.append(DeepSeekAgent(model_dir=deepseek_model_dir))

        self.agent_weights = [a.weight for a in self.agents]

        # ✅ Load state (accuracy persistence)
        self._load_state()

        print("\n" + "=" * 60)
        print("🤖 AGENTIC LOOP INITIALIZED (PERSISTENT)")
        print("=" * 60)
        print(f"   Agents: {len(self.agents)}")
        for a in self.agents:
            status = "✅" if getattr(a, 'available', True) else "⚠️"
            acc = a.get_accuracy()
            print(f"      {status} {a.name:10s} | acc: {acc:.1%} | "
                  f"recent: {a.recent_accuracy:.1%} | trades: {a.total_predictions}")
        print("=" * 60)

    def _load_state(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            self.ensemble_correct = state.get('ensemble_correct', 0)
            self.ensemble_total = state.get('ensemble_total', 0)
            for agent_data in state.get('agents', []):
                for agent in self.agents:
                    if agent.name == agent_data.get('name'):
                        agent.from_state(agent_data)
            print(f"   ✅ State loaded: "
                  f"ensemble {self.ensemble_correct}/{self.ensemble_total}")
        except Exception as e:
            print(f"   ⚠️ State load failed: {e}")

    def save_state(self, path=STATE_FILE):
        state = {
            'timestamp': str(datetime.now()),
            'ensemble_total': self.ensemble_total,
            'ensemble_correct': self.ensemble_correct,
            'agents': [a.to_state() for a in self.agents]
        }
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"   ⚠️ Could not save state: {e}")

    def save_decision_log(self, path=DECISION_LOG_FILE):
        if not self.decision_log:
            return
        try:
            df = pd.DataFrame(self.decision_log)
            if 'agent_votes' in df.columns:
                df['agent_votes'] = df['agent_votes'].astype(str)
            df.to_csv(path, index=False)
        except Exception as e:
            print(f"   ⚠️ Decision log save failed: {e}")

    def _normalize_weights(self):
        total = sum(self.agent_weights)
        if total > 0:
            self.agent_weights = [w / total for w in self.agent_weights]

    def get_consensus(self, symbol, symbol_data, volatility, market_regime,
                      sector=None, atr=None, drawdown=0):
        votes = []
        agent_details = {}

        for agent in self.agents:
            if agent.name == "XGBoost":
                agent.set_symbol(symbol)

        for i, agent in enumerate(self.agents):
            try:
                if agent.name == "XGBoost":
                    prob, conf = agent.predict(self._get_features(symbol_data))
                elif agent.name == "Technical":
                    prob, conf = agent.analyze(symbol_data)
                elif agent.name == "Risk":
                    prob, conf = agent.assess(symbol, volatility, market_regime, atr, drawdown)
                elif agent.name == "Memory":
                    prob, conf = agent.get_similar_pattern(symbol, self._get_features(symbol_data))
                elif agent.name == "Qwen3":
                    prob, conf = agent.analyze(symbol, symbol_data)
                elif agent.name == "DeepSeek":
                    prob, conf = agent.analyze(symbol, symbol_data)
                else:
                    prob, conf = 0.5, 0.5
            except Exception:
                prob, conf = 0.5, 0.3

            dynamic_weight = agent.get_dynamic_weight()
            self.agent_weights[i] = dynamic_weight

            votes.append({
                'agent': agent.name,
                'score': prob,
                'confidence': conf,
                'weight': dynamic_weight
            })
            agent_details[agent.name] = {
                'score': round(prob, 3),
                'confidence': round(conf, 3),
                'weight': round(dynamic_weight, 3)
            }

        self._normalize_weights()
        for i, v in enumerate(votes):
            v['weight'] = self.agent_weights[i]

        total_w = sum(v['weight'] for v in votes)
        weighted_score = sum(v['score'] * v['weight'] for v in votes) / total_w if total_w > 0 else 0.5
        consensus_confidence = sum(v['confidence'] * v['weight'] for v in votes) / total_w if total_w > 0 else 0.5

        bullish_votes = sum(1 for v in votes if v['score'] > 0.55)
        bearish_votes = sum(1 for v in votes if v['score'] < 0.45)
        vote_agreement = max(bullish_votes, bearish_votes) / len(votes)

        if weighted_score >= 0.65 and vote_agreement >= 0.5:
            decision = 'STRONG_BUY'
        elif weighted_score >= 0.55:
            decision = 'BUY'
        elif weighted_score <= 0.35 and vote_agreement >= 0.5:
            decision = 'STRONG_SELL'
        elif weighted_score <= 0.45:
            decision = 'SELL'
        else:
            decision = 'HOLD'

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'decision': decision,
            'score': weighted_score,
            'confidence': consensus_confidence,
            'vote_agreement': vote_agreement,
            'agent_votes': agent_details
        }
        self.decision_log.append(log_entry)
        if len(self.decision_log) > 5000:
            self.decision_log = self.decision_log[-5000:]

        return decision, weighted_score, consensus_confidence, agent_details

    def after_trade_feedback(self, trade_result):
        """✅ FIXED: accuracy update reliably, works even without prior decision match"""
        symbol = trade_result.get('symbol')
        pnl = float(trade_result.get('pnl', 0))
        pnl_pct = float(trade_result.get('pnl_pct', 0))

        # Breakeven — skip
        if abs(pnl) < 0.01 and abs(pnl_pct) < 0.01:
            return None

        was_win = pnl > 0

        # ✅ Prefer explicit agent_predictions from caller
        agent_votes = trade_result.get('agent_predictions')

        # Fallback 1: last decision log for this symbol
        if not agent_votes:
            matching = [d for d in self.decision_log if d.get('symbol') == symbol]
            if matching:
                agent_votes = matching[-1].get('agent_votes', {})

        # Fallback 2: skip per-agent update (but still log ensemble)
        if not agent_votes:
            print(f"   ⚠️ No agent_votes for {symbol} — only ensemble will be updated")
            agent_votes = {}

        # Ensemble accuracy
        self.ensemble_total += 1
        if was_win:
            self.ensemble_correct += 1
        ensemble_accuracy = self.ensemble_correct / self.ensemble_total

        # Per-agent update
        updated_count = 0
        for agent in self.agents:
            if agent.name not in agent_votes:
                continue
            v = agent_votes[agent.name]
            score = float(v.get('score', 0.5))
            confidence = float(v.get('confidence', 0.5))

            if was_win:
                was_correct = score > 0.5
            else:
                was_correct = score <= 0.5

            agent.update_performance(was_correct, confidence)
            updated_count += 1

            if agent.name == "Risk":
                agent.update_history(symbol, not was_win)

        # Memory
        memory_agent = next((a for a in self.agents if a.name == "Memory"), None)
        if memory_agent:
            if 'features' not in trade_result:
                trade_result['features'] = self._get_features_from_trade(trade_result)
            memory_agent.remember_trade(trade_result)

        # Log
        self.performance_log.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'was_win': was_win,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'ensemble_accuracy': ensemble_accuracy
        })

        print(f"\n   📊 Feedback: {symbol} — "
              f"{'WIN ✅' if was_win else 'LOSS ❌'} "
              f"(PnL: {pnl:+.2f} | {pnl_pct:+.2f}%)")
        print(f"      Agents updated: {updated_count}/{len(self.agents)}")
        print(f"      Ensemble: {self.ensemble_correct}/{self.ensemble_total} "
              f"({ensemble_accuracy:.1%})")
        for a in self.agents:
            print(f"      {a.name:10s}: acc={a.get_accuracy():.1%} "
                  f"recent={a.recent_accuracy:.1%} n={a.total_predictions}")

        # ✅ Persist
        self.save_state()
        self.save_decision_log()
        self._save_performance_log()

        return {a.name: a.get_dynamic_weight() for a in self.agents}

    def record_trade_feedback(self, trade_result):
        """✅ Alias — backward compatible"""
        return self.after_trade_feedback(trade_result)

    def _save_performance_log(self):
        if not self.performance_log:
            return
        try:
            pd.DataFrame(self.performance_log).to_csv(PERFORMANCE_LOG_FILE, index=False)
        except:
            pass

    def _get_features(self, symbol_data):
        if len(symbol_data) < 10:
            return np.zeros((1, 15))

        f = []
        close = symbol_data['close'].iloc[-1]
        f.append(close if not pd.isna(close) else 100)

        vol = symbol_data['volume'].iloc[-1] if 'volume' in symbol_data.columns else 0
        f.append(vol / 1e6 if not pd.isna(vol) else 0)

        if len(symbol_data) >= 5:
            r = (symbol_data['close'].iloc[-1] - symbol_data['close'].iloc[-5]) / symbol_data['close'].iloc[-5]
            f.append(r if not pd.isna(r) else 0)
        else:
            f.append(0)

        if len(symbol_data) >= 10:
            r = (symbol_data['close'].iloc[-1] - symbol_data['close'].iloc[-10]) / symbol_data['close'].iloc[-10]
            f.append(r if not pd.isna(r) else 0)
        else:
            f.append(0)

        for col, default in [
            ('volatility', 0.02), ('volatility_5d', 0.02),
            ('volume_ratio', 1), ('rsi_oversold', 0), ('rsi_overbought', 0),
        ]:
            if col in symbol_data.columns:
                v = symbol_data[col].iloc[-1]
                f.append(v if not pd.isna(v) else default)
            else:
                f.append(default)

        for col, div in [
            ('dist_from_sr', 100), ('sr_strength', 3),
            ('is_bullish_div', 1), ('div_strength', 3),
            ('dist_from_ema', 100), ('above_ema', 1),
        ]:
            if col in symbol_data.columns:
                v = symbol_data[col].iloc[-1]
                f.append(v / div if not pd.isna(v) else 0)
            else:
                f.append(0)

        while len(f) < 15:
            f.append(0)
        return np.array(f[:15]).reshape(1, -1)

    def _get_features_from_trade(self, trade_result):
        return trade_result.get('features', np.zeros(15))

    def get_summary(self):
        rows = []
        for a in self.agents:
            rows.append({
                'agent': a.name,
                'accuracy': f"{a.get_accuracy():.1%}",
                'recent_accuracy': f"{a.recent_accuracy:.1%}",
                'total_predictions': a.total_predictions,
                'current_weight': f"{a.get_dynamic_weight():.2f}"
            })
        return pd.DataFrame(rows)

    def get_ensemble_accuracy(self):
        return self.ensemble_correct / self.ensemble_total if self.ensemble_total > 0 else 0.5


# =========================
# WALK-FORWARD OUTCOME HELPER
# =========================

def evaluate_prediction_outcome(symbol_df, prediction_date, lookahead=5):
    """
    Walk-forward outcome: prediction-এর পর N দিনের actual price।
    Training-এ feedback দিতে ব্যবহার করুন।
    """
    df = symbol_df.copy().reset_index(drop=True)
    if 'date' not in df.columns or len(df) < 2:
        return None

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    pred_date = pd.to_datetime(prediction_date, errors='coerce')
    if pd.isna(pred_date):
        return None

    mask = df['date'] >= pred_date
    if not mask.any():
        return None
    pred_idx = int(df[mask].index[0])
    future_idx = min(pred_idx + lookahead, len(df) - 1)
    if future_idx <= pred_idx:
        return None

    entry = float(df.iloc[pred_idx]['close'])
    exit_ = float(df.iloc[future_idx]['close'])
    if entry <= 0:
        return None

    pnl_pct = ((exit_ - entry) / entry) * 100
    return {
        'entry_price': entry,
        'exit_price': exit_,
        'pnl': exit_ - entry,
        'pnl_pct': pnl_pct,
        'actual_win': pnl_win := pnl_pct > 0,
        'success': pnl_win
    }


# =========================
# INTEGRATION
# =========================

def integrate_with_ppo(agentic_loop, trade_result):
    feedback = agentic_loop.after_trade_feedback(trade_result)
    if feedback:
        avg_acc = np.mean([a.get_accuracy() for a in agentic_loop.agents])
        ens_acc = agentic_loop.get_ensemble_accuracy()
        combined = (avg_acc + ens_acc) / 2
        if combined > 0.6:
            return 1.2, {'agent_accuracy': avg_acc, 'ensemble_accuracy': ens_acc}
        elif combined < 0.4:
            return 0.8, {'agent_accuracy': avg_acc, 'ensemble_accuracy': ens_acc}
    return 1.0, {}


# =========================
# TEST
# =========================

if __name__ == "__main__":
    print("\n🧪 TESTING AGENTIC LOOP (persistent)")

    loop = AgenticLoop()

    sample = pd.DataFrame({
        'close': [100 + i for i in range(20)],
        'volume': [1000 + i * 100 for i in range(20)],
        'rsi': [25 + i * 2 for i in range(20)],
        'volatility': [0.02] * 20,
        'bb_position': [0.1 + i * 0.04 for i in range(20)],
        'dist_from_sr': [1 - i * 0.1 for i in range(20)],
        'is_support': [1] * 10 + [0] * 10,
        'is_bullish_div': [0] * 19 + [1],
        'is_bearish_div': [0] * 20,
        'div_strength': [0] * 19 + [2],
        'dist_from_ema': [5 - i * 0.5 for i in range(20)],
        'above_ema': [1] * 15 + [0] * 5,
        'sr_strength': [2] * 20,
        'macd': [-0.5 + i * 0.05 for i in range(20)],
        'macd_signal': [-0.4 + i * 0.04 for i in range(20)],
        'macd_hist': [0.1] * 20,
    })

    decision, score, confidence, details = loop.get_consensus(
        symbol="TEST", symbol_data=sample,
        volatility=0.02, market_regime="BULL", atr=2.5, drawdown=0.03
    )
    print(f"\n📊 Decision: {decision} | Score: {score:.3f} | Conf: {confidence:.3f}")

    print("\n📝 Feedback test...")
    loop.after_trade_feedback({
        'symbol': 'TEST',
        'pnl': 500.0,
        'pnl_pct': 5.0,
        'success': True,
        'agent_predictions': details,  # ✅ explicit
        'features': np.random.randn(15)
    })

    print("\n📊 Summary:")
    print(loop.get_summary().to_string())
    print(f"\n📈 Ensemble: {loop.get_ensemble_accuracy():.1%}")