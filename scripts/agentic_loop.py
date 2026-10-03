# agentic_loop.py - Multi-Agent Voting System with LLM Integration
# ✅ FIXED: Breakeven handling, SELL close logic, pnl_pct display
# ✅ REMOVED: News Agent, Sector Agent
# ✅ ADDED: Qwen3 Agent, DeepSeek Agent (real LLM inference)
# ✅ Weight redistribution: Qwen3=0.12, DeepSeek=0.08

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

# Try importing transformers for LLM agents
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available — LLM agents disabled")


# =========================
# TELEGRAM NOTIFICATION
# =========================

def send_telegram_message(message, token=None, chat_id=None):
    """Send message to Telegram"""
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
# BASE TRADING AGENT
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

    def update_performance(self, was_correct, confidence):
        self.total_predictions += 1
        if was_correct:
            self.correct_predictions += 1
        self.performance_history.append({
            'timestamp': datetime.now(),
            'correct': was_correct,
            'confidence': confidence
        })

        if len(self.performance_history) >= 20:
            recent = self.performance_history[-20:]
            self.recent_accuracy = sum(1 for p in recent if p['correct']) / len(recent)
        elif len(self.performance_history) > 0:
            self.recent_accuracy = self.correct_predictions / self.total_predictions

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


# =========================
# XGBOOST AGENT
# =========================

class XGBoostAgent(TradingAgent):
    """XGBoost model as an agent"""

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
                        model_path = os.path.join(self.model_dir, file)
                        self.models[symbol] = joblib.load(model_path)
                    except Exception as e:
                        print(f"   ⚠️ Failed to load {symbol}: {e}")

                if self.models:
                    print(f"   ✅ XGBoost Agent loaded {len(self.models)} models")
                    self._load_model_quality()
                else:
                    print(f"   ⚠️ No XGBoost models found in {self.model_dir}")
            else:
                print(f"   ⚠️ XGBoost model directory not found: {self.model_dir}")
        except Exception as e:
            print(f"   ⚠️ XGBoost Agent init failed: {e}")

    def _load_model_quality(self):
        metadata_path = './csv/model_metadata.csv'
        if os.path.exists(metadata_path):
            try:
                metadata = pd.read_csv(metadata_path)
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
    """Technical analysis agent (RSI, MACD, Bollinger, S/R, Divergence)"""

    def __init__(self):
        super().__init__("Technical", weight=0.20)

    def analyze(self, symbol_data):
        if len(symbol_data) < 20:
            return 0.5, 0.3

        signals = []
        confidences = []

        # RSI Analysis
        if 'rsi' in symbol_data.columns:
            rsi = symbol_data['rsi'].iloc[-1]
            if not pd.isna(rsi):
                if rsi < 30:
                    signals.append(1)
                    confidences.append(min(0.8, (30 - rsi) / 30 + 0.3))
                elif rsi > 70:
                    signals.append(0)
                    confidences.append(min(0.8, (rsi - 70) / 30 + 0.3))
                else:
                    signals.append(0.5)
                    confidences.append(0.4)

        # MACD Analysis
        if 'macd' in symbol_data.columns and 'macd_signal' in symbol_data.columns:
            macd = symbol_data['macd'].iloc[-1]
            signal = symbol_data['macd_signal'].iloc[-1]
            macd_hist = symbol_data['macd_hist'].iloc[-1] if 'macd_hist' in symbol_data.columns else 0

            if not pd.isna(macd) and not pd.isna(signal):
                if macd > signal and macd_hist > 0:
                    signals.append(1)
                    confidences.append(0.7)
                elif macd < signal and macd_hist < 0:
                    signals.append(0)
                    confidences.append(0.7)
                else:
                    signals.append(0.5)
                    confidences.append(0.4)

        # Bollinger Bands
        if 'bb_position' in symbol_data.columns:
            bb_pos = symbol_data['bb_position'].iloc[-1]
            if not pd.isna(bb_pos):
                if bb_pos < 0.2:
                    signals.append(1)
                    confidences.append(0.6)
                elif bb_pos > 0.8:
                    signals.append(0)
                    confidences.append(0.6)

        # Support/Resistance
        if 'dist_from_sr' in symbol_data.columns and 'is_support' in symbol_data.columns:
            dist_sr = symbol_data['dist_from_sr'].iloc[-1]
            is_support = symbol_data['is_support'].iloc[-1]

            if not pd.isna(dist_sr) and not pd.isna(is_support):
                if is_support == 1 and abs(dist_sr) < 2:
                    signals.append(1)
                    confidences.append(0.65)
                elif is_support == 0 and abs(dist_sr) < 2:
                    signals.append(0)
                    confidences.append(0.65)

        # RSI Divergence
        if 'is_bullish_div' in symbol_data.columns:
            if symbol_data['is_bullish_div'].iloc[-1] == 1:
                signals.append(1)
                confidences.append(0.75)
        if 'is_bearish_div' in symbol_data.columns:
            if symbol_data['is_bearish_div'].iloc[-1] == 1:
                signals.append(0)
                confidences.append(0.75)

        if not signals:
            return 0.5, 0.3

        weighted_score = sum(s * c for s, c in zip(signals, confidences)) / sum(confidences)
        avg_confidence = sum(confidences) / len(confidences)
        return weighted_score, avg_confidence


# =========================
# RISK AGENT
# =========================

class RiskAgent(TradingAgent):
    """Risk management agent"""

    def __init__(self):
        super().__init__("Risk", weight=0.15)
        self.symbol_history = defaultdict(lambda: {'trades': 0, 'losses': 0, 'consecutive_losses': 0})

    def assess(self, symbol, volatility, market_regime, atr=None, drawdown=0):
        risk_score = 0.5

        if volatility > 0.03:
            risk_score -= 0.2
        elif volatility < 0.01:
            risk_score += 0.1

        if market_regime == 'BEAR':
            risk_score -= 0.25
        elif market_regime == 'BULL':
            risk_score += 0.15

        if atr is not None:
            atr_ratio = atr / 100
            if atr_ratio > 0.03:
                risk_score -= 0.15
            elif atr_ratio < 0.01:
                risk_score += 0.1

        if drawdown > 0.1:
            risk_score -= 0.2
        elif drawdown > 0.05:
            risk_score -= 0.1

        hist = self.symbol_history[symbol]
        if hist['consecutive_losses'] >= 2:
            risk_score -= 0.15
        elif hist['trades'] > 10 and hist['losses'] / hist['trades'] > 0.6:
            risk_score -= 0.2

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
# QWEN3 AGENT (NEW)
# =========================

class Qwen3Agent(TradingAgent):
    """Qwen3-0.6B LLM as an agent for trading signals"""

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
            print(f"   ⚠️ Qwen3 Agent: transformers not available")
            return

        if not os.path.exists(os.path.join(self.model_dir, "config.json")):
            print(f"   ⚠️ Qwen3 Agent: model not found at {self.model_dir}")
            return

        try:
            print(f"   🔄 Loading Qwen3 from {self.model_dir}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.available = True
            print(f"   ✅ Qwen3 Agent loaded on {self.device}")
        except Exception as e:
            print(f"   ⚠️ Qwen3 Agent load failed: {e}")
            self.available = False

    def analyze(self, symbol, symbol_data):
        """Analyze using Qwen3 LLM"""
        if not self.available or len(symbol_data) < 5:
            return 0.5, 0.3

        try:
            latest = symbol_data.iloc[-1]
            close = float(latest.get('close', 0))
            rsi = float(latest.get('rsi', 50)) if not pd.isna(latest.get('rsi', 50)) else 50
            macd = float(latest.get('macd', 0)) if not pd.isna(latest.get('macd', 0)) else 0

            prompt = f"""Symbol: {symbol} | Price: {close:.2f}
RSI: {rsi:.1f} | MACD: {macd:.4f}

Provide trading signal (BUY/SELL/HOLD) with confidence%.

RECOMMENDATION:"""

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Parse signal
            score = 0.5
            confidence = 0.4

            if re.search(r'\bBUY\b', response, re.IGNORECASE):
                score = 0.75
                confidence = 0.6
            elif re.search(r'\bSELL\b', response, re.IGNORECASE):
                score = 0.25
                confidence = 0.6
            else:
                score = 0.5
                confidence = 0.3

            # Extract confidence from response
            conf_match = re.search(r'(\d+)%', response)
            if conf_match:
                extracted_conf = float(conf_match.group(1)) / 100.0
                if 0.3 <= extracted_conf <= 0.95:
                    confidence = extracted_conf

            return score, confidence
        except Exception as e:
            return 0.5, 0.3


# =========================
# DEEPSEEK AGENT (NEW)
# =========================

class DeepSeekAgent(TradingAgent):
    """DeepSeek-R1-Distill-Qwen-1.5B LLM as an agent"""

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
            print(f"   ⚠️ DeepSeek Agent: transformers not available")
            return

        if not os.path.exists(os.path.join(self.model_dir, "config.json")):
            print(f"   ⚠️ DeepSeek Agent: model not found at {self.model_dir}")
            return

        try:
            print(f"   🔄 Loading DeepSeek from {self.model_dir}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.available = True
            print(f"   ✅ DeepSeek Agent loaded on {self.device}")
        except Exception as e:
            print(f"   ⚠️ DeepSeek Agent load failed: {e}")
            self.available = False

    def analyze(self, symbol, symbol_data):
        """Analyze using DeepSeek LLM"""
        if not self.available or len(symbol_data) < 5:
            return 0.5, 0.3

        try:
            latest = symbol_data.iloc[-1]
            close = float(latest.get('close', 0))
            rsi = float(latest.get('rsi', 50)) if not pd.isna(latest.get('rsi', 50)) else 50
            macd = float(latest.get('macd', 0)) if not pd.isna(latest.get('macd', 0)) else 0

            prompt = f"""Symbol: {symbol} | Price: {close:.2f}
RSI: {rsi:.1f} | MACD: {macd:.4f}

Analyze and provide trading signal (BUY/SELL/HOLD) with confidence%.

SIGNAL:"""

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Parse signal
            score = 0.5
            confidence = 0.4

            if re.search(r'\bBUY\b', response, re.IGNORECASE):
                score = 0.75
                confidence = 0.6
            elif re.search(r'\bSELL\b', response, re.IGNORECASE):
                score = 0.25
                confidence = 0.6
            else:
                score = 0.5
                confidence = 0.3

            conf_match = re.search(r'(\d+)%', response)
            if conf_match:
                extracted_conf = float(conf_match.group(1)) / 100.0
                if 0.3 <= extracted_conf <= 0.95:
                    confidence = extracted_conf

            return score, confidence
        except Exception as e:
            return 0.5, 0.3


# =========================
# MEMORY AGENT
# =========================

class MemoryAgent(TradingAgent):
    """Memory agent - learns from past mistakes"""

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

            if win_rate > 0.55:
                return 0.65, confidence
            elif win_rate < 0.45:
                return 0.35, confidence

        return 0.5, 0.3


# =========================
# MAIN AGENTIC LOOP
# =========================

class AgenticLoop:
    """Main Agentic Loop system - coordinates all agents"""

    def __init__(self, xgb_model_dir='./csv/xgboost/',
                 qwen3_model_dir='./csv/llm_model_qwen3',
                 deepseek_model_dir='./csv/llm_model_deepseek'):
        self.agents = []
        self.vote_history = []
        self.decision_log = []
        self.performance_log = []

        # ✅ Initialize agents (NO Sector, NO News)
        self.agents.append(XGBoostAgent(xgb_model_dir))
        self.agents.append(TechnicalAgent())
        self.agents.append(RiskAgent())
        self.agents.append(MemoryAgent())
        self.agents.append(Qwen3Agent(model_dir=qwen3_model_dir))       # ✅ NEW
        self.agents.append(DeepSeekAgent(model_dir=deepseek_model_dir)) # ✅ NEW

        self.agent_weights = self._get_initial_weights()
        self.ensemble_correct = 0
        self.ensemble_total = 0

        self._load_state()

        print("\n" + "=" * 60)
        print("🤖 AGENTIC LOOP INITIALIZED (LLM-Enhanced)")
        print("=" * 60)
        print(f"   Agents: {len(self.agents)}")
        for agent in self.agents:
            status = "✅" if getattr(agent, 'available', True) else "⚠️"
            print(f"      {status} {agent.name} (weight: {agent.weight})")
        print("=" * 60)

    def _load_state(self):
        state_file = './csv/agentic_loop_state.json'
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                for agent_data in state.get('agents', []):
                    for agent in self.agents:
                        if agent.name == agent_data['name']:
                            agent.correct_predictions = agent_data.get('correct', 0)
                            agent.total_predictions = agent_data.get('total', 0)
            except:
                pass

    def _get_initial_weights(self):
        return [agent.weight for agent in self.agents]

    def _normalize_weights(self):
        total = sum(self.agent_weights)
        if total > 0:
            self.agent_weights = [w / total for w in self.agent_weights]

    def get_consensus(self, symbol, symbol_data, volatility, market_regime,
                      sector=None, atr=None, drawdown=0):
        """
        Get consensus decision from all agents
        Note: `sector` param kept for backwards compatibility — ignored
        """
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
            except Exception as e:
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

        for i, vote in enumerate(votes):
            vote['weight'] = self.agent_weights[i]

        total_weight = sum(v['weight'] for v in votes)
        if total_weight > 0:
            weighted_score = sum(v['score'] * v['weight'] for v in votes) / total_weight
        else:
            weighted_score = 0.5

        consensus_confidence = sum(v['confidence'] * v['weight'] for v in votes) / total_weight if total_weight > 0 else 0.5

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
            'timestamp': datetime.now(),
            'symbol': symbol,
            'decision': decision,
            'score': weighted_score,
            'confidence': consensus_confidence,
            'vote_agreement': vote_agreement,
            'agent_votes': agent_details
        }
        self.decision_log.append(log_entry)

        if decision in ['STRONG_BUY', 'STRONG_SELL'] and consensus_confidence > 0.65:
            self._send_signal_alert(log_entry)

        return decision, weighted_score, consensus_confidence, agent_details

    def _send_signal_alert(self, log_entry):
        message = f"""
🚨 <b>Strong Signal Detected!</b>
📊 Symbol: {log_entry['symbol']}
🎯 Decision: {log_entry['decision']}
📈 Score: {log_entry['score']:.3f}
💪 Confidence: {log_entry['confidence']:.3f}
🤝 Vote Agreement: {log_entry['vote_agreement']:.0%}
"""
        send_telegram_message(message)

    def after_trade_feedback(self, trade_result):
        """Update agents based on trade outcome"""
        symbol = trade_result.get('symbol')
        pnl = trade_result.get('pnl', 0)
        pnl_pct = trade_result.get('pnl_pct', 0)

        if abs(pnl) < 0.01 and abs(pnl_pct) < 0.01:
            return None

        was_win = pnl > 0
        ppo_action = trade_result.get('ppo_action', None)
        recent_decisions = [d for d in self.decision_log if d['symbol'] == symbol]
        if not recent_decisions:
            return None

        last_decision = recent_decisions[-1]
        agent_votes = last_decision.get('agent_votes', {})
        ensemble_was_correct = was_win

        self.ensemble_total += 1
        if ensemble_was_correct:
            self.ensemble_correct += 1

        for agent in self.agents:
            if agent.name in agent_votes:
                agent_score = agent_votes[agent.name]['score']

                if was_win:
                    was_correct = (agent_score > 0.5)
                else:
                    was_correct = (agent_score <= 0.5)

                confidence = agent_votes[agent.name]['confidence']
                agent.update_performance(was_correct, confidence)

                if agent.name == "Risk":
                    agent.update_history(symbol, not was_win)

        memory_agent = next((a for a in self.agents if a.name == "Memory"), None)
        if memory_agent:
            if 'features' not in trade_result:
                trade_result['features'] = self._get_features_from_trade(trade_result)
            memory_agent.remember_trade(trade_result)

        ensemble_accuracy = self.ensemble_correct / self.ensemble_total if self.ensemble_total > 0 else 0.5
        self.performance_log.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'was_win': was_win,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'ppo_action': ppo_action,
            'ensemble_correct': ensemble_was_correct,
            'ensemble_accuracy': ensemble_accuracy
        })

        if pnl_pct == 0:
            entry_price = trade_result.get('entry_price', 0)
            exit_price = trade_result.get('exit_price', 0)
            if entry_price > 0 and exit_price > 0:
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100

        print(f"\n   📊 Agent Feedback for {symbol}:")
        print(f"      Trade Result: {'WIN ✅' if was_win else 'LOSS ❌'} (PnL: {pnl:+.2f} Tk | {pnl_pct:+.2f}%)")
        print(f"      Ensemble Correct: {'✅' if ensemble_was_correct else '❌'}")
        print(f"      Ensemble Accuracy: {ensemble_accuracy:.1%}")
        print(f"      Updating {len(self.agents)} agents...")

        # ✅ Auto-save state
        self.save_state()
        self.save_decision_log()

        return {a.name: a.get_dynamic_weight() for a in self.agents}

    def _get_features(self, symbol_data):
        """Extract 15-dim features for XGBoost"""
        if len(symbol_data) < 10:
            return np.zeros((1, 15))

        features = []

        close_price = symbol_data['close'].iloc[-1]
        features.append(close_price if not pd.isna(close_price) else 100)

        volume = symbol_data['volume'].iloc[-1] if 'volume' in symbol_data.columns else 0
        features.append(volume / 1e6 if not pd.isna(volume) else 0)

        if len(symbol_data) >= 5:
            ret_5d = (symbol_data['close'].iloc[-1] - symbol_data['close'].iloc[-5]) / symbol_data['close'].iloc[-5]
            features.append(ret_5d if not pd.isna(ret_5d) else 0)
        else:
            features.append(0)

        if len(symbol_data) >= 10:
            ret_10d = (symbol_data['close'].iloc[-1] - symbol_data['close'].iloc[-10]) / symbol_data['close'].iloc[-10]
            features.append(ret_10d if not pd.isna(ret_10d) else 0)
        else:
            features.append(0)

        if 'volatility' in symbol_data.columns:
            vol = symbol_data['volatility'].iloc[-1]
            features.append(vol if not pd.isna(vol) else 0.02)
        else:
            features.append(0.02)

        if 'volatility_5d' in symbol_data.columns:
            vol_5d = symbol_data['volatility_5d'].iloc[-1]
            features.append(vol_5d if not pd.isna(vol_5d) else 0.02)
        else:
            features.append(0.02)

        if 'volume_ratio' in symbol_data.columns:
            vol_ratio = symbol_data['volume_ratio'].iloc[-1]
            features.append(vol_ratio if not pd.isna(vol_ratio) else 1)
        else:
            features.append(1)

        if 'rsi_oversold' in symbol_data.columns:
            features.append(symbol_data['rsi_oversold'].iloc[-1])
        else:
            features.append(0)

        if 'rsi_overbought' in symbol_data.columns:
            features.append(symbol_data['rsi_overbought'].iloc[-1])
        else:
            features.append(0)

        if 'dist_from_sr' in symbol_data.columns:
            features.append(symbol_data['dist_from_sr'].iloc[-1] / 100)
        else:
            features.append(0)

        if 'sr_strength' in symbol_data.columns:
            features.append(symbol_data['sr_strength'].iloc[-1] / 3)
        else:
            features.append(0)

        if 'is_bullish_div' in symbol_data.columns:
            features.append(symbol_data['is_bullish_div'].iloc[-1])
        else:
            features.append(0)

        if 'div_strength' in symbol_data.columns:
            features.append(symbol_data['div_strength'].iloc[-1] / 3)
        else:
            features.append(0)

        if 'dist_from_ema' in symbol_data.columns:
            features.append(symbol_data['dist_from_ema'].iloc[-1] / 100)
        else:
            features.append(0)

        if 'above_ema' in symbol_data.columns:
            features.append(symbol_data['above_ema'].iloc[-1])
        else:
            features.append(0)

        while len(features) < 15:
            features.append(0)

        return np.array(features[:15]).reshape(1, -1)

    def _get_features_from_trade(self, trade_result):
        return trade_result.get('features', np.zeros(15))

    def get_summary(self):
        summary = []
        for agent in self.agents:
            summary.append({
                'agent': agent.name,
                'accuracy': f"{agent.get_accuracy():.1%}",
                'recent_accuracy': f"{agent.recent_accuracy:.1%}",
                'total_predictions': agent.total_predictions,
                'current_weight': f"{agent.get_dynamic_weight():.2f}"
            })
        return pd.DataFrame(summary)

    def get_ensemble_accuracy(self):
        if self.ensemble_total == 0:
            return 0.5
        return self.ensemble_correct / self.ensemble_total

    def save_decision_log(self, path='./csv/agentic_loop_log.csv'):
        if self.decision_log:
            try:
                df = pd.DataFrame(self.decision_log)
                if 'agent_votes' in df.columns:
                    df['agent_votes'] = df['agent_votes'].astype(str)
                df.to_csv(path, index=False)
            except Exception as e:
                print(f"   ⚠️ Could not save decision log: {e}")

    def save_state(self, path='./csv/agentic_loop_state.json'):
        state = {
            'timestamp': str(datetime.now()),
            'ensemble_total': self.ensemble_total,
            'ensemble_correct': self.ensemble_correct,
            'agents': []
        }

        for agent in self.agents:
            state['agents'].append({
                'name': agent.name,
                'correct': agent.correct_predictions,
                'total': agent.total_predictions,
                'weight': agent.weight
            })

        try:
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"   ⚠️ Could not save state: {e}")


# =========================================================
# INTEGRATION WITH PPO
# =========================================================

def integrate_with_ppo(agentic_loop, trade_result):
    """Integrate Agentic Loop feedback with PPO training"""
    feedback = agentic_loop.after_trade_feedback(trade_result)

    if feedback:
        avg_agent_accuracy = np.mean([a.get_accuracy() for a in agentic_loop.agents])
        ensemble_accuracy = agentic_loop.get_ensemble_accuracy()
        combined_accuracy = (avg_agent_accuracy + ensemble_accuracy) / 2

        if combined_accuracy > 0.6:
            return 1.2, {'agent_accuracy': avg_agent_accuracy, 'ensemble_accuracy': ensemble_accuracy}
        elif combined_accuracy < 0.4:
            return 0.8, {'agent_accuracy': avg_agent_accuracy, 'ensemble_accuracy': ensemble_accuracy}

    return 1.0, {}


# =========================================================
# QUICK TEST
# =========================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 TESTING AGENTIC LOOP (with Qwen3 + DeepSeek)")
    print("=" * 60)

    loop = AgenticLoop()

    sample_data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
        'rsi': [25, 28, 30, 32, 35, 40, 45, 50, 55, 60, 65],
        'volatility': [0.02, 0.02, 0.015, 0.015, 0.01, 0.01, 0.012, 0.012, 0.01, 0.01, 0.008],
        'bb_position': [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85],
        'dist_from_sr': [1, 0.5, 0, -0.5, -1, -1.5, -2, -1.5, -1, -0.5, 0],
        'is_support': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        'is_bullish_div': [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        'is_bearish_div': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'div_strength': [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        'dist_from_ema': [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
        'above_ema': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        'sr_strength': [2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1],
        'macd': [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        'macd_signal': [-0.4, -0.2, 0.0, 0.2, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2],
        'macd_hist': [-0.1, -0.1, -0.1, -0.1, -0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    })

    decision, score, confidence, details = loop.get_consensus(
        symbol="TEST",
        symbol_data=sample_data,
        volatility=0.02,
        market_regime="BULL",
        atr=2.5,
        drawdown=0.03
    )

    print(f"\n   📊 Decision: {decision}")
    print(f"   Score: {score:.3f} | Confidence: {confidence:.3f}")

    print("\n   📝 Testing feedback loop...")
    loop.after_trade_feedback({
        'symbol': 'TEST',
        'pnl': 500.0,
        'pnl_pct': 5.0,
        'success': True,
        'ppo_action': 1,
        'features': np.random.randn(15)
    })

    print("\n   📊 Agent Performance Summary:")
    print(loop.get_summary().to_string())
    print(f"\n   📈 Ensemble Accuracy: {loop.get_ensemble_accuracy():.1%}")

    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE!")
    print("=" * 60)