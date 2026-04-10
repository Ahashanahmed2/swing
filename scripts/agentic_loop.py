# agentic_loop.py - Multi-Agent Voting System for Trading (with Sector Features)
# This integrates with your existing XGBoost + PPO system
# ✅ NEW: Sector Agent, Ensemble Weight Optimization, Performance Tracking, Telegram Notifications

import pandas as pd
import numpy as np
import os
import joblib
import json
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =========================
# TELEGRAM NOTIFICATION (NEW)
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
        self.recent_accuracy = 0.5  # ✅ NEW: Track recent performance

    def update_performance(self, was_correct, confidence):
        self.total_predictions += 1
        if was_correct:
            self.correct_predictions += 1
        self.performance_history.append({
            'timestamp': datetime.now(),
            'correct': was_correct,
            'confidence': confidence
        })
        
        # ✅ NEW: Calculate recent accuracy (last 20 predictions)
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
        """Dynamic weight based on recent performance"""
        base_weight = self.weight
        
        # ✅ NEW: Use recent accuracy for faster adaptation
        accuracy = self.recent_accuracy if self.total_predictions >= 10 else self.get_accuracy()

        # Boost weight if accurate, reduce if not
        if accuracy > 0.6:
            return base_weight * (1 + (accuracy - 0.6) * 1.5)
        elif accuracy < 0.4:
            return base_weight * (0.5 + accuracy * 0.5)
        return base_weight


# =========================
# XGBOOST AGENT
# =========================

class XGBoostAgent(TradingAgent):
    """Your existing XGBoost model as an agent - FIXED VERSION"""
    def __init__(self, xgb_model_dir):
        super().__init__("XGBoost", weight=0.35)  # ✅ Adjusted weight
        self.model_dir = xgb_model_dir
        self.models = {}  # Dictionary of symbol -> model
        self.current_symbol = None
        self.model_auc_scores = {}  # ✅ NEW: Track model quality
        self.load_models()

    def load_models(self):
        """Load all XGBoost models from directory"""
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
                    print(f"   ✅ XGBoost Agent loaded {len(self.models)} models from {self.model_dir}")
                    
                    # ✅ NEW: Load AUC scores from metadata if available
                    self._load_model_quality()
                else:
                    print(f"   ⚠️ No XGBoost models found in {self.model_dir}")
            else:
                print(f"   ⚠️ XGBoost model directory not found: {self.model_dir}")
        except Exception as e:
            print(f"   ⚠️ XGBoost Agent init failed: {e}")
    
    def _load_model_quality(self):
        """✅ NEW: Load model quality metrics from metadata"""
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
        """Set current symbol for prediction"""
        self.current_symbol = symbol

    def predict(self, features):
        """Predict using symbol-specific model"""
        if not self.models or self.current_symbol not in self.models:
            return 0.5, 0.3  # Lower confidence when no model
        
        try:
            model = self.models[self.current_symbol]
            prob = model.predict_proba(features)[0, 1]
            
            # ✅ NEW: Adjust confidence based on model AUC
            base_confidence = 0.6
            if self.current_symbol in self.model_auc_scores:
                auc = self.model_auc_scores[self.current_symbol]
                base_confidence = min(0.9, max(0.4, auc))
            
            return prob, base_confidence
        except Exception as e:
            return 0.5, 0.3


# =========================
# TECHNICAL AGENT
# =========================

class TechnicalAgent(TradingAgent):
    """Technical analysis agent (RSI, MACD, Bollinger, Support/Resistance)"""
    def __init__(self):
        super().__init__("Technical", weight=0.20)

    def analyze(self, symbol_data):
        """Analyze technical indicators"""
        if len(symbol_data) < 20:
            return 0.5, 0.3

        signals = []
        confidences = []

        # RSI Analysis
        if 'rsi' in symbol_data.columns:
            rsi = symbol_data['rsi'].iloc[-1]
            if not pd.isna(rsi):
                if rsi < 30:
                    signals.append(1)  # Oversold → Buy signal
                    confidences.append(min(0.8, (30 - rsi) / 30 + 0.3))
                elif rsi > 70:
                    signals.append(0)  # Overbought → Sell signal
                    confidences.append(min(0.8, (rsi - 70) / 30 + 0.3))
                else:
                    # Neutral zone
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

        # Bollinger Bands Analysis
        if 'bb_position' in symbol_data.columns:
            bb_pos = symbol_data['bb_position'].iloc[-1]
            if not pd.isna(bb_pos):
                if bb_pos < 0.2:  # Near lower band
                    signals.append(1)
                    confidences.append(0.6)
                elif bb_pos > 0.8:  # Near upper band
                    signals.append(0)
                    confidences.append(0.6)

        # ✅ NEW: Support/Resistance Analysis
        if 'dist_from_sr' in symbol_data.columns and 'is_support' in symbol_data.columns:
            dist_sr = symbol_data['dist_from_sr'].iloc[-1]
            is_support = symbol_data['is_support'].iloc[-1]
            
            if not pd.isna(dist_sr) and not pd.isna(is_support):
                if is_support == 1 and abs(dist_sr) < 2:
                    signals.append(1)  # Near support → Buy
                    confidences.append(0.65)
                elif is_support == 0 and abs(dist_sr) < 2:
                    signals.append(0)  # Near resistance → Sell
                    confidences.append(0.65)

        # ✅ NEW: RSI Divergence
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

        # Weighted average of signals
        weighted_score = sum(s * c for s, c in zip(signals, confidences)) / sum(confidences)
        avg_confidence = sum(confidences) / len(confidences)

        return weighted_score, avg_confidence


# =========================
# RISK AGENT
# =========================

class RiskAgent(TradingAgent):
    """Risk management agent - position sizing and stop-loss"""
    def __init__(self):
        super().__init__("Risk", weight=0.15)
        self.symbol_history = defaultdict(lambda: {'trades': 0, 'losses': 0, 'consecutive_losses': 0})

    def assess(self, symbol, volatility, market_regime, atr=None, drawdown=0):
        """Assess risk level"""
        risk_score = 0.5

        # Adjust based on volatility
        if volatility > 0.03:
            risk_score -= 0.2
        elif volatility < 0.01:
            risk_score += 0.1

        # Adjust based on market regime
        if market_regime == 'BEAR':
            risk_score -= 0.25
        elif market_regime == 'BULL':
            risk_score += 0.15

        # ✅ NEW: Adjust based on ATR (volatility measure)
        if atr is not None:
            atr_ratio = atr / 100  # Normalize
            if atr_ratio > 0.03:
                risk_score -= 0.15
            elif atr_ratio < 0.01:
                risk_score += 0.1

        # ✅ NEW: Adjust based on current drawdown
        if drawdown > 0.1:  # 10% drawdown
            risk_score -= 0.2
        elif drawdown > 0.05:  # 5% drawdown
            risk_score -= 0.1

        # Adjust based on symbol history
        hist = self.symbol_history[symbol]
        if hist['consecutive_losses'] >= 2:
            risk_score -= 0.15
        elif hist['trades'] > 10 and hist['losses'] / hist['trades'] > 0.6:
            risk_score -= 0.2

        confidence = 0.7 + (abs(risk_score - 0.5) * 0.3)
        return max(0.1, min(0.9, risk_score)), confidence

    def update_history(self, symbol, was_loss):
        """✅ NEW: Update symbol trading history"""
        hist = self.symbol_history[symbol]
        hist['trades'] += 1
        if was_loss:
            hist['losses'] += 1
            hist['consecutive_losses'] += 1
        else:
            hist['consecutive_losses'] = 0


# =========================
# SECTOR AGENT (NEW)
# =========================

class SectorAgent(TradingAgent):
    """Sector analysis agent - tracks sector momentum and rotation"""
    def __init__(self):
        super().__init__("Sector", weight=0.15)
        self.sector_data = {}
        self.sector_momentum = {}
        self.sector_ranks = {}
        self.load_sector_data()
    
    def load_sector_data(self):
        """Load sector performance data"""
        sector_file = './csv/sector_performance.csv'
        if os.path.exists(sector_file):
            try:
                df = pd.read_csv(sector_file)
                for _, row in df.iterrows():
                    sector = row.get('sector', 'Unknown')
                    self.sector_momentum[sector] = row.get('momentum', 0)
                    self.sector_ranks[sector] = row.get('rank', 0)
            except:
                pass
    
    def analyze(self, symbol, sector, current_price=None):
        """Analyze sector strength for a symbol"""
        if sector == 'Unknown' or sector not in self.sector_momentum:
            return 0.5, 0.3
        
        momentum = self.sector_momentum.get(sector, 0)
        rank = self.sector_ranks.get(sector, 999)
        total_sectors = len(self.sector_ranks) if self.sector_ranks else 1
        
        # Normalize rank (higher is better)
        rank_score = 1 - (rank / total_sectors) if total_sectors > 0 else 0.5
        
        # Combine momentum and rank
        if momentum > 0.03:
            score = 0.65 + min(momentum * 2, 0.25)
            confidence = 0.7
        elif momentum < -0.03:
            score = 0.35 - min(abs(momentum) * 2, 0.25)
            confidence = 0.7
        else:
            score = 0.5
            confidence = 0.4
        
        # Adjust by rank
        score = score * 0.7 + rank_score * 0.3
        
        return max(0.1, min(0.9, score)), confidence


# =========================
# NEWS AGENT
# =========================

class NewsAgent(TradingAgent):
    """Sentiment analysis from news (placeholder for now)"""
    def __init__(self):
        super().__init__("News", weight=0.05)  # ✅ Reduced weight

    def analyze_sentiment(self, symbol):
        """Analyze news sentiment for symbol"""
        # TODO: Integrate with actual news API
        return 0.5, 0.3


# =========================
# MEMORY AGENT
# =========================

class MemoryAgent(TradingAgent):
    """Memory agent - learns from past mistakes"""
    def __init__(self):
        super().__init__("Memory", weight=0.10)  # ✅ Adjusted weight
        self.mistake_memory = []
        self.success_memory = []
        self.pattern_memory = defaultdict(list)

    def remember_trade(self, trade_result):
        trade_result['timestamp'] = datetime.now()
        if trade_result.get('pnl', 0) < 0:
            self.mistake_memory.append(trade_result)
        else:
            self.success_memory.append(trade_result)
        
        # ✅ NEW: Store pattern features
        if 'features' in trade_result:
            symbol = trade_result.get('symbol', 'UNKNOWN')
            self.pattern_memory[symbol].append({
                'features': trade_result['features'],
                'pnl': trade_result['pnl'],
                'timestamp': datetime.now()
            })
            
            # Keep only last 50 patterns per symbol
            if len(self.pattern_memory[symbol]) > 50:
                self.pattern_memory[symbol] = self.pattern_memory[symbol][-50:]

    def get_similar_pattern(self, symbol, current_features):
        """Find similar past patterns and predict outcome"""
        if symbol not in self.pattern_memory or len(self.pattern_memory[symbol]) < 5:
            return 0.5, 0.3
        
        # Simple similarity: compare recent price changes
        patterns = self.pattern_memory[symbol]
        
        # Count winning vs losing patterns
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
    """
    Main Agentic Loop system that coordinates all agents
    ✅ NEW: Sector Agent, Ensemble Weight Optimization, Performance Tracking
    """

    def __init__(self, xgb_model_dir='./csv/xgboost/'):
        self.agents = []
        self.vote_history = []
        self.decision_log = []
        self.performance_log = []  # ✅ NEW: Track ensemble performance

        # Initialize agents
        self.agents.append(XGBoostAgent(xgb_model_dir))
        self.agents.append(TechnicalAgent())
        self.agents.append(RiskAgent())
        self.agents.append(SectorAgent())  # ✅ NEW
        self.agents.append(NewsAgent())
        self.agents.append(MemoryAgent())

        # Adjustable weights
        self.agent_weights = self._get_initial_weights()
        
        # ✅ NEW: Track ensemble accuracy
        self.ensemble_correct = 0
        self.ensemble_total = 0
        
        # ✅ NEW: Load previous state if exists
        self._load_state()

        print("\n" + "="*60)
        print("🤖 AGENTIC LOOP INITIALIZED (with Sector Agent)")
        print("="*60)
        print(f"   Agents: {len(self.agents)}")
        for agent in self.agents:
            print(f"      - {agent.name} (weight: {agent.weight})")
        print("="*60)
    
    def _load_state(self):
        """✅ NEW: Load previous agent state"""
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
        """✅ NEW: Normalize weights to sum to 1.0"""
        total = sum(self.agent_weights)
        if total > 0:
            self.agent_weights = [w / total for w in self.agent_weights]

    def get_consensus(self, symbol, symbol_data, volatility, market_regime, 
                      sector='Unknown', atr=None, drawdown=0):
        """
        Get consensus decision from all agents
        ✅ NEW: Added sector, atr, drawdown parameters
        Returns: (decision, score, confidence, details)
        """
        votes = []
        agent_details = {}

        # Set current symbol for XGBoost agent
        for agent in self.agents:
            if agent.name == "XGBoost":
                agent.set_symbol(symbol)

        for i, agent in enumerate(self.agents):
            if agent.name == "XGBoost":
                prob, conf = agent.predict(self._get_features(symbol_data))

            elif agent.name == "Technical":
                prob, conf = agent.analyze(symbol_data)

            elif agent.name == "Risk":
                prob, conf = agent.assess(symbol, volatility, market_regime, atr, drawdown)

            elif agent.name == "Sector":
                prob, conf = agent.analyze(symbol, sector, symbol_data['close'].iloc[-1] if len(symbol_data) > 0 else None)

            elif agent.name == "News":
                prob, conf = agent.analyze_sentiment(symbol)

            elif agent.name == "Memory":
                prob, conf = agent.get_similar_pattern(symbol, self._get_features(symbol_data))

            else:
                prob, conf = 0.5, 0.5

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

        # ✅ NEW: Normalize weights before voting
        self._normalize_weights()
        
        # Recalculate with normalized weights
        for i, vote in enumerate(votes):
            vote['weight'] = self.agent_weights[i]

        # Calculate weighted consensus
        total_weight = sum(v['weight'] for v in votes)
        if total_weight > 0:
            weighted_score = sum(v['score'] * v['weight'] for v in votes) / total_weight
        else:
            weighted_score = 0.5

        # Calculate consensus confidence
        consensus_confidence = sum(v['confidence'] * v['weight'] for v in votes) / total_weight if total_weight > 0 else 0.5
        
        # ✅ NEW: Calculate vote agreement (how many agents agree)
        bullish_votes = sum(1 for v in votes if v['score'] > 0.55)
        bearish_votes = sum(1 for v in votes if v['score'] < 0.45)
        vote_agreement = max(bullish_votes, bearish_votes) / len(votes)

        # Determine decision
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

        # Log the decision
        log_entry = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'sector': sector,  # ✅ NEW
            'decision': decision,
            'score': weighted_score,
            'confidence': consensus_confidence,
            'vote_agreement': vote_agreement,  # ✅ NEW
            'agent_votes': agent_details
        }
        self.decision_log.append(log_entry)
        
        # ✅ NEW: Send Telegram for strong signals
        if decision in ['STRONG_BUY', 'STRONG_SELL'] and consensus_confidence > 0.65:
            self._send_signal_alert(log_entry)

        return decision, weighted_score, consensus_confidence, agent_details
    
    def _send_signal_alert(self, log_entry):
        """✅ NEW: Send Telegram alert for strong signals"""
        message = f"""
🚨 <b>Strong Signal Detected!</b>
📊 Symbol: {log_entry['symbol']}
🏭 Sector: {log_entry.get('sector', 'Unknown')}
🎯 Decision: {log_entry['decision']}
📈 Score: {log_entry['score']:.3f}
💪 Confidence: {log_entry['confidence']:.3f}
🤝 Vote Agreement: {log_entry['vote_agreement']:.0%}
"""
        send_telegram_message(message)

    def after_trade_feedback(self, trade_result):
        """
        Update agents based on trade outcome
        This is the LEARNING loop!
        ✅ NEW: Track ensemble performance, optimize weights
        """
        symbol = trade_result.get('symbol')
        pnl = trade_result.get('pnl', 0)
        was_win = pnl > 0

        # Find the decision that led to this trade
        recent_decisions = [d for d in self.decision_log if d['symbol'] == symbol]
        if not recent_decisions:
            return None

        last_decision = recent_decisions[-1]
        agent_votes = last_decision.get('agent_votes', {})
        
        # ✅ NEW: Determine if ensemble was correct
        ensemble_score = last_decision.get('score', 0.5)
        if was_win:
            ensemble_was_correct = (ensemble_score > 0.5)
        else:
            ensemble_was_correct = (ensemble_score <= 0.5)
        
        self.ensemble_total += 1
        if ensemble_was_correct:
            self.ensemble_correct += 1

        # Update each agent's performance
        for agent in self.agents:
            if agent.name in agent_votes:
                agent_score = agent_votes[agent.name]['score']

                # Determine if agent was correct
                if was_win:
                    was_correct = (agent_score > 0.5)
                else:
                    was_correct = (agent_score <= 0.5)

                confidence = agent_votes[agent.name]['confidence']
                agent.update_performance(was_correct, confidence)
                
                # ✅ NEW: Update Risk agent's symbol history
                if agent.name == "Risk":
                    agent.update_history(symbol, not was_win)

        # Update Memory agent
        memory_agent = next((a for a in self.agents if a.name == "Memory"), None)
        if memory_agent:
            # Add features to trade result for pattern memory
            if 'features' not in trade_result:
                trade_result['features'] = self._get_features_from_trade(trade_result)
            memory_agent.remember_trade(trade_result)
        
        # ✅ NEW: Update Sector agent data
        sector = trade_result.get('sector', 'Unknown')
        if sector != 'Unknown':
            sector_agent = next((a for a in self.agents if a.name == "Sector"), None)
            if sector_agent:
                # Update sector performance
                if sector not in sector_agent.sector_data:
                    sector_agent.sector_data[sector] = {'wins': 0, 'total': 0}
                sector_agent.sector_data[sector]['total'] += 1
                if was_win:
                    sector_agent.sector_data[sector]['wins'] += 1

        # ✅ NEW: Log ensemble performance
        ensemble_accuracy = self.ensemble_correct / self.ensemble_total if self.ensemble_total > 0 else 0.5
        self.performance_log.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'was_win': was_win,
            'pnl': pnl,
            'ensemble_correct': ensemble_was_correct,
            'ensemble_accuracy': ensemble_accuracy
        })

        # Log feedback
        print(f"\n   📊 Agent Feedback for {symbol}:")
        print(f"      Trade Result: {'WIN ✅' if was_win else 'LOSS ❌'} (PnL: {pnl:.2%})")
        print(f"      Ensemble Correct: {'✅' if ensemble_was_correct else '❌'}")
        print(f"      Ensemble Accuracy: {ensemble_accuracy:.1%}")
        print(f"      Updating {len(self.agents)} agents...")

        # Return updated weights
        return {a.name: a.get_dynamic_weight() for a in self.agents}

    def _get_features(self, symbol_data):
        """Extract features from symbol data for XGBoost"""
        if len(symbol_data) < 10:
            return np.zeros((1, 15))  # ✅ Increased features

        features = []

        # Price features
        close_price = symbol_data['close'].iloc[-1]
        features.append(close_price if not pd.isna(close_price) else 100)

        # Volume
        volume = symbol_data['volume'].iloc[-1] if 'volume' in symbol_data.columns else 0
        features.append(volume / 1e6 if not pd.isna(volume) else 0)

        # Returns (5d and 10d)
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

        # Volatility
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

        # Volume ratio
        if 'volume_ratio' in symbol_data.columns:
            vol_ratio = symbol_data['volume_ratio'].iloc[-1]
            features.append(vol_ratio if not pd.isna(vol_ratio) else 1)
        else:
            features.append(1)

        # RSI signals
        if 'rsi_oversold' in symbol_data.columns:
            features.append(symbol_data['rsi_oversold'].iloc[-1])
        else:
            features.append(0)
            
        if 'rsi_overbought' in symbol_data.columns:
            features.append(symbol_data['rsi_overbought'].iloc[-1])
        else:
            features.append(0)

        # Support/Resistance
        if 'dist_from_sr' in symbol_data.columns:
            features.append(symbol_data['dist_from_sr'].iloc[-1] / 100)
        else:
            features.append(0)
            
        if 'sr_strength' in symbol_data.columns:
            features.append(symbol_data['sr_strength'].iloc[-1] / 3)
        else:
            features.append(0)

        # Divergence
        if 'is_bullish_div' in symbol_data.columns:
            features.append(symbol_data['is_bullish_div'].iloc[-1])
        else:
            features.append(0)
            
        if 'div_strength' in symbol_data.columns:
            features.append(symbol_data['div_strength'].iloc[-1] / 3)
        else:
            features.append(0)

        # EMA
        if 'dist_from_ema' in symbol_data.columns:
            features.append(symbol_data['dist_from_ema'].iloc[-1] / 100)
        else:
            features.append(0)
            
        if 'above_ema' in symbol_data.columns:
            features.append(symbol_data['above_ema'].iloc[-1])
        else:
            features.append(0)

        # Fill remaining with zeros
        while len(features) < 15:
            features.append(0)

        return np.array(features[:15]).reshape(1, -1)
    
    def _get_features_from_trade(self, trade_result):
        """✅ NEW: Extract features from trade result for memory"""
        # Return placeholder if no features
        return trade_result.get('features', np.zeros(15))

    def get_summary(self):
        """Get performance summary of all agents"""
        summary = []
        for agent in self.agents:
            summary.append({
                'agent': agent.name,
                'accuracy': f"{agent.get_accuracy():.1%}",
                'recent_accuracy': f"{agent.recent_accuracy:.1%}",  # ✅ NEW
                'total_predictions': agent.total_predictions,
                'current_weight': f"{agent.get_dynamic_weight():.2f}"
            })
        return pd.DataFrame(summary)
    
    def get_ensemble_accuracy(self):
        """✅ NEW: Get ensemble accuracy"""
        if self.ensemble_total == 0:
            return 0.5
        return self.ensemble_correct / self.ensemble_total

    def save_decision_log(self, path='./csv/agentic_loop_log.csv'):
        """Save decision log to CSV"""
        if self.decision_log:
            try:
                df = pd.DataFrame(self.decision_log)
                if 'agent_votes' in df.columns:
                    df['agent_votes'] = df['agent_votes'].astype(str)
                df.to_csv(path, index=False)
                print(f"   ✅ Agentic Loop log saved: {path}")
            except Exception as e:
                print(f"   ⚠️ Could not save decision log: {e}")
    
    def save_state(self, path='./csv/agentic_loop_state.json'):
        """✅ NEW: Save agent state for persistence"""
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
            print(f"   💾 Agentic Loop state saved: {path}")
        except Exception as e:
            print(f"   ⚠️ Could not save state: {e}")


# =========================================================
# INTEGRATION WITH YOUR EXISTING SYSTEM
# =========================================================

def integrate_with_ppo(agentic_loop, trade_result):
    """
    Integrate Agentic Loop feedback with PPO training
    This can be called from your ppo_train.py
    ✅ NEW: Returns more detailed feedback
    """
    # Get feedback from agents
    feedback = agentic_loop.after_trade_feedback(trade_result)

    # Adjust PPO reward based on agent consensus
    if feedback:
        avg_agent_accuracy = np.mean([a.get_accuracy() for a in agentic_loop.agents])
        ensemble_accuracy = agentic_loop.get_ensemble_accuracy()
        
        # Combine agent and ensemble accuracy
        combined_accuracy = (avg_agent_accuracy + ensemble_accuracy) / 2

        if combined_accuracy > 0.6:
            print(f"   🚀 Agent consensus strong! Boosting PPO reward")
            return 1.2, {'agent_accuracy': avg_agent_accuracy, 'ensemble_accuracy': ensemble_accuracy}
        elif combined_accuracy < 0.4:
            print(f"   ⚠️ Agent consensus weak! Reducing PPO reward")
            return 0.8, {'agent_accuracy': avg_agent_accuracy, 'ensemble_accuracy': ensemble_accuracy}

    return 1.0, {}


# =========================================================
# QUICK TEST
# =========================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTING AGENTIC LOOP (with Sector Agent)")
    print("="*60)

    # Initialize
    loop = AgenticLoop()

    # Test with sample data
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
        'sr_strength': [2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1]
    })

    # Get consensus with sector
    decision, score, confidence, details = loop.get_consensus(
        symbol="TEST",
        symbol_data=sample_data,
        volatility=0.02,
        market_regime="BULL",
        sector="Technology",
        atr=2.5,
        drawdown=0.03
    )

    print(f"\n   📊 Consensus Decision: {decision}")
    print(f"   Score: {score:.3f}")
    print(f"   Confidence: {confidence:.3f}")

    print("\n   🤖 Agent Votes:")
    for agent, info in details.items():
        print(f"      {agent}: score={info['score']}, conf={info['confidence']}, weight={info['weight']}")

    # Test feedback
    print("\n   📝 Testing feedback loop...")
    loop.after_trade_feedback({
        'symbol': 'TEST',
        'pnl': 0.05,
        'success': True,
        'sector': 'Technology',
        'features': np.random.randn(15)
    })

    # Show summary
    print("\n   📊 Agent Performance Summary:")
    print(loop.get_summary().to_string())
    
    print(f"\n   📈 Ensemble Accuracy: {loop.get_ensemble_accuracy():.1%}")

    print("\n" + "="*60)
    print("✅ AGENTIC LOOP TEST COMPLETE!")
    print("="*60)