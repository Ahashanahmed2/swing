# agentic_loop.py - Multi-Agent Voting System for Trading
# This integrates with your existing XGBoost + PPO system

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class TradingAgent:
    """Base class for all trading agents"""
    def __init__(self, name, weight=1.0):
        self.name = name
        self.weight = weight
        self.performance_history = []
        self.correct_predictions = 0
        self.total_predictions = 0
        
    def update_performance(self, was_correct, confidence):
        self.total_predictions += 1
        if was_correct:
            self.correct_predictions += 1
        self.performance_history.append({
            'timestamp': datetime.now(),
            'correct': was_correct,
            'confidence': confidence
        })
        
    def get_accuracy(self):
        if self.total_predictions == 0:
            return 0.5
        return self.correct_predictions / self.total_predictions
    
    def get_dynamic_weight(self):
        """Dynamic weight based on recent performance"""
        base_weight = self.weight
        accuracy = self.get_accuracy()
        
        # Boost weight if accurate, reduce if not
        if accuracy > 0.6:
            return base_weight * (1 + (accuracy - 0.6))
        elif accuracy < 0.4:
            return base_weight * (0.5 + accuracy)
        return base_weight


class XGBoostAgent(TradingAgent):
    """Your existing XGBoost model as an agent"""
    def __init__(self, xgb_model_path):
        super().__init__("XGBoost", weight=0.4)
        self.model_path = xgb_model_path
        self.load_model()
        
    def load_model(self):
        try:
            import joblib
            self.model = joblib.load(self.model_path)
            print(f"   ✅ XGBoost Agent loaded: {self.model_path}")
        except:
            self.model = None
            print(f"   ⚠️ XGBoost Agent not loaded")
    
    def predict(self, features):
        if self.model is None:
            return 0.5, 0.5
        try:
            prob = self.model.predict_proba(features)[0, 1]
            return prob, prob
        except:
            return 0.5, 0.5


class TechnicalAgent(TradingAgent):
    """Technical analysis agent (RSI, MACD, Support/Resistance)"""
    def __init__(self):
        super().__init__("Technical", weight=0.2)
        
    def analyze(self, symbol_data):
        """Analyze technical indicators"""
        if len(symbol_data) < 20:
            return 0.5, 0.3
        
        signals = []
        confidences = []
        
        # RSI Analysis
        if 'rsi' in symbol_data.columns:
            rsi = symbol_data['rsi'].iloc[-1]
            if rsi < 30:
                signals.append(1)  # Oversold → Buy signal
                confidences.append(min(0.8, (30 - rsi) / 30))
            elif rsi > 70:
                signals.append(0)  # Overbought → Sell signal
                confidences.append(min(0.8, (rsi - 70) / 30))
        
        # MACD Analysis
        if 'macd' in symbol_data.columns and 'macd_signal' in symbol_data.columns:
            macd = symbol_data['macd'].iloc[-1]
            signal = symbol_data['macd_signal'].iloc[-1]
            if macd > signal:
                signals.append(1)
                confidences.append(0.6)
            elif macd < signal:
                signals.append(0)
                confidences.append(0.6)
        
        # Moving Average Analysis
        if 'sma_20' in symbol_data.columns and 'sma_50' in symbol_data.columns:
            sma_20 = symbol_data['sma_20'].iloc[-1]
            sma_50 = symbol_data['sma_50'].iloc[-1]
            close = symbol_data['close'].iloc[-1]
            
            if close > sma_20 > sma_50:
                signals.append(1)
                confidences.append(0.7)
            elif close < sma_20 < sma_50:
                signals.append(0)
                confidences.append(0.7)
        
        if not signals:
            return 0.5, 0.3
        
        # Weighted average of signals
        weighted_score = sum(s * c for s, c in zip(signals, confidences)) / sum(confidences)
        avg_confidence = sum(confidences) / len(confidences)
        
        return weighted_score, avg_confidence


class RiskAgent(TradingAgent):
    """Risk management agent - position sizing and stop-loss"""
    def __init__(self):
        super().__init__("Risk", weight=0.2)
        
    def assess(self, symbol, volatility, market_regime):
        """Assess risk level"""
        risk_score = 0.5
        
        # Adjust based on volatility
        if volatility > 0.03:
            risk_score -= 0.2
        elif volatility < 0.01:
            risk_score += 0.1
            
        # Adjust based on market regime
        if market_regime == 'BEAR':
            risk_score -= 0.3
        elif market_regime == 'BULL':
            risk_score += 0.2
            
        # Adjust based on symbol history
        if hasattr(self, 'symbol_history'):
            recent_losses = self.symbol_history.get(symbol, {}).get('recent_losses', 0)
            if recent_losses > 3:
                risk_score -= 0.2
                
        return max(0.1, min(0.9, risk_score)), 0.8


class NewsAgent(TradingAgent):
    """Sentiment analysis from news (placeholder for now)"""
    def __init__(self):
        super().__init__("News", weight=0.2)
        
    def analyze_sentiment(self, symbol):
        """Analyze news sentiment for symbol"""
        # TODO: Integrate with actual news API
        # For now, return neutral
        return 0.5, 0.5


class MemoryAgent(TradingAgent):
    """Memory agent - learns from past mistakes"""
    def __init__(self):
        super().__init__("Memory", weight=0.3)
        self.mistake_memory = []
        self.success_memory = []
        
    def remember_trade(self, trade_result):
        if trade_result.get('pnl', 0) < 0:
            self.mistake_memory.append(trade_result)
        else:
            self.success_memory.append(trade_result)
            
    def get_similar_pattern(self, current_pattern):
        """Find similar past patterns"""
        # Simple pattern matching
        if len(self.success_memory) > 0:
            return 0.6, 0.5
        return 0.5, 0.3


class AgenticLoop:
    """
    Main Agentic Loop system that coordinates all agents
    """
    
    def __init__(self, xgb_model_dir='./csv/xgboost/'):
        self.agents = []
        self.vote_history = []
        self.decision_log = []
        
        # Initialize agents
        self.agents.append(XGBoostAgent(xgb_model_dir))
        self.agents.append(TechnicalAgent())
        self.agents.append(RiskAgent())
        self.agents.append(NewsAgent())
        self.agents.append(MemoryAgent())
        
        # Adjustable weights
        self.agent_weights = self._get_initial_weights()
        
        print("\n" + "="*60)
        print("🤖 AGENTIC LOOP INITIALIZED")
        print("="*60)
        print(f"   Agents: {len(self.agents)}")
        for agent in self.agents:
            print(f"      - {agent.name} (weight: {agent.weight})")
        print("="*60)
    
    def _get_initial_weights(self):
        return [agent.weight for agent in self.agents]
    
    def get_consensus(self, symbol, symbol_data, volatility, market_regime):
        """
        Get consensus decision from all agents
        Returns: (decision, confidence, details)
        """
        votes = []
        agent_details = {}
        
        for agent in self.agents:
            if agent.name == "XGBoost":
                # Get features for XGBoost
                prob, conf = agent.predict(self._get_features(symbol_data))
                
            elif agent.name == "Technical":
                prob, conf = agent.analyze(symbol_data)
                
            elif agent.name == "Risk":
                prob, conf = agent.assess(symbol, volatility, market_regime)
                
            elif agent.name == "News":
                prob, conf = agent.analyze_sentiment(symbol)
                
            elif agent.name == "Memory":
                prob, conf = agent.get_similar_pattern(symbol_data.iloc[-10:])
                
            else:
                prob, conf = 0.5, 0.5
            
            dynamic_weight = agent.get_dynamic_weight()
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
        
        # Calculate weighted consensus
        total_weight = sum(v['weight'] for v in votes)
        if total_weight > 0:
            weighted_score = sum(v['score'] * v['weight'] for v in votes) / total_weight
        else:
            weighted_score = 0.5
        
        # Calculate consensus confidence
        consensus_confidence = sum(v['confidence'] * v['weight'] for v in votes) / total_weight if total_weight > 0 else 0.5
        
        # Determine decision
        if weighted_score >= 0.65:
            decision = 'STRONG_BUY'
        elif weighted_score >= 0.55:
            decision = 'BUY'
        elif weighted_score <= 0.35:
            decision = 'STRONG_SELL'
        elif weighted_score <= 0.45:
            decision = 'SELL'
        else:
            decision = 'HOLD'
        
        # Log the decision
        self.decision_log.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'decision': decision,
            'score': weighted_score,
            'confidence': consensus_confidence,
            'agent_votes': agent_details
        })
        
        return decision, weighted_score, consensus_confidence, agent_details
    
    def after_trade_feedback(self, trade_result):
        """
        Update agents based on trade outcome
        This is the LEARNING loop!
        """
        symbol = trade_result.get('symbol')
        pnl = trade_result.get('pnl', 0)
        was_win = pnl > 0
        
        # Find the decision that led to this trade
        recent_decisions = [d for d in self.decision_log if d['symbol'] == symbol]
        if not recent_decisions:
            return
        
        last_decision = recent_decisions[-1]
        agent_votes = last_decision.get('agent_votes', {})
        
        # Update each agent's performance
        for agent in self.agents:
            if agent.name in agent_votes:
                agent_score = agent_votes[agent.name]['score']
                
                # Determine if agent was correct
                if was_win:
                    # Winning trade: agents with BUY score > 0.5 were correct
                    was_correct = (agent_score > 0.5)
                else:
                    # Losing trade: agents with BUY score > 0.5 were wrong
                    was_correct = (agent_score <= 0.5)
                
                confidence = agent_votes[agent.name]['confidence']
                agent.update_performance(was_correct, confidence)
        
        # Update Memory agent
        memory_agent = next((a for a in self.agents if a.name == "Memory"), None)
        if memory_agent:
            memory_agent.remember_trade(trade_result)
        
        # Log feedback
        print(f"\n   📊 Agent Feedback for {symbol}:")
        print(f"      Trade Result: {'WIN ✅' if was_win else 'LOSS ❌'} (PnL: {pnl:.2%})")
        print(f"      Updating {len(self.agents)} agents...")
        
        # Return updated weights
        return {a.name: a.get_dynamic_weight() for a in self.agents}
    
    def _get_features(self, symbol_data):
        """Extract features from symbol data for XGBoost"""
        if len(symbol_data) < 10:
            return np.zeros((1, 10))
        
        features = []
        
        # Price features
        features.append(symbol_data['close'].iloc[-1])
        features.append(symbol_data['volume'].iloc[-1] if 'volume' in symbol_data.columns else 0)
        
        # Returns
        if len(symbol_data) >= 5:
            features.append((symbol_data['close'].iloc[-1] - symbol_data['close'].iloc[-5]) / symbol_data['close'].iloc[-5])
        else:
            features.append(0)
        
        # Volatility
        if 'volatility' in symbol_data.columns:
            features.append(symbol_data['volatility'].iloc[-1])
        else:
            features.append(0.02)
        
        # Fill remaining with zeros
        while len(features) < 10:
            features.append(0)
        
        return np.array(features).reshape(1, -1)
    
    def get_summary(self):
        """Get performance summary of all agents"""
        summary = []
        for agent in self.agents:
            summary.append({
                'agent': agent.name,
                'accuracy': f"{agent.get_accuracy():.1%}",
                'total_predictions': agent.total_predictions,
                'current_weight': f"{agent.get_dynamic_weight():.2f}"
            })
        return pd.DataFrame(summary)
    
    def save_decision_log(self, path='./csv/agentic_loop_log.csv'):
        """Save decision log to CSV"""
        if self.decision_log:
            df = pd.DataFrame(self.decision_log)
            df.to_csv(path, index=False)
            print(f"   ✅ Agentic Loop log saved: {path}")


# =========================================================
# INTEGRATION WITH YOUR EXISTING SYSTEM
# =========================================================

def integrate_with_ppo(agentic_loop, trade_result):
    """
    Integrate Agentic Loop feedback with PPO training
    This can be called from your ppo_train.py
    """
    # Get feedback from agents
    feedback = agentic_loop.after_trade_feedback(trade_result)
    
    # Adjust PPO reward based on agent consensus
    if feedback:
        # Boost reward if agents were confident and correct
        avg_agent_accuracy = np.mean([a.get_accuracy() for a in agentic_loop.agents])
        
        if avg_agent_accuracy > 0.6:
            print(f"   🚀 Agent consensus strong! Boosting PPO reward")
            return 1.2  # Reward multiplier
        elif avg_agent_accuracy < 0.4:
            print(f"   ⚠️ Agent consensus weak! Reducing PPO reward")
            return 0.8  # Reward penalty
    
    return 1.0


# =========================================================
# QUICK TEST
# =========================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTING AGENTIC LOOP")
    print("="*60)
    
    # Initialize
    loop = AgenticLoop()
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104],
        'volume': [1000, 1100, 1200, 1300, 1400],
        'rsi': [25, 28, 30, 32, 35],
        'volatility': [0.02, 0.02, 0.015, 0.015, 0.01]
    })
    
    # Get consensus
    decision, score, confidence, details = loop.get_consensus(
        symbol="TEST",
        symbol_data=sample_data,
        volatility=0.02,
        market_regime="BULL"
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
        'success': True
    })
    
    # Show summary
    print("\n   📊 Agent Performance Summary:")
    print(loop.get_summary().to_string())
    
    print("\n" + "="*60)
    print("✅ AGENTIC LOOP TEST COMPLETE!")
    print("="*60)
