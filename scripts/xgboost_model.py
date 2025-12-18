import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
from collections import Counter
import warnings
import os
warnings.filterwarnings('ignore')

class XGBoostTradingModel:
def init(self, n_estimators=1000, max_depth=5, learning_rate=0.01):
self.model = None
self.regression_model = None
self.scaler = StandardScaler()
self.feature_importance = None
self.params = {
'n_estimators': n_estimators,
'max_depth': max_depth,
'learning_rate': learning_rate,
'subsample': 0.8,
'colsample_bytree': 0.8,
'random_state': 42,
'early_stopping_rounds': 50,
'eval_metric': 'logloss',
'objective': 'binary:logistic'
}

def prepare_data_with_technical_indicators(self, market_data, trade_data):
    """
    সমস্ত টেকনিক্যাল ইন্ডিকেটরস সহ SL/TP reward system
    """
    symbol = market_data['symbol'].iloc[0] if len(market_data) > 0 else 'UNKNOWN'
    print(f"   📊 {symbol} - টেকনিক্যাল ইন্ডিকেটরস সহ ডাটা প্রিপেয়ার করা হচ্ছে...")
    
    # 1. মার্কেট ডাটা কপি এবং প্রিপ্রসেস
    market_data = market_data.copy()
    market_data = market_data.sort_values('date')
    market_data['date'] = pd.to_datetime(market_data['date'])
    
    # 2. ট্রেড সিগন্যাল খুঁজুন (SL এবং TP সহ)
    buy_signals = []
    reward_labels = []
    
    # প্রতিটি buy সিগন্যালের জন্য
    for _, trade_row in trade_data.iterrows():
        buy_date = pd.to_datetime(trade_row['date'])
        buy_price = trade_row['buy']
        
        # SL এবং TP ভ্যালু নিন
        sl_price = trade_row.get('SL', buy_price * 0.95)
        tp_price = trade_row.get('tp', buy_price * 1.10)
        
        # SL এবং TP validation
        if sl_price <= 0:
            sl_price = buy_price * 0.95
        if tp_price <= buy_price:
            tp_price = buy_price * 1.10
        
        # buy date পরের 10 দিন চেক করুন
        for days_ahead in range(1, 11):
            target_date = buy_date + pd.Timedelta(days=days_ahead)
            
            # target_date-এর মার্কেট ডাটা খুঁজুন
            market_row = market_data[market_data['date'] == target_date]
            
            if len(market_row) > 0:
                current_data = market_row.iloc[0]
                
                close_price = current_data['close']
                high_price = current_data['high']
                low_price = current_data['low']
                
                # 3. REWARD ক্যালকুলেশন লজিক
                reward = 0.0
                
                # লজিক 1: SL হিট চেক (low <= SL)
                sl_hit = low_price <= sl_price
                
                # লজিক 2: TP হিট চেক (high >= TP)
                tp_hit = high_price >= tp_price
                
                # লজিক 3: Profit/Loss বেসড রিওয়ার্ড
                current_profit_loss = (close_price - buy_price) / buy_price
                
                # রিওয়ার্ড ক্যালকুলেশন
                if sl_hit:
                    # SL হিট = নেগেটিভ রিওয়ার্ড
                    sl_severity = 1.0 - (days_ahead / 20.0)
                    reward = -1.0 * sl_severity
                    
                    # যদি ATR থাকে, ATR-বেসড adjustment
                    if 'atr' in current_data and pd.notna(current_data['atr']):
                        atr_multiplier = current_data['atr'] / buy_price
                        reward = reward * (1.0 + atr_multiplier * 2)
                
                elif tp_hit:
                    # TP হিট = পজিটিভ রিওয়ার্ড
                    tp_efficiency = 0.5 + (days_ahead / 20.0)
                    reward = 1.0 * tp_efficiency
                    
                    # যদি RSI থাকে, momentum check
                    if 'rsi' in current_data and pd.notna(current_data['rsi']):
                        if 30 <= current_data['rsi'] <= 70:
                            reward = reward * 1.1
                
                else:
                    # No hit = profit/loss based
                    if current_profit_loss > 0:
                        reward = 0.2 * current_profit_loss
                    else:
                        reward = 0.5 * current_profit_loss
                
                # 4. TECHNICAL INDICATOR ADJUSTMENTS
                
                # BB adjustment
                if all(ind in current_data for ind in ['bb_upper', 'bb_lower', 'close']):
                    if pd.notna(current_data['bb_upper']) and pd.notna(current_data['bb_lower']):
                        bb_position = (current_data['close'] - current_data['bb_lower']) / \
                                    (current_data['bb_upper'] - current_data['bb_lower'])
                        if bb_position < 0.2:
                            reward = reward * 1.15
                        elif bb_position > 0.8:
                            reward = reward * 0.85
                
                # MACD adjustment
                if 'macd_hist' in current_data and pd.notna(current_data['macd_hist']):
                    if current_data['macd_hist'] > 0:
                        reward = reward * 1.08
                    else:
                        reward = reward * 0.92
                
                # Candlestick patterns bonus
                pattern_bonus = 1.0
                bullish_patterns = ['Hammer', 'BullishEngulfing', 'MorningStar', 
                                  'PiercingLine', 'ThreeWhiteSoldiers']
                
                for pattern in bullish_patterns:
                    if pattern in current_data and current_data[pattern]:
                        pattern_bonus += 0.05
                
                bearish_patterns = ['Doji']
                for pattern in bearish_patterns:
                    if pattern in current_data and current_data[pattern]:
                        pattern_bonus -= 0.03
                
                reward = reward * pattern_bonus
                
                # VOLUME CONFIRMATION
                if 'volume' in current_data and pd.notna(current_data['volume']):
                    volume_avg = market_data['volume'].rolling(10).mean().iloc[-1]
                    volume_ratio = current_data['volume'] / volume_avg if volume_avg > 0 else 1.0
                    
                    if volume_ratio > 1.5:
                        reward = reward * 1.1
                
                # 5. FEATURE সংগ্রহ
                buy_date_features = market_data[market_data['date'] == buy_date]
                
                if len(buy_date_features) > 0:
                    features = buy_date_features.iloc[0].to_dict()
                    
                    # বেসিক ফিচারস
                    features['target_date'] = target_date
                    features['days_ahead'] = days_ahead
                    features['buy_price'] = buy_price
                    features['sl_price'] = sl_price
                    features['tp_price'] = tp_price
                    features['current_close'] = close_price
                    features['current_high'] = high_price
                    features['current_low'] = low_price
                    
                    # SL/TP ডিস্ট্যান্স
                    features['sl_distance_pct'] = (buy_price - sl_price) / buy_price
                    features['tp_distance_pct'] = (tp_price - buy_price) / buy_price
                    features['risk_reward_ratio'] = features['tp_distance_pct'] / features['sl_distance_pct']
                    
                    # কারেন্ট স্টেট
                    features['current_profit_loss_pct'] = current_profit_loss
                    features['sl_hit'] = int(sl_hit)
                    features['tp_hit'] = int(tp_hit)
                    
                    # টেকনিক্যাল স্টেট
                    features['close_vs_bb_upper'] = (close_price - features.get('bb_upper', close_price)) / close_price
                    features['close_vs_bb_lower'] = (close_price - features.get('bb_lower', close_price)) / close_price
                    features['macd_cross'] = 1 if features.get('macd', 0) > features.get('macd_signal', 0) else 0
                    
                    # RSI স্টেট
                    rsi = features.get('rsi', 50)
                    features['rsi_oversold'] = 1 if rsi < 30 else 0
                    features['rsi_overbought'] = 1 if rsi > 70 else 0
                    features['rsi_neutral'] = 1 if 30 <= rsi <= 70 else 0
                    
                    # Candlestick patterns স্টেট
                    for pattern in ['Hammer', 'BullishEngulfing', 'MorningStar', 'Doji']:
                        if pattern in features:
                            features[f'{pattern}_present'] = int(features[pattern])
                    
                    features['reward'] = reward
                    features['symbol'] = symbol
                    
                    buy_signals.append(features)
                    reward_labels.append(reward)
    
    if len(buy_signals) == 0:
        print(f"   ❌ কোন valid buy সিগন্যাল নেই")
        return pd.DataFrame(), []
    
    # 6. DATAFRAME তৈরি
    data_df = pd.DataFrame(buy_signals)
    
    # 7. টার্গেট ভ্যারিয়েবল
    data_df['signal_binary'] = (data_df['reward'] > 0).astype(int)
    data_df['reward_regression'] = data_df['reward']
    
    # 8. ফিচার সিলেকশন
    base_features = [
        # প্রাইস ফিচারস
        'open', 'high', 'low', 'close', 'volume', 'value', 'trades', 'change',
        
        # SL/TP ফিচারস
        'buy_price', 'sl_price', 'tp_price', 'days_ahead',
        'sl_distance_pct', 'tp_distance_pct', 'risk_reward_ratio',
        
        # টেকনিক্যাল ইন্ডিকেটরস
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower',
        'atr', 'zigzag',
        
        # RSI স্টেট
        'rsi_oversold', 'rsi_overbought', 'rsi_neutral',
        
        # MACD স্টেট
        'macd_cross',
        
        # BB পজিশন
        'close_vs_bb_upper', 'close_vs_bb_lower',
        
        # ক্যান্ডলেস্টিক প্যাটার্নস
        'Hammer_present', 'BullishEngulfing_present', 
        'MorningStar_present', 'Doji_present',
        'PiercingLine', 'ThreeWhiteSoldiers'
    ]
    
    available_features = []
    for f in base_features:
        if f in data_df.columns:
            nan_pct = data_df[f].isna().sum() / len(data_df)
            if nan_pct < 0.3:
                available_features.append(f)
    
    # 9. NaN হ্যান্ডলিং
    if len(available_features) > 0:
        important_features = ['open', 'high', 'low', 'close', 'volume', 'buy_price']
        important_features = [f for f in important_features if f in available_features]
        
        if important_features:
            valid_mask = data_df[important_features].notna().all(axis=1)
            data_df = data_df[valid_mask].copy()
    
    # 10. NaN ফিল
    for col in available_features:
        if data_df[col].isna().any():
            if data_df[col].dtype in ['float64', 'int64']:
                data_df[col] = data_df[col].fillna(data_df[col].median())
            else:
                data_df[col] = data_df[col].fillna(0)
    
    print(f"   📊 Total samples: {len(data_df)}")
    print(f"   🎯 SL hits: {data_df['sl_hit'].sum()}")
    print(f"   🎯 TP hits: {data_df['tp_hit'].sum()}")
    print(f"   📈 Good trades (reward>0): {data_df['signal_binary'].sum()}")
    print(f"   📉 Bad trades (reward<=0): {len(data_df) - data_df['signal_binary'].sum()}")
    print(f"   🔧 Features available: {len(available_features)}")
    print(f"   📊 Reward stats - Min: {data_df['reward'].min():.3f}, "
          f"Max: {data_df['reward'].max():.3f}, Mean: {data_df['reward'].mean():.3f}")
    
    return data_df, available_features

def train_with_technical_indicators(self, market_data, trade_data):
    """
    টেকনিক্যাল ইন্ডিকেটরস সহ REGRESSION মডেল ট্রেন করে
    """
    symbol = market_data['symbol'].iloc[0] if len(market_data) > 0 else 'UNKNOWN'
    print(f"   🤖 {symbol} - টেকনিক্যাল ইন্ডিকেটরস মডেল ট্রেনিং...")
    
    # 1. টেকনিক্যাল ইন্ডিকেটরস সহ ডাটা প্রিপেয়ার
    data, features = self.prepare_data_with_technical_indicators(market_data, trade_data)
    
    if len(data) < 15:
        print(f"   ⚠️ পর্যাপ্ত ডাটা নেই: {len(data)} samples (min 15)")
        return 0.0, 0.0
    
    if len(features) < 10:
        print(f"   ⚠️ পর্যাপ্ত ফিচার নেই: {len(features)} (min 10)")
        return 0.0, 0.0
    
    # 2. ফিচার এবং টার্গেট আলাদা করা
    X = data[features]
    y_regression = data['reward_regression']
    y_binary = data['signal_binary']
    
    # 3. ডাটা স্ট্যাটস
    print(f"   📊 Final data shape: {X.shape}")
    
    # 4. ট্রেন-টেস্ট স্প্লিট
    X_train, X_test, y_train_reg, y_test_reg, y_train_bin, y_test_bin = train_test_split(
        X, y_regression, y_binary,
        test_size=0.3,
        random_state=42,
        stratify=y_binary if len(np.unique(y_binary)) > 1 else None
    )
    
    print(f"   🏋️ Training samples: {X_train.shape[0]}")
    print(f"   🧪 Testing samples: {X_test.shape[0]}")
    
    # 5. REGRESSION মডেল তৈরি (XGBoost)
    print("   🚀 টেকনিক্যাল মডেল ট্রেনিং শুরু...")
    
    try:
        # ADVANCED XGBoost REGRESSION মডেল
        self.regression_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            objective='reg:squarederror',
            eval_metric='rmse',
            verbosity=0
        )
        
        # 6. মডেল ট্রেনিং
        self.regression_model.fit(
            X_train,
            y_train_reg,
            eval_set=[(X_test, y_test_reg)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # 7. পারফরম্যান্স ইভ্যালুয়েশন
        y_pred_reg = self.regression_model.predict(X_test)
        
        # Regression metrics
        mse = mean_squared_error(y_test_reg, y_pred_reg)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_reg, y_pred_reg)
        r2 = r2_score(y_test_reg, y_pred_reg)
        
        # Binary classification metrics
        y_pred_binary = (y_pred_reg > 0).astype(int)
        binary_accuracy = accuracy_score(y_test_bin, y_pred_binary)
        f1 = f1_score(y_test_bin, y_pred_binary, zero_division=0)
        
        print(f"   ✅ ট্রেনিং সম্পূর্ণ!")
        print(f"   📊 REGRESSION METRICS:")
        print(f"     RMSE: {rmse:.4f}")
        print(f"     MAE: {mae:.4f}")
        print(f"     R² Score: {r2:.4f}")
        print(f"   📊 BINARY CLASSIFICATION:")
        print(f"     Accuracy: {binary_accuracy:.4f}")
        print(f"     F1 Score: {f1:.4f}")
        print(f"   📊 PREDICTION DISTRIBUTION:")
        print(f"     Positive predictions: {(y_pred_reg > 0).sum()}/{len(y_pred_reg)} ({(y_pred_reg > 0).mean():.1%})")
        print(f"     Mean predicted reward: {y_pred_reg.mean():.3f}")
        
        # 8. ফিচার ইম্পরটেন্স
        if hasattr(self.regression_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': features,
                'importance': self.regression_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"   🏆 TOP 10 IMPORTANT FEATURES:")
            for i, row in self.feature_importance.head(10).iterrows():
                importance_stars = "★" * int(row['importance'] * 50)
                print(f"     {row['feature']:25s} {row['importance']:.4f} {importance_stars}")
        
        # 9. Technical analysis report
        print(f"   📈 TECHNICAL ANALYSIS SUMMARY:")
        
        # RSI effectiveness
        if 'rsi' in features:
            rsi_corr = data['rsi'].corr(data['reward'])
            print(f"     RSI-Reward Correlation: {rsi_corr:.3f}")
        
        # MACD effectiveness
        if 'macd' in features and 'macd_signal' in features:
            macd_bullish = (data['macd'] > data['macd_signal']).mean()
            macd_reward_when_bullish = data[data['macd'] > data['macd_signal']]['reward'].mean()
            print(f"     MACD Bullish %: {macd_bullish:.1%}, Avg Reward: {macd_reward_when_bullish:.3f}")
        
        # BB effectiveness
        if all(f in features for f in ['bb_upper', 'bb_lower']):
            bb_width = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            bb_width_corr = bb_width.corr(data['reward'])
            print(f"     BB Width-Reward Correlation: {bb_width_corr:.3f}")
        
        # Candlestick patterns effectiveness
        bullish_patterns = ['Hammer_present', 'BullishEngulfing_present', 'MorningStar_present']
        for pattern in bullish_patterns:
            if pattern in features:
                pattern_rate = data[pattern].mean()
                pattern_reward = data[data[pattern] == 1]['reward'].mean()
                if pattern_rate > 0:
                    print(f"     {pattern:25s} Rate: {pattern_rate:.1%}, Avg Reward: {pattern_reward:.3f}")
        
        return max(0, r2), binary_accuracy
        
    except Exception as e:
        print(f"   ❌ ট্রেনিং এরর: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0

def predict_with_technical_analysis(self, market_data, trade_data, days_ahead=5):
    """
    টেকনিক্যাল ইন্ডিকেটরস সহ প্রেডিকশন করে
    """
    if not hasattr(self, 'regression_model') or self.regression_model is None:
        raise ValueError("টেকনিক্যাল মডেল ট্রেন করা হয়নি")
    
    symbol = market_data['symbol'].iloc[0] if len(market_data) > 0 else 'UNKNOWN'
    print(f"   🔮 {symbol} - টেকনিক্যাল অ্যানালাইসিস প্রেডিকশন...")
    
    # 1. মার্কেট ডাটা প্রিপেয়ার
    market_data = market_data.copy()
    market_data = market_data.sort_values('date')
    market_data['date'] = pd.to_datetime(market_data['date'])
    
    # 2. সবচেয়ে recent buy সিগন্যাল নিন
    if len(trade_data) == 0:
        print(f"   ❌ {symbol}: কোন ট্রেড সিগন্যাল নেই")
        return pd.DataFrame(), pd.DataFrame()
    
    recent_buy = trade_data.sort_values('date').iloc[-1]
    buy_date = pd.to_datetime(recent_buy['date'])
    buy_price = recent_buy['buy']
    sl_price = recent_buy.get('SL', buy_price * 0.95)
    tp_price = recent_buy.get('tp', buy_price * 1.10)
    
    # 3. buy date-এর টেকনিক্যাল ডাটা নিন
    buy_date_data = market_data[market_data['date'] == buy_date]
    
    if len(buy_date_data) == 0:
        print(f"   ❌ {symbol}: buy date-এর ডাটা নেই")
        return pd.DataFrame(), pd.DataFrame()
    
    buy_tech_data = buy_date_data.iloc[0]
    
    # 4. ভবিষ্যতের জন্য প্রেডিকশন তৈরি
    predictions = []
    
    for days_ahead_val in range(1, days_ahead + 1):
        # ফিচার ভেক্টর তৈরি
        features_dict = {}
        
        # বেসিক ফিচারস
        features_dict['days_ahead'] = days_ahead_val
        features_dict['buy_price'] = buy_price
        features_dict['sl_price'] = sl_price
        features_dict['tp_price'] = tp_price
        features_dict['sl_distance_pct'] = (buy_price - sl_price) / buy_price
        features_dict['tp_distance_pct'] = (tp_price - buy_price) / buy_price
        features_dict['risk_reward_ratio'] = features_dict['tp_distance_pct'] / max(features_dict['sl_distance_pct'], 0.001)
        
        # টেকনিক্যাল ফিচারস (buy date-এর)
        tech_features = [
            'open', 'high', 'low', 'close', 'volume', 'value', 'trades', 'change',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'zigzag',
            'Hammer', 'BullishEngulfing', 'MorningStar', 'Doji',
            'PiercingLine', 'ThreeWhiteSoldiers'
        ]
        
        for feature in tech_features:
            if feature in buy_tech_data:
                features_dict[feature] = buy_tech_data[feature]
        
        # derived features
        if all(f in features_dict for f in ['bb_upper', 'bb_lower', 'close']):
            if features_dict['bb_upper'] != features_dict['bb_lower']:
                features_dict['close_vs_bb_upper'] = (
                    features_dict['close'] - features_dict['bb_upper']
                ) / features_dict['close']
                features_dict['close_vs_bb_lower'] = (
                    features_dict['close'] - features_dict['bb_lower']
                ) / features_dict['close']
        
        features_dict['macd_cross'] = 1 if features_dict.get('macd', 0) > features_dict.get('macd_signal', 0) else 0
        
        # RSI স্টেট
        rsi = features_dict.get('rsi', 50)
        features_dict['rsi_oversold'] = 1 if rsi < 30 else 0
        features_dict['rsi_overbought'] = 1 if rsi > 70 else 0
        features_dict['rsi_neutral'] = 1 if 30 <= rsi <= 70 else 0
        
        # ক্যান্ডলেস্টিক প্যাটার্নস স্টেট
        for pattern in ['Hammer', 'BullishEngulfing', 'MorningStar', 'Doji']:
            if pattern in features_dict:
                features_dict[f'{pattern}_present'] = int(features_dict[pattern])
        
        # 5. মডেলের জন্য ফিচার ভেক্টর তৈরি
        if self.feature_importance is not None:
            model_features = self.feature_importance['feature'].head(30).tolist()
        else:
            model_features = list(features_dict.keys())
        
        available_features = [f for f in model_features if f in features_dict]
        
        if len(available_features) < 10:
            print(f"   ⚠️ পর্যাপ্ত ফিচার নেই: {len(available_features)}")
            continue
        
        feature_vector = [features_dict[f] for f in available_features]
        feature_df = pd.DataFrame([feature_vector], columns=available_features)
        
        # 6. প্রেডিকশন
        predicted_reward = self.regression_model.predict(feature_df)[0]
        
        # 7. টেকনিক্যাল সিগন্যাল স্কোর
        technical_score = 0.5  # base
        
        # RSI স্কোর
        if 'rsi' in features_dict:
            rsi = features_dict['rsi']
            if 30 <= rsi <= 70:
                technical_score += 0.1
            elif rsi < 30:
                technical_score += 0.2
            elif rsi > 80:
                technical_score -= 0.1
        
        # MACD স্কোর
        if 'macd_cross' in features_dict and features_dict['macd_cross'] == 1:
            technical_score += 0.15
        
        # BB স্কোর
        if 'close_vs_bb_lower' in features_dict and features_dict['close_vs_bb_lower'] > -0.05:
            technical_score += 0.1
        
        # ক্যান্ডলেস্টিক প্যাটার্নস
        bullish_patterns = ['Hammer_present', 'BullishEngulfing_present', 'MorningStar_present']
        for pattern in bullish_patterns:
            if pattern in features_dict and features_dict[pattern] == 1:
                technical_score += 0.05
        
        technical_score = min(max(technical_score, 0), 1)
        
        # 8. ফাইনাল স্কোর
        confidence = abs(predicted_reward) * technical_score
        
        # 9. রেজাল্ট সংরক্ষণ
        pred_result = {
            'symbol': symbol,
            'buy_date': buy_date,
            'prediction_date': buy_date + pd.Timedelta(days=days_ahead_val),
            'days_ahead': days_ahead_val,
            'buy_price': buy_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'predicted_reward': predicted_reward,
            'predicted_profit': predicted_reward > 0,
            'technical_score': technical_score,
            'confidence': confidence,
            'current_rsi': features_dict.get('rsi', None),
            'current_macd': features_dict.get('macd', None),
            'current_atr': features_dict.get('atr', None),
            'bb_position': features_dict.get('close_vs_bb_lower', None),
            'has_bullish_pattern': any(
                features_dict.get(f'{p}_present', 0) == 1 
                for p in ['Hammer', 'BullishEngulfing', 'MorningStar']
            )
        }
        
        predictions.append(pred_result)
    
    if len(predictions) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # 10. রেজাল্ট DataFrame
    result_df = pd.DataFrame(predictions)
    
    # 11. শুধু positive prediction ফিল্টার
    positive_predictions = result_df[result_df['predicted_profit'] == True].copy()
    
    # 12. Confidence বেসড সর্টিং
    if len(positive_predictions) > 0:
        positive_predictions = positive_predictions.sort_values('confidence', ascending=False)
        
        # Risk management
        positive_predictions['stop_loss'] = positive_predictions['buy_price'] * 0.95
        positive_predictions['take_profit'] = positive_predictions['buy_price'] * 1.10
        positive_predictions['risk_reward_ratio'] = (
            positive_predictions['take_profit'] - positive_predictions['buy_price']
        ) / (positive_predictions['buy_price'] - positive_predictions['stop_loss'])
        
        # Combined score
        positive_predictions['combined_score'] = (
            positive_predictions['predicted_reward'] * 
            positive_predictions['confidence'] * 
            positive_predictions['risk_reward_ratio']
        )
        
        print(f"   ✅ {len(positive_predictions)} টি positive সিগন্যাল পাওয়া গেছে")
        print(f"   📊 Best signal: {positive_predictions.iloc[0]['predicted_reward']:.3f} reward, "
              f"{positive_predictions.iloc[0]['confidence']:.1%} confidence")
    
    return result_df, positive_predictions

def save_model(self, path):
    """
    মডেল সেভ করে
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if self.regression_model is not None:
        self.regression_model.save_model(path)
        print(f"   💾 Regression model saved: {path}")
    else:
        print(f"   ⚠️ No model to save")

def load_model(self, path):
    """
    মডেল লোড করে
    """
    self.regression_model = xgb.XGBRegressor()
    self.regression_model.load_model(path)
    print(f"   📥 Regression model loaded: {path}")



# ডিরেক্টরি তৈরি
os.makedirs('./models', exist_ok=True)
os.makedirs('./csv', exist_ok=True)

# 1. ডাটা লোড
print("\n📥 ডাটা লোড করা হচ্ছে...")

try:
    market_data = pd.read_csv("./csv/mongodb.csv")
    trade_data = pd.read_csv("./csv/trade_stock.csv")
    
    # তারিখ কনভার্ট
    market_data['date'] = pd.to_datetime(market_data['date'])
    trade_data['date'] = pd.to_datetime(trade_data['date'])
    
    print(f"   ✅ মার্কেট ডাটা: {market_data.shape}")
    print(f"   ✅ ট্রেড ডাটা: {trade_data.shape}")
    
    # কলাম চেক
    print(f"\n📊 মার্কেট ডাটা কলামস ({len(market_data.columns)}):")
    print(f"   {', '.join(market_data.columns.tolist())}")
    
    print(f"\n📊 ট্রেড ডাটা কলামস ({len(trade_data.columns)}):")
    print(f"   {', '.join(trade_data.columns.tolist())}")
    
except FileNotFoundError as e:
    print(f"❌ ফাইল পাওয়া যায়নি: {e}")
    print(f"   ফাইল পাথ চেক করুন: ./csv/mongodb.csv এবং ./csv/trade_stock.csv")
    return
except Exception as e:
    print(f"❌ ডাটা লোড করতে সমস্যা: {e}")
    return

# 2. Symbol লিস্ট তৈরি
market_symbols = set(market_data['symbol'].unique())
trade_symbols = set(trade_data['symbol'].unique())
common_symbols = sorted(market_symbols.intersection(trade_symbols))

print(f"\n📊 Symbol বিশ্লেষণ:")
print(f"   📈 মার্কেট symbols: {len(market_symbols)}")
print(f"   💰 ট্রেড symbols: {len(trade_symbols)}")
print(f"   ✅ কমন symbols: {len(common_symbols)}")

if len(common_symbols) == 0:
    print("❌ কোন কমন symbol পাওয়া যায়নি!")
    print("   মার্কেট এবং ট্রেড ডাটায় মিলনসই symbol নেই")
    return

print(f"\n🎯 প্রথম 10 টি symbol: {common_symbols[:10]}")

# 3. প্রতিটি symbol-এর জন্য আলাদা ট্রেনিং
results = []
all_positive_signals = []

print(f"\n🚀 {len(common_symbols)} টি symbol-এর ট্রেনিং শুরু...")
print("=" * 70)

for i, symbol in enumerate(common_symbols, 1):
    print(f"\n[{i}/{len(common_symbols)}] 🔄 Processing: {symbol}")
    print("-" * 50)
    
    # Symbol-specific ডাটা ফিল্টার
    symbol_market = market_data[market_data['symbol'] == symbol].copy()
    symbol_trade = trade_data[trade_data['symbol'] == symbol].copy()
    
    # ডাটা চেক
    if len(symbol_market) < 30:
        print(f"   ⚠️ মার্কেট ডাটা কম: {len(symbol_market)} days (minimum 30 required)")
        continue
        
    if len(symbol_trade) == 0:
        print(f"   ⚠️ কোন ট্রেড সিগন্যাল নেই")
        continue
    
    print(f"   📈 মার্কেট ডাটা: {symbol_market.shape[0]} days")
    print(f"   🎯 ট্রেড সিগন্যাল: {len(symbol_trade)} signals")
    
    # 4. মডেল তৈরি এবং ট্রেন
    model = XGBoostTradingModel(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05
    )
    
    try:
        # টেকনিক্যাল ইন্ডিকেটরস সহ ট্রেনিং
        r2_score, accuracy = model.train_with_technical_indicators(symbol_market, symbol_trade)
        
        # রেজাল্ট সংরক্ষণ
        result_entry = {
            'symbol': symbol,
            'r2_score': r2_score,
            'accuracy': accuracy,
            'market_days': len(symbol_market),
            'trade_signals': len(symbol_trade),
            'signal_percentage': len(symbol_trade) / len(symbol_market) * 100,
            'success': r2_score > 0.3 and accuracy > 0.6
        }
        
        results.append(result_entry)
        
        if result_entry['success']:
            # 5. প্রেডিকশন তৈরি
            print(f"   🔮 প্রেডিকশন তৈরি করা হচ্ছে...")
            all_preds, positive_signals = model.predict_with_technical_analysis(symbol_market, symbol_trade, days_ahead=5)
            
            if len(positive_signals) > 0:
                all_positive_signals.append(positive_signals)
                print(f"   ✅ {len(positive_signals)} টি positive সিগন্যাল পাওয়া গেছে")
                
                # মডেল সেভ
                model_path = f'./models/xgboost_tech_{symbol.replace("/", "_")}.json'
                model.save_model(model_path)
                
                # Symbol-specific প্রেডিকশন সেভ
                signals_path = f'./csv/predictions_tech_{symbol.replace("/", "_")}.csv'
                positive_signals.to_csv(signals_path, index=False)
            
        else:
            print(f"   ⚠️ Poor model performance, skipping predictions")
            
    except Exception as e:
        print(f"   ❌ ট্রেনিং ব্যর্থ: {str(e)[:80]}")
        results.append({
            'symbol': symbol,
            'r2_score': 0.0,
            'accuracy': 0.0,
            'market_days': len(symbol_market),
            'trade_signals': len(symbol_trade),
            'signal_percentage': len(symbol_trade) / len(symbol_market) * 100,
            'success': False,
            'error': str(e)[:80]
        })

# 6. ফাইনাল রেজাল্টস প্রসেস
print(f"\n{'='*70}")
print("📊 FINAL TRAINING SUMMARY")
print(f"{'='*70}")

if results:
    results_df = pd.DataFrame(results)
    
    # সফল ট্রেনিং
    successful = results_df[results_df['success'] == True]
    failed = results_df[results_df['success'] == False]
    
    print(f"✅ সফলভাবে ট্রেন হয়েছে: {len(successful)} symbols")
    print(f"❌ ব্যর্থ হয়েছে: {len(failed)} symbols")
    
    if len(successful) > 0:
        print(f"\n🏆 Top 5 Performing Symbols:")
        top_symbols = successful.sort_values('r2_score', ascending=False).head()
        for idx, row in top_symbols.iterrows():
            print(f"   {row['symbol']}:")
            print(f"     R² Score: {row['r2_score']:.3f}, Accuracy: {row['accuracy']:.3f}")
            print(f"     Signals: {row['trade_signals']}/{row['market_days']} ({row['signal_percentage']:.1f}%)")
    
    # সমস্ত positive সিগন্যাল একত্রিত
    if all_positive_signals:
        final_signals = pd.concat(all_positive_signals, ignore_index=True)
        
        # combined_score বেসড সর্ট
        if 'combined_score' in final_signals.columns:
            final_signals = final_signals.sort_values('combined_score', ascending=False)
        
        # CSV তে সেভ
        final_signals.to_csv("./csv/xgboost_tech_predictions.csv", index=False)
        
        print(f"\n📁 PREDICTIONS SUMMARY:")
        print(f"   মোট positive সিগন্যাল: {len(final_signals)}")
        print(f"   সেভ হয়েছে: ./csv/xgboost_tech_predictions.csv")
        
        # শীর্ষ 5 সিগন্যাল ডিসপ্লে
        if len(final_signals) > 0:
            print(f"\n🎯 TOP 5 TRADING OPPORTUNITIES:")
            top_5 = final_signals.head(5)
            for idx, row in top_5.iterrows():
                print(f"   {row['symbol']} - Buy: {row['buy_date'].date()}, Predict: {row['prediction_date'].date()}")
                print(f"     Buy Price: {row['buy_price']:.2f}, Pred Reward: {row['predicted_reward']:.3f}")
                print(f"     Confidence: {row['confidence']:.1%}, Tech Score: {row['technical_score']:.2f}")
                print(f"     RSI: {row['current_rsi']:.1f}, MACD: {row['current_macd']:.3f}")
                if row['has_bullish_pattern']:
                    print(f"     ✓ Bullish Pattern Present")
    
    # রেজাল্টস CSV তে সেভ
    results_df.to_csv("./csv/xgboost_tech_training_results.csv", index=False)
    print(f"\n📄 ট্রেনিং রেজাল্টস: ./csv/xgboost_tech_training_results.csv")
    
    # সামারি স্ট্যাটস
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   গড় R² Score: {results_df['r2_score'].mean():.3f}")
    print(f"   গড় Accuracy: {results_df['accuracy'].mean():.3f}")
    print(f"   সর্বোচ্চ R² Score: {results_df['r2_score'].max():.3f}")
    print(f"   মোট মডেল: {len(results_df)}")
    
    # Performance categories
    excellent = results_df[results_df['r2_score'] > 0.7]
    good = results_df[(results_df['r2_score'] > 0.5) & (results_df['r2_score'] <= 0.7)]
    average = results_df[(results_df['r2_score'] > 0.3) & (results_df['r2_score'] <= 0.5)]
    poor = results_df[results_df['r2_score'] <= 0.3]
    
    print(f"\n📊 PERFORMANCE CATEGORIES:")
    print(f"   Excellent (R² > 0.7): {len(excellent)} symbols")
    print(f"   Good (R² 0.5-0.7): {len(good)} symbols")
    print(f"   Average (R² 0.3-0.5): {len(average)} symbols")
    print(f"   Poor (R² ≤ 0.3): {len(poor)} symbols")
    
else:
    print("❌ কোন symbol সফলভাবে ট্রেন হয়নি")
    print("   সম্ভাব্য কারণ:")
    print("   1. খুব কম buy সিগন্যাল")
    print("   2. টেকনিক্যাল ইন্ডিকেটরস না থাকা")
    print("   3. ডাটা quality issue")

print(f"\n{'='*70}")
print("✅ PROGRAM COMPLETED")
print(f"{'='*70}")
