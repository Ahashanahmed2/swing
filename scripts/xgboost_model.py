# xgboost_model.py - সম্পূর্ণ আপডেটেড ভার্সন
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class XGBoostTradingModel:
    def __init__(self, n_estimators=1000, max_depth=5, learning_rate=0.01):
        self.model = None
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
            'eval_metric': 'logloss',  # ✅ বাইনারি ক্লাসিফিকেশনের জন্য
            'objective': 'binary:logistic'  # ✅ যোগ করুন
        }
        
    def prepare_data(self, market_data, trade_data):
    """
    Symbol-specific ডাটা প্রিপেয়ার করে
    """
    symbol = market_data['symbol'].iloc[0] if len(market_data) > 0 else 'UNKNOWN'
    
    # 1. ফিচার তৈরি
    market_data = market_data.copy()
    
    # প্রাইস-বেসড ফিচার
    market_data['returns'] = market_data['close'].pct_change()
    market_data['returns_ma'] = market_data['returns'].rolling(5).mean()
    market_data['volatility'] = market_data['returns'].rolling(5).std()
    
    # ভলিউম ফিচার
    market_data['volume_ma'] = market_data['volume'].rolling(5).mean()
    market_data['volume_ratio'] = market_data['volume'] / market_data['volume_ma']
    
    # প্রাইস ট্রেন্ড ফিচার
    market_data['price_ma_5'] = market_data['close'].rolling(5).mean()
    market_data['price_ma_10'] = market_data['close'].rolling(10).mean()
    market_data['price_ma_ratio'] = market_data['price_ma_5'] / market_data['price_ma_10']
    
    # 2. ডাটা মার্জ
    merged_data = pd.merge(
        market_data, 
        trade_data, 
        on=['symbol', 'date'], 
        how='left',
        suffixes=('', '_trade')
    )
    
    # 3. টার্গেট ভ্যারিয়েবল তৈরি
    merged_data['signal'] = merged_data['buy'].notna().astype(int)
    
    # RRR-based স্ট্রং বাই ডিটেকশন
    merged_data['signal_type'] = 0  # ডিফল্ট: নো সিগন্যাল
    
    if 'buy' in merged_data.columns and merged_data['buy'].notna().any():
        buy_mask = merged_data['buy'].notna()
        merged_data.loc[buy_mask, 'signal_type'] = 1  # সব buy কে রেগুলার বাই
        
        # যদি RRR থাকে তবে স্ট্রং বাই চিহ্নিত
        if 'RRR' in merged_data.columns:
            # Symbol-specific থ্রেশহোল্ড
            valid_rrr = merged_data.loc[buy_mask, 'RRR']
            if valid_rrr.notna().any():
                median_rrr = valid_rrr.median()
                strong_buy_threshold = max(median_rrr * 1.2, 1.5)  # মিডিয়ান থেকে 20% বেশি
                
                strong_buy_mask = buy_mask & (merged_data['RRR'] > strong_buy_threshold)
                merged_data.loc[strong_buy_mask, 'signal_type'] = 2
    else:
        # যদি কোন buy সিগন্যাল না থাকে
        merged_data['signal'] = 0
        merged_data['signal_type'] = 0
    
    # 4. ফিচার সিলেকশন
    base_features = [
        'open', 'high', 'low', 'close', 'volume',
        'returns', 'returns_ma', 'volatility',
        'volume_ma', 'volume_ratio',
        'price_ma_5', 'price_ma_10', 'price_ma_ratio'
    ]
    
    # টেকনিক্যাল ইন্ডিকেটরস যোগ (যদি থাকে)
    tech_indicators = ['rsi', 'macd', 'macd_hist', 'atr', 'marketCap']
    for indicator in tech_indicators:
        if indicator in merged_data.columns:
            base_features.append(indicator)
    
    # ট্রেড-বেসড ফিচার (যদি থাকে)
    trade_features = ['diff', 'RRR', 'position_size']
    for feature in trade_features:
        if feature in merged_data.columns:
            base_features.append(feature)
    
    # 5. শুধু সেই ফিচারগুলো নিন যেগুলো আছে
    available_features = [f for f in base_features if f in merged_data.columns]
    
    # 6. NaN ভ্যালু হ্যান্ডলিং
    original_len = len(merged_data)
    valid_mask = merged_data[available_features].notna().all(axis=1)
    merged_data = merged_data[valid_mask].copy()
    dropped_rows = original_len - len(merged_data)
    
    return merged_data, available_features

def train(self, market_data, trade_data):
    """
    Symbol-specific XGBoost মডেল ট্রেন করে (SMOTE ছাড়া)
    """
    symbol = market_data['symbol'].iloc[0] if len(market_data) > 0 else 'UNKNOWN'
    print(f"   🔄 {symbol} - মডেল ট্রেনিং শুরু...")

    # 1. ডাটা প্রিপেয়ার
    data, features = self.prepare_data(market_data, trade_data)

    if len(data) < 30:  # কমপক্ষে 30 দিনের ডাটা চাই
        print(f"   ⚠️ পর্যাপ্ত ডাটা নেই: {len(data)} days")
        return 0.0, 0.0

    # 2. ফিচার এবং টার্গেট আলাদা করা
    X = data[features]
    y_binary = data['signal']  # বাইনারি ক্লাসিফিকেশন

    # 3. ক্লাস ডিস্ট্রিবিউশন চেক
    class_counts = Counter(y_binary)
    total_samples = len(y_binary)

    print(f"   📊 ডাটা আকার: {X.shape}")
    print(f"   🎯 ক্লাস ডিস্ট্রিবিউশন: {dict(class_counts)}")
    print(f"   Buy সিগন্যাল: {class_counts.get(1, 0)} / {total_samples} ({class_counts.get(1, 0)/total_samples*100:.1f}%)")

    # 4. যদি buy সিগন্যাল খুব কম থাকে
    if class_counts.get(1, 0) < 2:
        print(f"   ⚠️ খুব কম buy সিগন্যাল ({class_counts.get(1, 0)}), মডেল ট্রেনিং সম্ভব নয়")
        return 0.0, 0.0

    # 5. ট্রেন-টেস্ট স্প্লিট (অবশ্যই stratified)
    try:
        # নিশ্চিত করুন যে y_binary-তে কমপক্ষে 2টি ক্লাস আছে
        unique_classes = np.unique(y_binary)
        if len(unique_classes) < 2:
            print(f"   ❌ শুধু 1টি ক্লাস পাওয়া গেছে: {unique_classes}")
            print(f"   ✅ কৃত্রিম buy সিগন্যাল তৈরি করা হচ্ছে...")
            
            # যদি সব 0 থাকে, 1টি কৃত্রিম buy সিগন্যাল যোগ করুন
            if len(data) > 10:
                # প্রথম 10টি ডাটার মধ্যে 1টি buy মার্ক করুন
                y_binary.iloc[0] = 1
                print(f"   ✅ 1টি কৃত্রিম buy সিগন্যাল যোগ করা হয়েছে")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, 
            test_size=0.3, 
            random_state=42,
            stratify=y_binary
        )
    except Exception as e:
        print(f"   ⚠️ Stratified split সম্ভব নয়: {str(e)[:50]}")
        # Regular split ব্যবহার করুন
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, 
            test_size=0.3, 
            random_state=42
        )

    # 6. ক্লাস ওয়েট ক্যালকুলেট
    n_class_0 = np.sum(y_train == 0)
    n_class_1 = np.sum(y_train == 1)

    if n_class_1 == 0:
        print(f"   ⚠️ ট্রেনিং সেটে কোন buy সিগন্যাল নেই")
        # একটি কৃত্রিম buy সিগন্যাল যোগ করুন
        if len(X_train) > 0:
            y_train.iloc[0] = 1
            n_class_1 = 1
            print(f"   ✅ 1টি কৃত্রিম buy সিগন্যাল যোগ করা হয়েছে")

    scale_pos_weight = n_class_0 / max(n_class_1, 1)  # Zero division এড়ানো
    print(f"   ⚖️ Class Weight: {scale_pos_weight:.2f}")
    print(f"   🏋️ ট্রেনিং স্যাম্পল: {X_train.shape[0]}")
    print(f"   🧪 টেস্টিং স্যাম্পল: {X_test.shape[0]}")

    # 7. XGBoost মডেল তৈরি (SMOTE ছাড়া)
    print("   🤖 মডেল ট্রেনিং শুরু...")

    try:
        self.model = xgb.XGBClassifier(
            n_estimators=self.params['n_estimators'],
            max_depth=self.params['max_depth'],
            learning_rate=self.params['learning_rate'],
            subsample=self.params['subsample'],
            colsample_bytree=self.params['colsample_bytree'],
            random_state=self.params['random_state'],
            early_stopping_rounds=self.params['early_stopping_rounds'],
            eval_metric=self.params['eval_metric'],
            objective=self.params['objective'],
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            verbosity=0
        )

        # 8. মডেল ট্রেনিং (SMOTE ব্যালেন্সড ডাটা ছাড়াই)
        self.model.fit(
            X_train,
            y_train,  # ✅ Original ডাটা, SMOTE ব্যালেন্সড নয়
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # 9. পারফরম্যান্স ইভ্যালুয়েশন
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"   ✅ ট্রেনিং সম্পূর্ণ!")
        print(f"   🎯 Accuracy: {accuracy:.4f}")
        print(f"   📈 F1 Score: {f1:.4f}")

        if y_test.sum() > 0:  # শুধু যদি টেস্টে buy সিগন্যাল থাকে
            print(f"\n   📊 Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['No Signal', 'Buy Signal']))

        # 10. ফিচার ইম্পরটেন্স
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"   🏆 Top 3 Important Features:")
            for i, row in self.feature_importance.head(3).iterrows():
                print(f"      {row['feature']}: {row['importance']:.4f}")

        return accuracy, f1

    except Exception as e:
        print(f"   ❌ ট্রেনিং এরর: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0

def predict(self, market_data, trade_data=None):
    """
    নতুন ডাটার উপর প্রেডিকশন করে
    """
    if self.model is None:
        raise ValueError("মডেল ট্রেন করা হয়নি। প্রথমে .train() মেথড কল করুন")
    
    # যদি trade_data না দেওয়া হয়, শুধু মার্কেট ডাটা ব্যবহার
    if trade_data is None:
        trade_data = pd.DataFrame(columns=['symbol', 'date', 'buy', 'RRR'])
    
    # ডাটা প্রিপেয়ার
    data, features = self.prepare_data(market_data, trade_data)
    
    if len(data) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # প্রেডিকশন
    predictions = self.model.predict(data[features])
    probabilities = self.model.predict_proba(data[features])
    
    # রেজাল্ট ডাটাফ্রেম
    result_df = data[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']].copy()
    result_df['predicted_signal'] = predictions
    result_df['signal_probability'] = probabilities[:, 1]  # ক্লাস 1-এর প্রোবাবিলিটি
    
    # মেট্রিক্স যোগ
    result_df['returns'] = result_df['close'].pct_change()
    result_df['volatility'] = result_df['returns'].rolling(5).std()
    
    # সিগন্যাল ফিল্টার (শুধু বাই সিগন্যাল)
    buy_signals = result_df[result_df['predicted_signal'] == 1].copy()
    
    # রিস্ক ম্যানেজমেন্ট প্যারামিটার যোগ
    if len(buy_signals) > 0:
        # ATR-বেসড স্টপ লস (যদি atr থাকে)
        if 'atr' in data.columns:
            buy_signals = buy_signals.merge(
                data[['date', 'atr']], 
                on='date', 
                how='left'
            )
            buy_signals['stop_loss'] = buy_signals['close'] - (buy_signals['atr'] * 1.5)
            buy_signals['take_profit'] = buy_signals['close'] + (buy_signals['atr'] * 3)
        else:
            buy_signals['stop_loss'] = buy_signals['close'] * 0.95
            buy_signals['take_profit'] = buy_signals['close'] * 1.10
        
        buy_signals['risk_reward_ratio'] = (buy_signals['take_profit'] - buy_signals['close']) / (buy_signals['close'] - buy_signals['stop_loss'])
        buy_signals['position_size'] = 100  # ডিফল্ট
        
        # কনফিডেন্স লেভেল বেসড সর্টিং
        buy_signals['confidence'] = buy_signals['signal_probability'] * buy_signals['risk_reward_ratio']
    
    return result_df, buy_signals

def save_model(self, path):
    """
    মডেল সেভ করে
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    self.model.save_model(path)

def load_model(self, path):
    """
    মডেল লোড করে
    """
    self.model = xgb.XGBClassifier()
    self.model.load_model(path)

def main():
    """
    মেইন এক্সিকিউশন ফাংশন
    """
    print("=" * 70)
    print("XGBOOST ট্রেডিং মডেল - অ্যাডভান্সড ট্রেনিং")
    print("=" * 70)
    
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
all_buy_signals = []

print(f"\n🚀 {len(common_symbols)} টি symbol-এর ট্রেনিং শুরু...")
print("=" * 70)

for i, symbol in enumerate(common_symbols, 1):
    print(f"\n[{i}/{len(common_symbols)}] 🔄 Processing: {symbol}")
    print("-" * 50)
    
    # Symbol-specific ডাটা ফিল্টার
    symbol_market = market_data[market_data['symbol'] == symbol].copy()
    symbol_trade = trade_data[trade_data['symbol'] == symbol].copy()
    
    # ডাটা চেক
    if len(symbol_market) < 50:
        print(f"   ⚠️ মার্কেট ডাটা কম: {len(symbol_market)} days (minimum 50 required)")
        continue
        
    if len(symbol_trade) == 0:
        print(f"   ⚠️ কোন ট্রেড সিগন্যাল নেই")
        continue
    
    print(f"   📈 মার্কেট ডাটা: {symbol_market.shape[0]} days")
    print(f"   🎯 ট্রেড সিগন্যাল: {len(symbol_trade)} signals")
    
    # 4. মডেল তৈরি এবং ট্রেন
    model = XGBoostTradingModel(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05
    )
    
    try:
        accuracy, f1 = model.train(symbol_market, symbol_trade)
        
        # রেজাল্ট সংরক্ষণ
        result_entry = {
            'symbol': symbol,
            'accuracy': accuracy,
            'f1_score': f1,
            'market_days': len(symbol_market),
            'trade_signals': len(symbol_trade),
            'signal_percentage': len(symbol_trade) / len(symbol_market) * 100,
            'success': accuracy > 0
        }
        
        results.append(result_entry)
        
        if accuracy > 0.5:  # শুধু ভালো মডেলগুলো প্রেডিক্ট করবে
            # 5. প্রেডিকশন তৈরি
            print(f"   🔮 প্রেডিকশন তৈরি করা হচ্ছে...")
            all_preds, buy_signals = model.predict(symbol_market)
            
            if len(buy_signals) > 0:
                all_buy_signals.append(buy_signals)
                print(f"   ✅ {len(buy_signals)} টি বাই সিগন্যাল পাওয়া গেছে")
                
                # মডেল সেভ
                model_path = f'./models/xgboost_{symbol.replace("/", "_")}.json'
                model.save_model(model_path)
                
                # Symbol-specific প্রেডিকশন সেভ
                buy_signals_path = f'./csv/predictions_{symbol.replace("/", "_")}.csv'
                buy_signals.to_csv(buy_signals_path, index=False)
            
        else:
            print(f"   ⚠️ Low accuracy, skipping predictions")
            
    except Exception as e:
        print(f"   ❌ ট্রেনিং ব্যর্থ: {str(e)[:80]}")
        results.append({
            'symbol': symbol,
            'accuracy': 0.0,
            'f1_score': 0.0,
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
        top_symbols = successful.sort_values('f1_score', ascending=False).head()
        for idx, row in top_symbols.iterrows():
            print(f"   {row['symbol']}:")
            print(f"     Accuracy: {row['accuracy']:.3f}, F1: {row['f1_score']:.3f}")
            print(f"     Signals: {row['trade_signals']}/{row['market_days']} ({row['signal_percentage']:.1f}%)")
    
    # সমস্ত বাই সিগন্যাল একত্রিত
    if all_buy_signals:
        final_signals = pd.concat(all_buy_signals, ignore_index=True)
        
        # কনফিডেন্স বেসড সর্ট
        if 'confidence' in final_signals.columns:
            final_signals = final_signals.sort_values('confidence', ascending=False)
        
        # CSV তে সেভ
        final_signals.to_csv("./csv/xgboost_all_predictions.csv", index=False)
        
        print(f"\n📁 PREDICTIONS SUMMARY:")
        print(f"   মোট বাই সিগন্যাল: {len(final_signals)}")
        print(f"   সেভ হয়েছে: ./csv/xgboost_all_predictions.csv")
        
        # শীর্ষ 5 সিগন্যাল ডিসপ্লে
        if len(final_signals) > 0:
            print(f"\n🎯 TOP 5 TRADING OPPORTUNITIES:")
            top_5 = final_signals.head(5)
            for idx, row in top_5.iterrows():
                confidence = row.get('confidence', row.get('signal_probability', 0))
                print(f"   {row['symbol']} - {row['date'].date()}")
                print(f"     Price: {row['close']:.2f}, Signal: {row['signal_probability']:.1%}")
                if 'risk_reward_ratio' in row:
                    print(f"     R/R: {row['risk_reward_ratio']:.2f}, Confidence: {confidence:.3f}")
    
    # রেজাল্টস CSV তে সেভ
    results_df.to_csv("./csv/xgboost_training_results.csv", index=False)
    print(f"\n📄 ট্রেনিং রেজাল্টস: ./csv/xgboost_training_results.csv")
    
    # সামারি স্ট্যাটস
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   গড় Accuracy: {results_df['accuracy'].mean():.3f}")
    print(f"   গড় F1 Score: {results_df['f1_score'].mean():.3f}")
    print(f"   সর্বোচ্চ Accuracy: {results_df['accuracy'].max():.3f}")
    print(f"   মোট মডেল: {len(results_df)}")
    
else:
    print("❌ কোন symbol সফলভাবে ট্রেন হয়নি")
    print("   সম্ভাব্য কারণ:")
    print("   1. খুব কম buy সিগন্যাল")
    print("   2. ডাটা quality issue")
    print("   3. Feature engineering সমস্যা")

print(f"\n{'='*70}")
print("✅ PROGRAM COMPLETED")
print(f"{'='*70}")

if name="main:
  main()