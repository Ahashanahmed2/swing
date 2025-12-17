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
            'early_stopping_rounds': 50,  # ✅ এখানে সরিয়ে আনা হয়েছে
            'eval_metric': 'mlogloss'  # মাল্টি-ক্লাসের জন্য উপযুক্ত
        }
        
    def prepare_data(self, market_data, trade_data):
        """
        মার্কেট এবং ট্রেড ডাটা প্রিপেয়ার করে
        """
        print("📊 লোডিং এবং ডাটা প্রিপেয়ার করা হচ্ছে...")
        print(f"   মার্কেট ডাটা আকার: {market_data.shape}")
        print(f"   ট্রেড ডাটা আকার: {trade_data.shape}")
        
        # 1. ফিচার তৈরি
        market_data['returns'] = market_data['close'].pct_change()
        market_data['volatility'] = market_data['returns'].rolling(5).std()
        market_data['volume_ma'] = market_data['volume'].rolling(5).mean()
        
        # 2. ডাটা মার্জ
        merged_data = pd.merge(market_data, trade_data, 
                              on=['symbol', 'date'], 
                              how='left')
        
        # 3. টার্গেট ভ্যারিয়েবল তৈরি
        merged_data['signal'] = merged_data['buy'].notna().astype(int)
        merged_data['signal_type'] = 0  # ডিফল্ট: নো সিগন্যাল
        
        # ক্লাস 1: রেগুলার বাই সিগন্যাল
        buy_mask = merged_data['buy'].notna()
        merged_data.loc[buy_mask, 'signal_type'] = 1
        
        # ক্লাস 2: স্ট্রং বাই সিগন্যাল (যদি RRR1 > 2.0)
        strong_buy_mask = buy_mask & (merged_data['RRR'] > 2.0)
        merged_data.loc[strong_buy_mask, 'signal_type'] = 2
        
        print(f"   বাই সিগন্যাল পাওয়া গেছে: {merged_data['signal'].sum()} out of {len(merged_data)} samples ({merged_data['signal'].sum()/len(merged_data)*100:.2f}%)")
        print(f"   টার্গেট ডিস্ট্রিবিউশন: {dict(Counter(merged_data['signal_type']))}")
        
        # 4. ফিচার সিলেকশন
        features = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'volatility', 'volume_ma',
            'marketCap', 'rsi', 'macd', 'macd_hist',
            'atr', 'Hammer', 'BullishEngulfing', 
            'MorningStar', 'Doji', 'diff'
        ]
        
        # শুধু সেই ফিচারগুলো নিন যেগুলো আছে
        available_features = [f for f in features if f in merged_data.columns]
        print(f"   ব্যবহার করা হচ্ছে {len(available_features)} টি ফিচার")
        
        # 5. NaN ভ্যালু হ্যান্ডলিং
        original_len = len(merged_data)
        merged_data = merged_data.dropna(subset=available_features + ['signal_type'])
        dropped_rows = original_len - len(merged_data)
        print(f"   NaN ভ্যালু সহ {dropped_rows} টি সারি ড্রপ করা হয়েছে")
        print(f"   ফাইনাল ডাটা আকার: {merged_data.shape}")
        
        return merged_data, available_features
    
    def train(self, market_data, trade_data):
        """
        XGBoost মডেল ট্রেন করে
        """
        # 1. ডাটা প্রিপেয়ার
        data, features = self.prepare_data(market_data, trade_data)
        
        # 2. ফিচার এবং টার্গেট আলাদা করা
        X = data[features]
        y_binary = data['signal']  # বাইনারি ক্লাসিফিকেশন
        y_multi = data['signal_type']  # মাল্টি-ক্লাস ক্লাসিফিকেশন
        
        # 3. ট্রেন-টেস্ট স্প্লিট
        X_train, X_test, y_bin_train, y_bin_test, y_multi_train, y_multi_test = train_test_split(
            X, y_binary, y_multi, test_size=0.3, random_state=42, stratify=y_multi
        )
        
        print(f"\n🤖 XGBoost মডেল ট্রেনিং শুরু...")
        print(f"   ট্রেনিং স্যাম্পল: {X_train.shape[0]}")
        print(f"   টেস্টিং স্যাম্পল: {X_test.shape[0]}")
        print(f"   ক্লাস ডিস্ট্রিবিউশন - বাইনারি: {dict(Counter(y_bin_train))}")
        print(f"   ক্লাস ডিস্ট্রিবিউশন - মাল্টি: {dict(Counter(y_multi_train))}")
        
        # 4. SMOTE অ্যাপ্লাই করা (শ্রেণী ব্যালেন্সের জন্য)
        print("   ক্লাস ব্যালেন্সিংয়ের জন্য SMOTE অ্যাপ্লাই করা হচ্ছে...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_bin_train_balanced = smote.fit_resample(X_train, y_bin_train)
        print(f"   SMOTE পরে: {dict(Counter(y_bin_train_balanced))}")
        
        # 5. ক্লাস ওয়েট ক্যালকুলেট
        scale_pos_weight = len(y_bin_train_balanced[y_bin_train_balanced == 0]) / len(y_bin_train_balanced[y_bin_train_balanced == 1])
        print(f"   ক্লাস ওয়েট (scale_pos_weight): {scale_pos_weight:.2f}")
        
        # 6. XGBoost মডেল তৈরি
        print("   বাইনারি ক্লাসিফিকেশন মডেল ট্রেনিং...")
        
        # ✅ early_stopping_rounds এখন কনস্ট্রাক্টরে
        self.model = xgb.XGBClassifier(
            n_estimators=self.params['n_estimators'],
            max_depth=self.params['max_depth'],
            learning_rate=self.params['learning_rate'],
            subsample=self.params['subsample'],
            colsample_bytree=self.params['colsample_bytree'],
            random_state=self.params['random_state'],
            early_stopping_rounds=self.params['early_stopping_rounds'],  # ✅ এখানে
            eval_metric=self.params['eval_metric'],
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False
        )
        
        # 7. মডেল ট্রেনিং
        # ✅ eval_set এখন .fit() মেথডে
        self.model.fit(
            X_train_balanced,
            y_bin_train_balanced,
            eval_set=[(X_test, y_bin_test)],  # ✅ এখানে
            verbose=False
        )
        
        # 8. পারফরম্যান্স ইভ্যালুয়েশন
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_bin_test, y_pred)
        f1 = f1_score(y_bin_test, y_pred)
        
        print(f"\n✅ মডেল ট্রেনিং সম্পূর্ণ!")
        print(f"   একুরেসি: {accuracy:.4f}")
        print(f"   F1 স্কোর: {f1:.4f}")
        print(f"\n📊 ক্লাসিফিকেশন রিপোর্ট:")
        print(classification_report(y_bin_test, y_pred, target_names=['No Signal', 'Buy Signal']))
        
        # 9. ফিচার ইম্পরটেন্স
        self.feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🏆 Top 5 গুরুত্বপূর্ণ ফিচার:")
        for i, row in self.feature_importance.head().iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return accuracy, f1
    
    def predict(self, market_data, trade_data):
        """
        নতুন ডাটার উপর প্রেডিকশন করে
        """
        data, features = self.prepare_data(market_data, trade_data)
        
        if self.model is None:
            raise ValueError("মডেল ট্রেন করা হয়নি। প্রথমে .train() মেথড কল করুন")
        
        # প্রেডিকশন
        predictions = self.model.predict(data[features])
        probabilities = self.model.predict_proba(data[features])
        
        # রেজাল্ট ডাটাফ্রেম
        result_df = data[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']].copy()
        result_df['predicted_signal'] = predictions
        result_df['signal_probability'] = probabilities[:, 1]  # ক্লাস 1-এর প্রোবাবিলিটি
        
        # সিগন্যাল ফিল্টার (শুধু বাই সিগন্যাল)
        buy_signals = result_df[result_df['predicted_signal'] == 1].copy()
        
        # রিস্ক ম্যানেজমেন্ট প্যারামিটার যোগ
        buy_signals['position_size'] = 100  # ডিফল্ট
        buy_signals['stop_loss'] = buy_signals['close'] * 0.95  # 5% স্টপ লস
        buy_signals['take_profit'] = buy_signals['close'] * 1.10  # 10% টেক প্রফিট
        buy_signals['risk_reward_ratio'] = (buy_signals['take_profit'] - buy_signals['close']) / (buy_signals['close'] - buy_signals['stop_loss'])
        
        return result_df, buy_signals
    
    def save_model(self, path='./models/xgboost_model.json'):
        """
        মডেল সেভ করে
        """
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        print(f"✅ মডেল সেভ করা হয়েছে: {path}")
    
    def load_model(self, path='./models/xgboost_model.json'):
        """
        মডেল লোড করে
        """
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        print(f"✅ মডেল লোড করা হয়েছে: {path}")

def main():
    """
    মেইন এক্সিকিউশন ফাংশন
    """
    print("=" * 70)
    print("XGBOOST ট্রেডিং মডেল - অ্যাডভান্সড ট্রেনিং")
    print("=" * 70)
    
    # 1. ডাটা লোড
    print("\n📥 ডাটা লোড করা হচ্ছে...")
    
    try:
        market_data = pd.read_csv("./csv/mongodb.csv")
        trade_data = pd.read_csv("./csv/trade_stock.csv")
        
        # তারিখ কনভার্ট
        market_data['date'] = pd.to_datetime(market_data['date'])
        trade_data['date'] = pd.to_datetime(trade_data['date'])
        
    except FileNotFoundError as e:
        print(f"❌ ফাইল পাওয়া যায়নি: {e}")
        return
    except Exception as e:
        print(f"❌ ডাটা লোড করতে সমস্যা: {e}")
        return
    
    # 2. XGBoost মডেল তৈরি এবং ট্রেন
    model = XGBoostTradingModel(
        n_estimators=500,  # কমিয়ে আনা হয়েছে দ্রুত ট্রেনিংয়ের জন্য
        max_depth=4,
        learning_rate=0.05
    )
    
    try:
        accuracy, f1 = model.train(market_data, trade_data)
        
        # 3. মডেল সেভ
        model.save_model('./models/xgboost_trading_model.json')
        
        # 4. প্রেডিকশন তৈরি
        print("\n🔮 নতুন প্রেডিকশন তৈরি করা হচ্ছে...")
        all_predictions, buy_signals = model.predict(market_data, trade_data)
        
        # 5. রেজাল্ট সেভ
        buy_signals.to_csv("./csv/xgboost_predictions.csv", index=False)
        print(f"✅ {len(buy_signals)} টি বাই সিগন্যাল CSV তে সেভ করা হয়েছে: ./csv/xgboost_predictions.csv")
        
        # 6. সামারি ডিসপ্লে
        print(f"\n📈 ট্রেনিং সামারি:")
        print(f"   • ফাইনাল একুরেসি: {accuracy:.2%}")
        print(f"   • F1 স্কোর: {f1:.4f}")
        print(f"   • টপ ফিচার: {model.feature_importance.iloc[0]['feature']}")
        print(f"   • টোটাল সিগন্যাল: {len(buy_signals)}")
        
        if len(buy_signals) > 0:
            print(f"\n🎯 শীর্ষ 3 ট্রেডিং সুযোগ:")
            top_signals = buy_signals.sort_values('signal_probability', ascending=False).head(3)
            for idx, row in top_signals.iterrows():
                print(f"   {row['symbol']} - {row['date'].date()}")
                print(f"     প্রাইস: {row['close']:.2f}, প্রোবাবিলিটি: {row['signal_probability']:.2%}")
                print(f"     R/R রেশিও: {row['risk_reward_ratio']:.2f}")
        
    except Exception as e:
        print(f"❌ ট্রেনিং/প্রেডিকশনে এরর: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
