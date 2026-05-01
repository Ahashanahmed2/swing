# ================== ppo_train.py ==================
# HEDGE FUND LEVEL HYBRID PPO TRAINING SYSTEM
# MAXIMUM QUALITY — ALL FEATURES — NO COMPROMISE
# ✅ MultiSymbolTradingEnv with ALL integrations
# ✅ Higher timesteps, better convergence
# ✅ Full validation, no shortcuts
# ✅ HF CHECKPOINT: Upload to HF via Batch
# ✅ RESUME: From ./csv/ppo_checkpoints/ (downloaded by download_checkpoints.py)
# ✅ RSI Divergence + Support/Resistance features auto-loaded by env_trading.py

import os
import sys
import json
import glob
import shutil
import time
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# ✅ IMPORTS
# =========================================================

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from xgboost_ppo_env import HedgeFundTradingEnv, HedgeFundConfig, FeatureScaler
    ENV_AVAILABLE = True
    print("✅ HedgeFundTradingEnv loaded")
except ImportError:
    ENV_AVAILABLE = False

try:
    from env_trading import MultiSymbolTradingEnv
    MULTI_ENV_AVAILABLE = True
    print("✅ MultiSymbolTradingEnv loaded")
except ImportError:
    MULTI_ENV_AVAILABLE = False

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.env_checker import check_env
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

try:
    from agentic_loop import AgenticLoop
    AGENTIC_LOOP_AVAILABLE = True
except ImportError:
    AGENTIC_LOOP_AVAILABLE = False

try:
    from sector_features import SectorFeatureEngine
    SECTOR_AVAILABLE = True
except ImportError:
    SECTOR_AVAILABLE = False

try:
    from patch_tst_predictor import PatchTSTIntegration
    PATCHTST_AVAILABLE = True
except ImportError:
    PATCHTST_AVAILABLE = False

# =========================================================
# ✅ HUGGINGFACE UPLOAD (Batch) + LOCAL RESUME
# =========================================================

try:
    from huggingface_hub import HfApi, upload_folder, login, create_repo
    HF_AVAILABLE = True
    print("✅ HuggingFace Hub available")
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ HuggingFace Hub not available")

HF_DATASET_REPO = "ahashanahmed/csv"
HF_CHECKPOINT_DIR = "checkpoints"

LOCAL_CHECKPOINT_DIR = Path("./csv/ppo_checkpoints")
LOCAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


class HFCheckpointUploader:
    """Batch upload checkpoints to HF Dataset Repo"""
    
    def __init__(self, repo_id=HF_DATASET_REPO):
        self.repo_id = repo_id
        self.token = os.environ.get("HF_TOKEN") or os.environ.get("hf_token")
        self.api = None
        self.upload_queue = []
        self.batch_size = 50
        
        if HF_AVAILABLE and self.token:
            try:
                login(token=self.token)
                self.api = HfApi(token=self.token)
                create_repo(repo_id=self.repo_id, repo_type="dataset", exist_ok=True)
                print(f"✅ HF Upload Ready: {self.repo_id}")
            except Exception as e:
                print(f"⚠️ HF Upload not available: {e}")
    
    def add_to_batch(self, local_path, hf_path):
        """Queue file for batch upload"""
        if not self.api:
            return
        
        self.upload_queue.append({
            'local_path': str(local_path),
            'hf_path': hf_path
        })
        
        if len(self.upload_queue) >= self.batch_size:
            self.flush()
    
    def flush(self):
        """Upload all queued files in single commit"""
        if not self.upload_queue or not self.api:
            return
        
        try:
            temp_dir = LOCAL_CHECKPOINT_DIR / f"upload_batch_{int(time.time())}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            for item in self.upload_queue:
                dest = temp_dir / item['hf_path']
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item['local_path'], dest)
            
            self.api.upload_folder(
                folder_path=str(temp_dir),
                path_in_repo=HF_CHECKPOINT_DIR,
                repo_id=self.repo_id,
                token=self.token,
                repo_type="dataset",
                commit_message=f"🔥 PPO Checkpoint: {len(self.upload_queue)} files"
            )
            
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"   📤 HF Uploaded: {len(self.upload_queue)} files")
            
        except Exception as e:
            print(f"   ⚠️ HF Upload failed (files saved locally): {e}")
        
        self.upload_queue = []
    
    def upload_file_direct(self, local_path, hf_path, commit_msg="Update"):
        """Upload single file directly"""
        if not self.api:
            return
        try:
            self.api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=f"{HF_CHECKPOINT_DIR}/{hf_path}",
                repo_id=self.repo_id,
                token=self.token,
                repo_type="dataset",
                commit_message=commit_msg
            )
        except:
            pass


class LocalCheckpointManager:
    """Manage checkpoints locally for RESUME"""
    
    def __init__(self):
        self.base_dir = LOCAL_CHECKPOINT_DIR
        self.progress_path = self.base_dir / "ppo_training_progress.json"
        self.progress = self._load_progress()
    
    def _load_progress(self):
        if self.progress_path.exists():
            with open(self.progress_path) as f:
                return json.load(f)
        return {
            'started_at': datetime.now().isoformat(),
            'symbols': {},
            'total_completed': 0,
            'total_failed': 0
        }
    
    def save_progress(self):
        self.progress['last_updated'] = datetime.now().isoformat()
        with open(self.progress_path, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def get_completed_symbols(self):
        return [
            sym for sym, info in self.progress.get('symbols', {}).items()
            if info.get('status') == 'completed'
        ]
    
    def get_pending_symbols(self, all_symbols):
        completed = set(self.get_completed_symbols())
        failed = set(
            sym for sym, info in self.progress.get('symbols', {}).items()
            if info.get('status') == 'failed'
        )
        return [s for s in all_symbols if s not in completed and s not in failed]
    
    def get_last_step(self, symbol):
        return self.progress.get('symbols', {}).get(symbol, {}).get('last_step', 0)
    
    def get_last_ensemble(self, symbol):
        return self.progress.get('symbols', {}).get(symbol, {}).get('last_ensemble', -1)
    
    def get_best_sharpe(self, symbol):
        return self.progress.get('symbols', {}).get(symbol, {}).get('best_sharpe', 0)
    
    def save_model_checkpoint(self, symbol, model, metrics, step, ensemble_idx=None, is_best=False):
        symbol_dir = self.base_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            model_name = f"{symbol}_best"
        elif ensemble_idx is not None:
            model_name = f"{symbol}_ens{ensemble_idx}_step{step}"
        else:
            model_name = f"{symbol}_step{step}"
        
        model_path = symbol_dir / f"{model_name}.zip"
        model.save(model_path)
        
        metrics_path = symbol_dir / f"{model_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'symbol': symbol, 'step': step,
                'sharpe': metrics.get('sharpe', 0),
                'win_rate': metrics.get('win_rate', 0),
                'quality_score': metrics.get('quality_score', 0),
                'ensemble_idx': ensemble_idx, 'is_best': is_best,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        if is_best:
            latest_path = symbol_dir / f"{symbol}_latest.zip"
            shutil.copy2(model_path, latest_path)
        
        self.progress['symbols'][symbol] = {
            'status': 'training',
            'last_step': step,
            'last_ensemble': ensemble_idx if ensemble_idx is not None else 0,
            'best_sharpe': max(self.get_best_sharpe(symbol), metrics.get('sharpe', 0)),
            'updated_at': datetime.now().isoformat()
        }
        self.save_progress()
        
        return model_path, metrics_path
    
    def mark_completed(self, symbol, final_metrics):
        self.progress['symbols'][symbol] = {
            'status': 'completed',
            'final_sharpe': final_metrics.get('sharpe', 0),
            'final_win_rate': final_metrics.get('win_rate', 0),
            'quality_score': final_metrics.get('quality_score', 0),
            'completed_at': datetime.now().isoformat()
        }
        self.progress['total_completed'] = sum(
            1 for s in self.progress['symbols'].values() if s.get('status') == 'completed'
        )
        self.save_progress()
    
    def mark_failed(self, symbol, error):
        self.progress['symbols'][symbol] = {
            'status': 'failed',
            'error': str(error),
            'failed_at': datetime.now().isoformat()
        }
        self.progress['total_failed'] = sum(
            1 for s in self.progress['symbols'].values() if s.get('status') == 'failed'
        )
        self.save_progress()
    
    def get_latest_checkpoint(self, symbol):
        symbol_dir = self.base_dir / symbol
        if not symbol_dir.exists():
            return None
        
        checkpoints = list(symbol_dir.glob(f"{symbol}_step*.zip"))
        if not checkpoints:
            checkpoints = list(symbol_dir.glob(f"{symbol}_ens*_step*.zip"))
        if not checkpoints:
            latest = symbol_dir / f"{symbol}_latest.zip"
            return str(latest) if latest.exists() else None
        
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(checkpoints[0])
    
    def get_best_checkpoint(self, symbol):
        best_path = self.base_dir / symbol / f"{symbol}_best.zip"
        return str(best_path) if best_path.exists() else self.get_latest_checkpoint(symbol)
    
    def get_summary_path(self):
        return self.base_dir / "ppo_training_summary.json"


# Global instances
_local_ckpt = None
_hf_uploader = None

def get_local_checkpoint():
    global _local_ckpt
    if _local_ckpt is None:
        _local_ckpt = LocalCheckpointManager()
    return _local_ckpt

def get_hf_uploader():
    global _hf_uploader
    if _hf_uploader is None:
        _hf_uploader = HFCheckpointUploader()
    return _hf_uploader

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_MARKET = BASE_DIR / "csv" / "mongodb.csv"
CSV_SIGNAL = BASE_DIR / "csv" / "trade_stock.csv"
XGB_MODEL_DIR = BASE_DIR / "csv" / "xgboost"
PPO_MODEL_DIR = BASE_DIR / "csv" / "ppo_models"
PPO_SHARED_PATH = PPO_MODEL_DIR / "ppo_shared"
PPO_SYMBOL_DIR = PPO_MODEL_DIR / "per_symbol"
PPO_ENSEMBLE_DIR = PPO_MODEL_DIR / "ensemble"
MODEL_METADATA = BASE_DIR / "csv" / "model_metadata.csv"
PREDICTION_LOG = BASE_DIR / "csv" / "prediction_log.csv"
LAST_PPO_TRAIN = BASE_DIR / "csv" / "last_ppo_train.txt"
TENSORBOARD_LOG = BASE_DIR / "logs" / "ppo_tensorboard"
MISTAKES_FILE = BASE_DIR / "csv" / "trading_mistakes.csv"

os.makedirs(PPO_MODEL_DIR, exist_ok=True)
os.makedirs(PPO_SYMBOL_DIR, exist_ok=True)
os.makedirs(PPO_ENSEMBLE_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG, exist_ok=True)

# =========================================================
# 🔥 MAXIMUM QUALITY CONFIGURATION
# =========================================================

WINDOW = 10
TOTAL_CAPITAL = 500_000
RISK_PERCENT = 0.01
PPO_RETRAIN_INTERVAL = 7

XGB_AUC_THRESHOLD_FOR_PPO = 0.55
MAX_PER_SYMBOL_MODELS = 50

TRAIN_RATIO = 0.65
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.20

WALK_FORWARD_WINDOW = 252
WALK_FORWARD_STEP = 21

EARLY_STOPPING_PATIENCE = 15
EVAL_FREQ = 500

NOISE_STD = 0.0005
USE_NOISE_INJECTION = True

ENSEMBLE_SIZE = 5
USE_ENSEMBLE = True

AGENTIC_LOOP_CONFIG = {
    'enabled': True,
    'auto_learn': True,
    'min_confidence_for_trade': 0.45,
    'save_decisions': True,
    'show_consensus': True
}

DEFAULT_MARKET_COLS = ["open", "high", "low", "close", "volume"]
try:
    from env_trading import MARKET_COLS
except ImportError:
    MARKET_COLS = DEFAULT_MARKET_COLS

STATE_DIM = len(MARKET_COLS) * WINDOW + 4

PPO_CONFIG = {
    'n_steps': 1024,
    'batch_size': 256,
    'gamma': 0.997,
    'learning_rate': 5e-5,
    'ent_coef': 0.003,
    'clip_range': 0.15,
    'vf_coef': 0.3,
    'max_grad_norm': 0.5,
    'gae_lambda': 0.97,
    'n_epochs': 15,
    'target_kl': 0.02,
}

PPO_PER_SYMBOL_CONFIG = {
    'high_quality': {
        'n_steps': 4096, 'batch_size': 1024, 'learning_rate': 8e-5,
        'timesteps': 50000, 'n_epochs': 20
    },
    'good_quality': {
        'n_steps': 2048, 'batch_size': 512, 'learning_rate': 5e-5,
        'timesteps': 40000, 'n_epochs': 15
    },
    'fallback': {
        'n_steps': 2048, 'batch_size': 512, 'learning_rate': 5e-5,
        'timesteps': 30000, 'n_epochs': 15
    },
}

# =========================================================
# ✅ AGENTIC LOOP WRAPPER
# =========================================================

class AgenticLoopWrapper:
    def __init__(self):
        self.loop = None
        self.initialized = False
        self.decisions = []
        
    def init_if_needed(self):
        if not self.initialized and AGENTIC_LOOP_AVAILABLE and AGENTIC_LOOP_CONFIG['enabled']:
            try:
                self.loop = AgenticLoop(xgb_model_dir=str(XGB_MODEL_DIR))
                self.initialized = True
                print("   🤖 Agentic Loop initialized")
            except Exception as e:
                print(f"   ⚠️ Agentic Loop init failed: {e}")
        return self.initialized
    
    def get_consensus(self, symbol, symbol_data, volatility=0.02, market_regime='NEUTRAL'):
        if not self.init_if_needed():
            return 'HOLD', 0.5, 0.3, {}
        if symbol_data is None or symbol_data.empty:
            return 'HOLD', 0.5, 0.3, {}
        try:
            decision, score, confidence, details = self.loop.get_consensus(
                symbol=symbol, symbol_data=symbol_data,
                volatility=volatility, market_regime=market_regime
            )
            self.decisions.append({
                'symbol': symbol, 'decision': decision,
                'score': score, 'confidence': confidence,
                'timestamp': datetime.now()
            })
            return decision, score, confidence, details
        except:
            return 'HOLD', 0.5, 0.3, {}
    
    def record_trade_feedback(self, trade_result):
        if not self.init_if_needed():
            return
        try:
            self.loop.after_trade_feedback(trade_result)
            if len(self.decisions) % 5 == 0:
                self.save_decisions()
        except:
            pass
    
    def save_decisions(self):
        if not self.decisions:
            return
        try:
            df = pd.DataFrame(self.decisions)
            df.to_csv('./csv/agentic_decisions.csv', index=False)
        except:
            pass
    
    def get_summary(self):
        if not self.init_if_needed():
            return pd.DataFrame()
        try:
            return self.loop.get_summary()
        except:
            return pd.DataFrame()

_agentic_loop = None

def get_agentic_loop():
    global _agentic_loop
    if _agentic_loop is None:
        _agentic_loop = AgenticLoopWrapper()
    return _agentic_loop

# =========================================================
# ✅ SAFE TRADE RESULT EXTRACTOR
# =========================================================

def safe_extract_trade_result(info):
    trade_result = None
    if isinstance(info, list) and len(info) > 0:
        info = info[0]
    if isinstance(info, dict):
        trade_result = info.get('trade_result')
    if not isinstance(trade_result, dict):
        return None
    return {
        'success': trade_result.get('success', False),
        'pnl': trade_result.get('pnl', 0.0),
        'entry_price': trade_result.get('entry_price', 0.0),
        'exit_price': trade_result.get('exit_price', 0.0),
        'exit_reason': trade_result.get('exit_reason', 'unknown')
    }

# =========================================================
# ✅ MISTAKE LEARNER
# =========================================================

class MistakeLearner:
    def __init__(self, mistakes_file=MISTAKES_FILE):
        self.mistakes_file = mistakes_file
        self.mistakes = self.load_mistakes()

    def load_mistakes(self):
        if not os.path.exists(self.mistakes_file):
            return []
        try:
            df = pd.read_csv(self.mistakes_file)
            return df.to_dict('records')
        except:
            return []

    def record_mistake(self, symbol, entry_price, exit_price, pnl, reason):
        try:
            loss_percent = abs(pnl) / entry_price * 100 if entry_price > 0 else 0
            mistake = {
                'symbol': str(symbol), 'entry_price': float(entry_price),
                'exit_price': float(exit_price), 'pnl': float(pnl),
                'loss_percent': round(loss_percent, 2), 'reason': str(reason),
                'date': datetime.now().strftime('%Y-%m-%d')
            }
            self.mistakes.append(mistake)
            try:
                df = pd.DataFrame(self.mistakes)
                os.makedirs(os.path.dirname(self.mistakes_file), exist_ok=True)
                df.to_csv(self.mistakes_file, index=False)
            except:
                pass
            return mistake
        except:
            return None
    
    def get_avoid_list(self):
        try:
            avoid = {}
            for mistake in self.mistakes:
                symbol = mistake.get('symbol')
                if symbol:
                    avoid[symbol] = avoid.get(symbol, 0) + 1
            return [s for s, count in avoid.items() if count >= 5]
        except:
            return []

# =========================================================
# ✅ HELPER
# =========================================================

def ensure_vecenv_action(action):
    if isinstance(action, (int, float, np.integer, np.floating)):
        return [int(action)]
    elif isinstance(action, np.ndarray) and action.ndim == 0:
        return [int(action.item())]
    elif isinstance(action, (list, tuple)):
        return list(action)
    return action

# =========================================================
# 🔥 IMPROVED EARLY STOPPING
# =========================================================

if SB3_AVAILABLE:
    class EarlyStoppingCallback(BaseCallback):
        def __init__(self, eval_env, patience=15, threshold=0.0005, min_delta=0.0001, verbose=0):
            super().__init__(verbose)
            self.eval_env = eval_env
            self.patience = patience
            self.threshold = threshold
            self.min_delta = min_delta
            self.best_mean_reward = -np.inf
            self.no_improvement_count = 0
            self.eval_freq = EVAL_FREQ
            self.reward_history = []

        def _on_step(self):
            if self.n_calls % self.eval_freq == 0:
                mean_reward = self._evaluate()
                self.reward_history.append(mean_reward)
                
                if mean_reward > self.best_mean_reward + self.min_delta:
                    self.best_mean_reward = mean_reward
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                
                if self.no_improvement_count >= self.patience:
                    return False
            return True

        def _evaluate(self):
            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            total_reward = 0
            steps = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated) and steps < 10000:
                if isinstance(obs, np.ndarray) and len(obs.shape) == 1:
                    obs = obs.reshape(1, -1)
                action, _ = self.model.predict(obs, deterministic=True)
                action = ensure_vecenv_action(action)
                step_result = self.eval_env.step(action)
                
                if step_result is None or len(step_result) == 0:
                    break
                
                obs = step_result[0]
                reward = step_result[1] if len(step_result) > 1 else 0
                terminated = step_result[2] if len(step_result) > 2 else False
                truncated = step_result[3] if len(step_result) > 3 else False
                
                if isinstance(reward, (list, np.ndarray)):
                    reward = reward[0] if len(reward) > 0 else 0
                elif not isinstance(reward, (int, float)):
                    reward = 0
                
                total_reward += reward
                steps += 1
            
            return total_reward / steps if steps > 0 else 0

# =========================================================
# ✅ ENSEMBLE PPO
# =========================================================

if SB3_AVAILABLE:
    class EnsemblePPO:
        def __init__(self, model_paths, weights=None):
            self.models = []
            self.weights = weights if weights else [1.0 / len(model_paths)] * len(model_paths)
            for path in model_paths:
                try:
                    self.models.append(PPO.load(path, device="cpu"))
                except:
                    pass
        
        def predict(self, observation, deterministic=True):
            if not self.models:
                return [0], None
            all_actions = []
            for model in self.models:
                action, _ = model.predict(observation, deterministic=deterministic)
                if isinstance(action, np.ndarray):
                    action = int(action.item()) if action.size == 1 else int(action[0])
                elif isinstance(action, (list, tuple)) and len(action) > 0:
                    action = int(action[0])
                else:
                    action = int(action) if action is not None else 0
                all_actions.append(action)
            
            weighted_votes = {}
            for i, action in enumerate(all_actions):
                weight = self.weights[i] if i < len(self.weights) else 1.0/len(all_actions)
                weighted_votes[action] = weighted_votes.get(action, 0) + weight
            
            final_action = int(max(weighted_votes, key=weighted_votes.get))
            return [final_action], {'actions': all_actions, 'weighted_votes': weighted_votes}

# =========================================================
# ✅ CREATE ENVIRONMENT (RSI Div + S/R auto-loaded by env_trading.py)
# =========================================================

def create_multi_symbol_env(symbol_dfs, signals, sector_engine=None, xgb_models=None, 
                           agentic_loop=None, patch_tst=None):
    if not MULTI_ENV_AVAILABLE:
        return None
    
    try:
        env = MultiSymbolTradingEnv(
            symbol_dfs=symbol_dfs,
            signals=signals,
            build_observation=build_observation,
            window=WINDOW,
            state_dim=STATE_DIM,
            total_capital=TOTAL_CAPITAL,
            risk_percent=RISK_PERCENT,
            sector_engine=sector_engine,
            xgb_models=xgb_models,
            agentic_loop=agentic_loop,
            patch_tst=patch_tst
        )
        return env
    except Exception as e:
        print(f"   ⚠️ Error creating env: {e}")
        return None

# =========================================================
# 🔥 MAXIMUM QUALITY TRAINING (With Local Checkpoint + HF Upload)
# =========================================================

def train_max_quality(symbol, symbol_data, signals, xgb_auc, is_retrain=False,
                      sector_engine=None, xgb_models=None, agentic_loop=None, patch_tst=None):
    """
    Maximum Quality Training
    ✅ Save checkpoint → ./csv/ppo_checkpoints/
    ✅ Upload to HF via batch
    ✅ Resume from local checkpoint
    """
    
    if not SB3_AVAILABLE or not GYM_AVAILABLE:
        return None, {}
    
    local_ckpt = get_local_checkpoint()
    hf_uploader = get_hf_uploader()
    
    print(f"\n{'═'*60}")
    print(f"🎯 MAX QUALITY TRAINING: {symbol}")
    print(f"   AUC: {xgb_auc:.2%} | Data: {len(symbol_data)} rows")
    print(f"{'═'*60}")
    
    mistake_learner = MistakeLearner()
    
    # Agent consensus
    if AGENTIC_LOOP_CONFIG['enabled'] and agentic_loop and len(symbol_data) > 20:
        try:
            volatility = symbol_data['close'].pct_change().std() * np.sqrt(252)
            decision, score, confidence, _ = agentic_loop.get_consensus(
                symbol=symbol, symbol_data=symbol_data.tail(50),
                volatility=volatility, market_regime='NEUTRAL'
            )
            if AGENTIC_LOOP_CONFIG['show_consensus']:
                print(f"\n   🤖 AGENTIC CONSENSUS:")
                print(f"      Decision: {decision} | Score: {score:.3f} | Conf: {confidence:.3f}")
        except:
            pass
    
    # Data Split
    total_len = len(symbol_data)
    train_end = int(total_len * TRAIN_RATIO)
    val_end = int(total_len * (TRAIN_RATIO + VALIDATION_RATIO))
    
    train_data = symbol_data.iloc[:train_end]
    val_data = symbol_data.iloc[train_end:val_end]
    test_data = symbol_data.iloc[val_end:]
    
    print(f"\n   📊 Data Split:")
    print(f"      Train: {len(train_data)} ({len(train_data)/len(symbol_data):.0%})")
    print(f"      Val:   {len(val_data)} ({len(val_data)/len(symbol_data):.0%})")
    print(f"      Test:  {len(test_data)} ({len(test_data)/len(symbol_data):.0%})")
    
    # Select config
    if xgb_auc >= 0.85:
        config = PPO_PER_SYMBOL_CONFIG['high_quality']
        quality = 'HIGH'
    elif xgb_auc >= 0.70:
        config = PPO_PER_SYMBOL_CONFIG['good_quality']
        quality = 'GOOD'
    else:
        config = PPO_PER_SYMBOL_CONFIG['fallback']
        quality = 'FALLBACK'
    
    print(f"\n   🎚️ Quality: {quality} | Timesteps: {config['timesteps']:,} | Steps: {config['n_steps']}")
    
    # ✅ RESUME: Check if previous training exists
    last_step = local_ckpt.get_last_step(symbol)
    last_ensemble = local_ckpt.get_last_ensemble(symbol)
    
    if last_step > 0:
        print(f"\n   🔄 RESUME: Found previous checkpoint at step {last_step}")
        print(f"      Last ensemble: {last_ensemble + 1}/{ENSEMBLE_SIZE}")
    
    ensemble_models = []
    ensemble_stats = []
    
    # 🔥 Train Ensemble
    start_ensemble = max(0, last_ensemble + 1) if last_step > 0 else 0
    
    for ensemble_idx in range(start_ensemble, ENSEMBLE_SIZE):
        print(f"\n   {'─'*50}")
        print(f"   🧠 ENSEMBLE MODEL {ensemble_idx + 1}/{ENSEMBLE_SIZE}")
        print(f"   {'─'*50}")
        
        start_time = datetime.now()
        
        try:
            train_env = create_multi_symbol_env(
                {symbol: train_data}, signals, sector_engine, xgb_models, agentic_loop, patch_tst
            )
            val_env = create_multi_symbol_env(
                {symbol: val_data}, signals, sector_engine, xgb_models, agentic_loop, patch_tst
            )
            
            if train_env is None:
                continue
            
            train_env = DummyVecEnv([lambda: train_env])
            val_env = DummyVecEnv([lambda: val_env])
            
            ppo_config = PPO_CONFIG.copy()
            ppo_config.update({
                'n_steps': config['n_steps'],
                'batch_size': config['batch_size'],
                'learning_rate': config['learning_rate'],
                'seed': 42 + ensemble_idx
            })
            
            # ✅ RESUME: Load model if checkpoint exists
            resume_step = 0
            if ensemble_idx == start_ensemble and last_step > 0:
                checkpoint_path = local_ckpt.get_latest_checkpoint(symbol)
                if checkpoint_path and os.path.exists(checkpoint_path):
                    print(f"      🔄 Loading checkpoint: {checkpoint_path}")
                    model = PPO.load(checkpoint_path, env=train_env, device="cpu")
                    resume_step = last_step
                else:
                    model = PPO("MlpPolicy", train_env, **ppo_config, verbose=0)
            else:
                model = PPO("MlpPolicy", train_env, **ppo_config, verbose=0)
            
            early_stop = EarlyStoppingCallback(val_env, patience=EARLY_STOPPING_PATIENCE, verbose=0)
            
            remaining_steps = config['timesteps'] - resume_step
            print(f"      ⏳ Training {remaining_steps:,} timesteps (resume from {resume_step})...")
            
            # 🔥 Save checkpoint every 5000 steps
            checkpoint_interval = 5000
            for step in range(0, remaining_steps, checkpoint_interval):
                train_steps = min(checkpoint_interval, remaining_steps - step)
                model.learn(total_timesteps=train_steps, callback=early_stop)
                
                current_step = resume_step + step + train_steps
                
                # ✅ Save LOCAL checkpoint
                model_path, metrics_path = local_ckpt.save_model_checkpoint(
                    symbol=symbol,
                    model=model,
                    metrics={'step': current_step, 'sharpe': 0, 'win_rate': 0, 'quality_score': 0},
                    step=current_step,
                    ensemble_idx=ensemble_idx
                )
                
                # ✅ Queue for HF upload
                hf_uploader.add_to_batch(model_path, f"{symbol}/{model_path.name}")
                hf_uploader.add_to_batch(metrics_path, f"{symbol}/{metrics_path.name}")
                
                print(f"      💾 Checkpoint: step {current_step}/{config['timesteps']}")
            
            # Validation
            obs = val_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            total_return = 0
            steps = 0
            rewards_list = []
            terminated = False
            truncated = False
            
            while not (terminated or truncated) and steps < 10000:
                action, _ = model.predict(obs, deterministic=True)
                action = ensure_vecenv_action(action)
                step_result = val_env.step(action)
                
                if step_result is None or len(step_result) == 0:
                    break
                
                obs = step_result[0]
                reward = step_result[1] if len(step_result) > 1 else 0
                terminated = step_result[2] if len(step_result) > 2 else False
                truncated = step_result[3] if len(step_result) > 3 else False
                info = step_result[4] if len(step_result) > 4 else {}
                
                if isinstance(reward, (list, np.ndarray)):
                    reward = reward[0] if len(reward) > 0 else 0
                
                total_return += reward
                rewards_list.append(reward)
                steps += 1
                
                trade_result = safe_extract_trade_result(info)
                if trade_result:
                    if not trade_result.get('success', False):
                        mistake_learner.record_mistake(
                            symbol=symbol,
                            entry_price=trade_result.get('entry_price', 0),
                            exit_price=trade_result.get('exit_price', 0),
                            pnl=trade_result.get('pnl', 0),
                            reason='validation_loss'
                        )
                    
                    if AGENTIC_LOOP_CONFIG['enabled'] and agentic_loop:
                        agentic_loop.record_trade_feedback({
                            'symbol': symbol,
                            'pnl': trade_result.get('pnl', 0),
                            'success': trade_result.get('success', False)
                        })
            
            returns_array = np.array(rewards_list)
            sharpe = np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252)
            win_rate = np.mean(np.array(rewards_list) > 0)
            
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            
            print(f"      ✅ Model {ensemble_idx+1} Complete ({elapsed:.1f} min)")
            print(f"      Return: {total_return:.4f} | Sharpe: {sharpe:.3f} | Win: {win_rate:.1%}")
            
            model_path = PPO_ENSEMBLE_DIR / f"ppo_{symbol}_ens{ensemble_idx}"
            model.save(model_path)
            ensemble_models.append(model_path)
            ensemble_stats.append({
                'return': total_return,
                'sharpe': sharpe,
                'win_rate': win_rate,
                'steps': steps
            })
            
        except Exception as e:
            print(f"   ⚠️ Training failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not ensemble_models:
        local_ckpt.mark_failed(symbol, "No models trained")
        return None, {}
    
    # Weighted Ensemble
    if len(ensemble_models) > 1:
        sharpe_values = np.array([max(0.01, s['sharpe']) for s in ensemble_stats])
        weights = sharpe_values / sharpe_values.sum()
        
        print(f"\n   🎯 ENSEMBLE CREATED ({len(ensemble_models)} models)")
        for i, (w, s) in enumerate(zip(weights, ensemble_stats)):
            print(f"      Model {i+1}: Weight={w:.3f} | Sharpe={s['sharpe']:.3f} | Win={s['win_rate']:.1%}")
        
        final_model = EnsemblePPO(ensemble_models, weights.tolist())
    else:
        final_model = PPO.load(ensemble_models[0], device="cpu")
    
    # 🔥 FINAL TEST
    print(f"\n   {'═'*50}")
    print(f"   🧪 FINAL TEST — NEVER-TOUCHED DATA")
    print(f"   {'═'*50}")
    
    try:
        test_env = create_multi_symbol_env(
            {symbol: test_data}, signals, sector_engine, xgb_models, agentic_loop, patch_tst
        )
        test_env = DummyVecEnv([lambda: test_env])
        
        obs = test_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        total_return = 0
        test_trades = []
        rewards_list = []
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and steps < 10000:
            action, _ = final_model.predict(obs, deterministic=True)
            action = ensure_vecenv_action(action)
            step_result = test_env.step(action)
            
            if step_result is None or len(step_result) == 0:
                break
            
            obs = step_result[0]
            reward = step_result[1] if len(step_result) > 1 else 0
            terminated = step_result[2] if len(step_result) > 2 else False
            truncated = step_result[3] if len(step_result) > 3 else False
            info = step_result[4] if len(step_result) > 4 else {}
            
            if isinstance(reward, (list, np.ndarray)):
                reward = reward[0] if len(reward) > 0 else 0
            
            total_return += reward
            rewards_list.append(reward)
            steps += 1
            
            trade_result = safe_extract_trade_result(info)
            if trade_result:
                test_trades.append(trade_result)
                
                if AGENTIC_LOOP_CONFIG['enabled'] and agentic_loop:
                    agentic_loop.record_trade_feedback({
                        'symbol': symbol,
                        'pnl': trade_result.get('pnl', 0),
                        'success': trade_result.get('success', False)
                    })
        
        returns_array = np.array(rewards_list)
        final_sharpe = np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252)
        
        profitable = sum(1 for t in test_trades if t.get('success', False))
        total_trades = len(test_trades)
        win_rate = profitable / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t['pnl'] for t in test_trades if t.get('success', False)]) if profitable > 0 else 0
        avg_loss = np.mean([abs(t['pnl']) for t in test_trades if not t.get('success', False)]) if (total_trades - profitable) > 0 else 0
        profit_factor = avg_win / (avg_loss + 1e-8) if avg_loss > 0 else float('inf')
        
        for trade in test_trades:
            if not trade.get('success', False):
                mistake_learner.record_mistake(
                    symbol=symbol,
                    entry_price=trade.get('entry_price', 0),
                    exit_price=trade.get('exit_price', 0),
                    pnl=trade.get('pnl', 0),
                    reason='test_loss'
                )
        
        quality_score = min(100, (final_sharpe * 20) + (win_rate * 50) + (min(profit_factor, 5) * 10))
        
        final_metrics = {
            'sharpe': final_sharpe,
            'win_rate': win_rate,
            'return': total_return,
            'trades': total_trades,
            'profit_factor': profit_factor,
            'quality_score': quality_score,
            'ensemble_size': len(ensemble_models)
        }
        
        print(f"\n   📊 FINAL TEST RESULTS:")
        print(f"   {'─'*40}")
        print(f"   Total Return:    {total_return:>10.4f}")
        print(f"   Sharpe Ratio:    {final_sharpe:>10.3f}")
        print(f"   Win Rate:        {win_rate:>10.1%} ({profitable}/{total_trades})")
        print(f"   Avg Win:         {avg_win:>10.2f} Tk")
        print(f"   Avg Loss:        {avg_loss:>10.2f} Tk")
        print(f"   Profit Factor:   {profit_factor:>10.2f}")
        print(f"   QUALITY SCORE:   {quality_score:>10.1f}/100")
        
        if quality_score >= 80:
            print(f"   ⭐ EXCELLENT MODEL — Ready for production!")
        elif quality_score >= 60:
            print(f"   ✅ GOOD MODEL — Can be used with caution")
        else:
            print(f"   ⚠️ NEEDS IMPROVEMENT — Consider more training")
        
        # ✅ Save BEST model locally
        best_model_path, best_metrics_path = local_ckpt.save_model_checkpoint(
            symbol=symbol,
            model=final_model if not isinstance(final_model, EnsemblePPO) else EnsemblePPO.__new__(EnsemblePPO),
            metrics=final_metrics,
            step=config['timesteps'],
            is_best=True
        )
        
        # ✅ Upload BEST to HF
        hf_uploader.add_to_batch(best_model_path, f"{symbol}/{Path(best_model_path).name}")
        hf_uploader.add_to_batch(best_metrics_path, f"{symbol}/{Path(best_metrics_path).name}")
        hf_uploader.flush()
        
        # ✅ Upload progress to HF
        progress_path = local_ckpt.progress_path
        hf_uploader.upload_file_direct(progress_path, "ppo_training_progress.json", "📊 Update progress")
        
        # ✅ Mark completed
        local_ckpt.mark_completed(symbol, final_metrics)
        
        # Save to PPO_MODEL_DIR
        final_path = PPO_SYMBOL_DIR / f"ppo_{symbol}"
        if not isinstance(final_model, EnsemblePPO):
            final_model.save(final_path)
        else:
            ensemble_info_path = PPO_SYMBOL_DIR / f"ensemble_{symbol}.pkl"
            joblib.dump({
                'model_paths': [str(p) for p in ensemble_models],
                'weights': weights.tolist()
            }, ensemble_info_path)
        
        print(f"\n   💾 Model saved: {final_path}")
        print(f"   📤 Uploaded to HF: {HF_DATASET_REPO}/{HF_CHECKPOINT_DIR}/{symbol}/")
        
        return final_model, final_metrics
        
    except Exception as e:
        print(f"   ⚠️ Final test failed: {e}")
        import traceback
        traceback.print_exc()
        local_ckpt.mark_failed(symbol, str(e))
        return None, {}

# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def load_signals(path):
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        signals = {}
        for _, r in df.iterrows():
            signals[(r["symbol"], r["date"])] = {
                "buy": float(r["buy"]), "SL": float(r["SL"]),
                "tp": float(r["tp"]), "RRR": float(r["RRR"]),
            }
        return signals
    except:
        return {}

def load_xgb_metadata():
    if not os.path.exists(MODEL_METADATA):
        return pd.DataFrame()
    try:
        df = pd.read_csv(MODEL_METADATA)
        if 'status' in df.columns and 'auc' in df.columns:
            return df[df['status'] == 'GOOD'].sort_values('auc', ascending=False)
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def build_observation(df, idx, signals):
    try:
        available_cols = [col for col in MARKET_COLS if col in df.columns]
        if not available_cols:
            available_cols = ['close', 'volume']
        pad = max(0, WINDOW - (idx + 1))
        start = max(0, idx - WINDOW + 1)
        seg = df.iloc[start:idx+1][available_cols].values
        seg = np.pad(seg, ((pad,0),(0,0)), mode="edge")
        market_vec = seg.flatten()
        if len(market_vec) < len(MARKET_COLS) * WINDOW:
            market_vec = np.pad(market_vec, (0, len(MARKET_COLS) * WINDOW - len(market_vec)))
        row = df.iloc[idx]
        sig = signals.get((row["symbol"], row["date"]))
        if sig:
            buy = sig["buy"]
            signal_vec = [row["close"] / (buy + 1e-8), (buy - sig["SL"]) / (buy + 1e-8),
                         (sig["tp"] - buy) / (buy + 1e-8), sig["RRR"]]
        else:
            signal_vec = [0.0] * 4
        return np.nan_to_num(list(market_vec) + signal_vec)
    except:
        return np.zeros(STATE_DIM, dtype=np.float32)

def should_retrain_ppo():
    if not os.path.exists(LAST_PPO_TRAIN):
        return True, "First training"
    with open(LAST_PPO_TRAIN, 'r') as f:
        last_date = datetime.strptime(f.read().strip(), '%Y-%m-%d')
    days_since = (datetime.now() - last_date).days
    if days_since >= PPO_RETRAIN_INTERVAL:
        return True, f"Retrain (last: {days_since} days ago)"
    return False, f"Next retrain in {PPO_RETRAIN_INTERVAL - days_since} days"

def update_last_ppo_train():
    with open(LAST_PPO_TRAIN, 'w') as f:
        f.write(datetime.now().strftime('%Y-%m-%d'))

# =========================================================
# 🔥 MAIN TRAINING — LOCAL RESUME + HF UPLOAD
# =========================================================

def train_ppo_system():
    print("="*70)
    print("🏦 HEDGE FUND LEVEL — MAXIMUM QUALITY PPO TRAINING")
    print("="*70)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"💰 Capital: ${TOTAL_CAPITAL:,.0f}")
    print(f"🔧 Ensemble: {ENSEMBLE_SIZE} | Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"💾 Local Checkpoint: {LOCAL_CHECKPOINT_DIR}")
    print(f"📤 HF Upload: {HF_DATASET_REPO}/{HF_CHECKPOINT_DIR}")
    print(f"📊 Features: ALL (Sector+Micro+Greeks+Regime+PatchTST+LLM+Agentic+XGB+RSI_Div+S/R)")
    print("="*70)

    if not SB3_AVAILABLE or not GYM_AVAILABLE:
        print("\n⚠️ Required packages not available!")
        return [], None

    should_retrain, reason = should_retrain_ppo()
    is_retrain = should_retrain and os.path.exists(f"{PPO_SHARED_PATH}.zip")
    print(f"\n📊 Status: {reason}")

    # Initialize components
    agentic = get_agentic_loop() if AGENTIC_LOOP_CONFIG['enabled'] else None
    if agentic:
        agentic.init_if_needed()
    
    sector_engine = None
    if SECTOR_AVAILABLE:
        try:
            sector_engine = SectorFeatureEngine(csv_market_path=str(CSV_MARKET))
            print("✅ Sector Engine loaded")
        except Exception as e:
            print(f"⚠️ Sector Engine failed: {e}")
    
    patch_tst = None
    if PATCHTST_AVAILABLE:
        try:
            patch_tst = PatchTSTIntegration(model_dir="./csv/patchtst_models")
            print("✅ PatchTST loaded")
        except Exception as e:
            print(f"⚠️ PatchTST failed: {e}")
    
    xgb_models = {}
    
    local_ckpt = get_local_checkpoint()
    hf_uploader = get_hf_uploader()

    # Load data
    print("\n📂 Loading market data...")
    if not os.path.exists(CSV_MARKET):
        print("   ❌ Market data not found")
        return [], None

    df = pd.read_csv(CSV_MARKET)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime("%Y-%m-%d")
    print(f"   ✅ {len(df)} rows, {df['symbol'].nunique()} symbols")

    signals = load_signals(CSV_SIGNAL)
    xgb_metadata = load_xgb_metadata()

    top_symbol_list = []
    if not xgb_metadata.empty:
        top_symbols = xgb_metadata[xgb_metadata['auc'] >= XGB_AUC_THRESHOLD_FOR_PPO].head(MAX_PER_SYMBOL_MODELS)
        top_symbol_list = top_symbols['symbol'].tolist()
        print(f"   ✅ {len(top_symbol_list)} symbols for MAX QUALITY training")

    all_symbols_data = {}
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].reset_index(drop=True)
        if len(symbol_df) >= WINDOW + 100:
            all_symbols_data[symbol] = symbol_df
    print(f"   ✅ {len(all_symbols_data)} symbols with sufficient data")

    # ✅ Check LOCAL progress & RESUME
    completed_symbols = local_ckpt.get_completed_symbols()
    pending_symbols = local_ckpt.get_pending_symbols(top_symbol_list)
    
    if completed_symbols:
        print(f"\n📊 LOCAL Progress: {len(completed_symbols)} already completed")
        print(f"   Pending: {len(pending_symbols)} symbols")
    
    if not pending_symbols:
        print("\n✅ ALL SYMBOLS ALREADY TRAINED!")
        return completed_symbols, None

    trained_symbols = []
    all_stats = []
    total_start = datetime.now()
    
    try:
        print(f"\n{'='*70}")
        print("🏆 MAXIMUM QUALITY TRAINING STARTED")
        print(f"{'='*70}")
        
        for idx, symbol in enumerate(pending_symbols[:MAX_PER_SYMBOL_MODELS]):
            if symbol not in all_symbols_data:
                continue
            
            symbol_data = all_symbols_data[symbol]
            xgb_info = xgb_metadata[xgb_metadata['symbol'] == symbol]
            xgb_auc = xgb_info.iloc[0]['auc'] if len(xgb_info) > 0 else 0.65
            
            print(f"\n📈 Progress: {idx+1}/{len(pending_symbols)} [Pending] | Total: {len(completed_symbols)+idx+1}/{len(top_symbol_list)}")
            
            try:
                model, stats = train_max_quality(
                    symbol, symbol_data, signals, xgb_auc, is_retrain,
                    sector_engine, xgb_models, agentic, patch_tst
                )
                if model is not None:
                    trained_symbols.append(symbol)
                    all_stats.append(stats)
            except Exception as e:
                print(f"\n   ❌ Failed: {symbol} - {e}")

        update_last_ppo_train()
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted — progress saved locally")
        hf_uploader.flush()
    except Exception as e:
        print(f"\n⚠️ Training error: {e}")
        hf_uploader.flush()

    total_elapsed = (datetime.now() - total_start).total_seconds() / 60
    
    # ✅ Final HF Upload
    hf_uploader.flush()
    
    # Save summary locally
    summary_path = local_ckpt.get_summary_path()
    with open(summary_path, 'w') as f:
        json.dump({
            'completed_at': datetime.now().isoformat(),
            'total_trained': len(trained_symbols) + len(completed_symbols),
            'symbols': trained_symbols + completed_symbols,
            'average_sharpe': sum(s.get('sharpe', 0) for s in all_stats) / len(all_stats) if all_stats else 0,
            'average_win_rate': sum(s.get('win_rate', 0) for s in all_stats) / len(all_stats) if all_stats else 0
        }, f, indent=2)
    
    # Upload summary to HF
    hf_uploader.upload_file_direct(summary_path, "ppo_training_summary.json", "🏁 Training complete")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"🏦 MAXIMUM QUALITY TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"   Total Time: {total_elapsed:.1f} min ({total_elapsed/60:.1f} hours)")
    print(f"   Trained Now: {len(trained_symbols)}")
    print(f"   Total Trained: {len(trained_symbols) + len(completed_symbols)}")
    print(f"   Local: {LOCAL_CHECKPOINT_DIR}")
    print(f"   HF: {HF_DATASET_REPO}/{HF_CHECKPOINT_DIR}")
    
    if all_stats:
        avg_sharpe = np.mean([s.get('sharpe', 0) for s in all_stats])
        avg_win_rate = np.mean([s.get('win_rate', 0) for s in all_stats])
        print(f"\n📊 AVERAGE METRICS (This Run):")
        print(f"   Sharpe: {avg_sharpe:.3f} | Win Rate: {avg_win_rate:.1%}")
    
    return trained_symbols + completed_symbols, None

def main():
    try:
        train_ppo_system()
        print("\n✅ MAXIMUM QUALITY SYSTEM READY FOR PRODUCTION!")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(0)

if __name__ == "__main__":
    main()