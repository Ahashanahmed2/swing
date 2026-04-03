# check_obs_size.py - আপনার project এ রান করুন

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from xgboost_ppo_env import HedgeFundTradingEnv, HedgeFundConfig, MARKET_COLS
    
    print("="*50)
    print("OBSERVATION SIZE CHECK")
    print("="*50)
    
    # Check MARKET_COLS
    print(f"\n1. MARKET_COLS length: {len(MARKET_COLS)}")
    print(f"   Columns: {MARKET_COLS}")
    
    # Calculate expected size
    WINDOW = 10  # আপনার window size
    expected_size = len(MARKET_COLS) * WINDOW + 4
    print(f"\n2. Expected observation size: {len(MARKET_COLS)} × {WINDOW} + 4 = {expected_size}")
    
    # Try to create environment
    print(f"\n3. Creating test environment...")
    
    # Create dummy data
    import pandas as pd
    import numpy as np
    
    dummy_data = pd.DataFrame({
        'symbol': ['TEST'] * 100,
        'date': pd.date_range('2024-01-01', periods=100),
        'open': np.random.randn(100) + 100,
        'high': np.random.randn(100) + 101,
        'low': np.random.randn(100) + 99,
        'close': np.random.randn(100) + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Add any missing MARKET_COLS columns
    for col in MARKET_COLS:
        if col not in dummy_data.columns:
            dummy_data[col] = 0
    
    env = HedgeFundTradingEnv(
        data=dummy_data,
        xgb_model_dir="./csv/xgboost/",
        config=HedgeFundConfig()
    )
    
    # Get actual observation
    obs, _ = env.reset()
    actual_size = len(obs)
    
    print(f"\n4. Actual observation size: {actual_size}")
    print(f"   Observation space shape: {env.observation_space.shape[0]}")
    
    # Compare
    print(f"\n5. RESULT:")
    if actual_size == expected_size:
        print(f"   ✅ MATCH: {actual_size} == {expected_size}")
    else:
        print(f"   ❌ MISMATCH: {actual_size} != {expected_size}")
        print(f"   Difference: {actual_size - expected_size}")
    
    # Check observation space
    expected_space = env.observation_space.shape[0]
    if actual_size == expected_space:
        print(f"   ✅ Observation space matches: {actual_size}")
    else:
        print(f"   ❌ Observation space mismatch!")
        print(f"      Reset returns: {actual_size}")
        print(f"      Space expects: {expected_space}")
    
    print("\n" + "="*50)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nCheck if xgboost_ppo_env.py exists and has:")
    print("  - HedgeFundTradingEnv class")
    print("  - HedgeFundConfig class")
    print("  - MARKET_COLS variable")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()