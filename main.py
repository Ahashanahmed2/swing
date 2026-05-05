#github
#git remote add origin https://github.com/Ahashanahmed2/swing.git



#main.py
import sys
import subprocess
import os

scripts = [
    
    #"scripts/dse_scraper.py",no
    "scripts/date_check.py",
    "scripts/hf_download.py",
    
    "scripts/mongodb.py",
    "scripts/sector_candle.py",
    "scripts/sector_weekly_diver_daily_symbol.py",
    "scripts/support_buy.py",
    "scripts/add_ai_value_support_resistance.py",
    "scripts/sync_emails.py",
    "scripts/trand.py",
    "scripts/trend_signal.py",
   
    #"scripts/pattarn.py",no
    #"scripts/elliott_wave.py",
    "scripts/ema_200.py",
    "scripts/support.py",
    "scripts/buy_csv.py",
    #"scripts/buy.py",no
    "scripts/trade_stock_ai.py",
    "scripts/swing_buy.py",
    #"scripts/liquidly.py",no
    "scripts/orderblock.py",
    "scripts/short_buy.py",
    "scripts/rsi_diver.py",
    "scripts/rsi_crose_above_30.py",
    "scripts/gape.py",
    "scripts/gape_buy.py",
    "scripts/g_gape_buy.py",
    "scripts/g_swing_buy.py",
    "scripts/g_30_buy.py",
    "scripts/daily_buy.csv",
    "scripts/csv_to_ai_signal_file_transport.py",
    "scripts/macd.py",
    "scripts/macd_daily.py",
    "scripts/down_macd.py",
    "scripts/ai_trade.py",
    "scripts/xgboost_retrain.py",
    #"scripts/ppo_train.py",no
    "scripts/nightly_trader.py",
    #"scripts/agentic_loop.py",no
    "scripts/generate_final_ai_signals.py",
    "scripts/generate_pdf.py",
    "scripts/upload_csv.py",
    "scripts/dayliMassage.py",
    "scripts/email_reports.py",
    "scripts/save_to_mongodb.py"
    
    #"scripts/generate_pattern_training_data_complete.py",no
    #"scripts/llm_tra"in.py",no
    #"scripts/fix_and_train_missing_symbol.py",no
]

for script in scripts:
    script_path=os.path.abspath(script)
    try:
        subprocess.run([sys.executable,script_path],check=True)
        print(f"Finished {script}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}:{e}")
        break
    except FileNotFoundError:
        print(f"File not found :{script}")
        break
