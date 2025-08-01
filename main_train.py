#github
#git remote add origin https://github.com/Ahashanahmed2/swing.git



#main.py
import sys
import subprocess
import os

scripts = [
    'scripts/mongodb.py',
    'scripts/swing_High_Low.py',
    'scripts/swing_point.py',
    'scripts/imbalance.py',
    'scripts/main_rsu_divergence.py',
    'scripts/uptrand_downtrand.py',
    'scripts/trands.py',
    'scripts/rsi_crose_above_30.py',
    'scripts/rsi_diversence.py',
    'scripts/backtest_and_retrain.py',
    #'scripts/train_model_ppo.py',
    'scripts/signals.py',
    'scripts/error_analysis.py',
    'scripts/loss_logging_callback.py',
    'scripts/generate_signal.py',
    'scripts/sort_signals.py',
    'scripts/filter_trends.py',
    'scripts/generate_pdf.py',
    'scripts/dayliMassage.py',
    'scripts/email_reports.py',
    
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
