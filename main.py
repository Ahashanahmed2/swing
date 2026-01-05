#github
#git remote add origin https://github.com/Ahashanahmed2/swing.git



#main.py
import sys
import subprocess
import os

scripts = [
    #"scripts/mongodb.py",
    "scripts/trand.py",
    "scripts/swing_buy.py",
    "scripts/liquidly.py",
    "scripts/orderblock.py",
    "scripts/rsi_diver.py",
    "scripts/rsi_diver_retest.py",
    "scripts/short_buy.py",
    "scripts/gape.py",
    "scripts/gape_buy.py",
    "scripts/excellent.py",
    "scripts/excellent_buy.csv",
    "scripts/rsi_crose_above_30.py",
    "scripts/trend_signal.py",
    "scripts/buy.py",
    "scripts/profit_loss_generator.py",
    #"scripts/ppo_trading.py",
    "scripts/generate_pdf.py",
    "scripts/dayliMassage.py",
    #"scripts/email_reports.py",
    
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
