# monthly.py : retrain and test the signal model (run once a month, not daily)
import os
import subprocess
import sys

# change these two to your real numbers
COST = "0.01"          # round-trip commission + slippage (0.01 = 1%)
MIN_VALUE = "1000000"  # minimum 20-day average of close * volume

scripts = [
    ["scripts/hf_download.py"],
    ["scripts/mongodb.py"],
    ["scripts/train_signal.py", "--cost", COST, "--min-value", MIN_VALUE],
    ["scripts/train_signal.py", "--final-test"],
    ["scripts/upload_csv.py"],
]

for name, *args in scripts:
    try:
        subprocess.run([sys.executable, os.path.abspath(name), *args], check=True)
        print(f"Finished {name}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {name}: {e}")
        sys.exit(1)
