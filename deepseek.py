# ================== deepseek.py ==================
# ✅ DeepSeek Training Pipeline Orchestrator
# ✅ Step 1: Download (model + data + checkpoints)
# ✅ Step 2: Generate pattern training data
# ✅ Step 3: Train DeepSeek-R1-Distill-Qwen-1.5B

# github
# git remote add origin https://github.com/Ahashanahmed2/swing.git

import sys
import subprocess
import os
from datetime import datetime

# =========================================================
# PIPELINE SCRIPTS
# =========================================================

scripts = [
    "scripts/deepseek_download.py",
    "scripts/generate_pattern_training_data_complete.py",
    "scripts/deepseek_train.py",
]

# =========================================================
# RUN PIPELINE
# =========================================================

def run_pipeline():
    print("=" * 60)
    print(f"🚀 DeepSeek Pipeline Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    env = os.environ.copy()
    failed = []

    for i, script in enumerate(scripts, 1):
        script_path = os.path.abspath(script)

        print(f"\n{'=' * 60}")
        print(f"Step {i}/{len(scripts)}: {script}")
        print(f"{'=' * 60}")

        # ✅ File existence check
        if not os.path.exists(script_path):
            print(f"❌ File not found: {script}")
            failed.append(script)
            sys.exit(1)

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                check=True,
                env=env,
                timeout=22000,  # 2 hours per script
            )
            print(f"✅ Finished: {script}")

        except subprocess.CalledProcessError as e:
            print(f"❌ FAILED: {script} (exit code {e.returncode})")
            sys.exit(1)

        except subprocess.TimeoutExpired:
            print(f"❌ TIMEOUT: {script} (exceeded 2 hours)")
            sys.exit(1)

        except Exception as e:
            print(f"❌ ERROR: {script} — {type(e).__name__}: {e}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print(f"🎉 DeepSeek Pipeline Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
