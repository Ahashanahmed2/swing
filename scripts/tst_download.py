# ================== tst_download.py (UPDATED) ==================
# PatchTST Data Downloader from Hugging Face
# ✅ Downloads only LATEST checkpoints
# ✅ Deletes old checkpoints after download
# ✅ Rate limit handling with auto-retry

import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================

HF_REPO = "ahashanahmed/csv"
LOCAL_DIR = "./csv"

# Files/Folders needed for PatchTST
PATCHTST_PATTERNS = [
    "mongodb.csv",
    "support_resistance.csv",
    "rsi_diver.csv",
    "sector/**/*.csv",
    "patchtst_models/**",
]

MAX_RETRIES = 5
RATE_LIMIT_WAIT = 300  # 5 minutes


# =========================================================
# CLEANUP: DELETE OLD CHECKPOINTS
# =========================================================

def delete_old_checkpoints(keep_only_latest=True):
    """
    Delete old checkpoint files, keep only:
    - patchtst_model.pt (best model)
    - Latest epoch checkpoint
    - progress.json, mistakes.json, scaler.pkl
    """
    
    models_dir = Path(LOCAL_DIR) / "patchtst_models"
    
    if not models_dir.exists():
        print("   ℹ️ No patchtst_models directory to clean")
        return
    
    deleted_count = 0
    kept_count = 0
    
    for sym_dir in models_dir.iterdir():
        if not sym_dir.is_dir() or sym_dir.name.startswith('_'):
            continue
        
        checkpoint_dir = sym_dir / "checkpoints"
        
        if not checkpoint_dir.exists():
            continue
        
        # Get all checkpoint files
        checkpoint_files = sorted(checkpoint_dir.glob("epoch_*.pt"))
        
        if len(checkpoint_files) <= 1:
            kept_count += len(checkpoint_files)
            continue
        
        if keep_only_latest:
            # Keep only the latest checkpoint
            latest = checkpoint_files[-1]
            
            for ckpt in checkpoint_files[:-1]:  # সব পুরাতন
                ckpt.unlink()
                deleted_count += 1
            
            kept_count += 1  # latest
            
            # Rename latest to epoch_final.pt
            # (optional, keep as is)
    
    print(f"\n🧹 CHECKPOINT CLEANUP:")
    print(f"   🗑️ Deleted: {deleted_count} old checkpoints")
    print(f"   📂 Kept: {kept_count} latest checkpoints")
    
    return deleted_count


def delete_empty_dirs():
    """Delete empty checkpoint directories"""
    
    models_dir = Path(LOCAL_DIR) / "patchtst_models"
    
    if not models_dir.exists():
        return
    
    for sym_dir in models_dir.iterdir():
        if not sym_dir.is_dir() or sym_dir.name.startswith('_'):
            continue
        
        checkpoint_dir = sym_dir / "checkpoints"
        
        if checkpoint_dir.exists():
            # Delete empty checkpoint dir
            files = list(checkpoint_dir.glob("*"))
            if not files:
                checkpoint_dir.rmdir()


# =========================================================
# KEEP ONLY LATEST CHECKPOINTS ON HF TOO
# =========================================================

def get_latest_checkpoint_info(symbol):
    """Get latest checkpoint info from progress.json"""
    
    progress_path = Path(LOCAL_DIR) / f"patchtst_models/{symbol}/progress.json"
    
    if progress_path.exists():
        import json
        with open(progress_path) as f:
            progress = json.load(f)
        return progress.get('last_epoch', 0), progress.get('last_loss', 0)
    
    return 0, 0


# =========================================================
# DOWNLOAD FUNCTIONS (Same as before)
# =========================================================

def download_patchtst_data():
    """Download all required PatchTST data with rate limit handling"""
    
    print("=" * 60)
    print("📥 PatchTST Data Downloader")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"📂 Repo: {HF_REPO}")
    print(f"💾 Local: {LOCAL_DIR}")
    print(f"📊 Patterns: {len(PATCHTST_PATTERNS)}")
    print("=" * 60)
    
    # ============================
    # STEP 1: DELETE OLD CHECKPOINTS
    # ============================
    print(f"\n🧹 Step 1: Cleaning old checkpoints...")
    delete_old_checkpoints(keep_only_latest=True)
    delete_empty_dirs()
    
    # ============================
    # STEP 2: DOWNLOAD FROM HF
    # ============================
    
    hf_token = os.getenv("HF_TOKEN", "") or os.getenv("hf_token", "")
    if not hf_token:
        print("❌ HF_TOKEN not found in environment!")
        return False
    
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        print("❌ huggingface_hub not installed!")
        return False
    
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    
    for pattern in PATCHTST_PATTERNS:
        print(f"\n📥 Downloading: {pattern}")
        print("-" * 40)
        
        attempt = 0
        success = False
        
        # Skip patchtst_models if no HF token or first run
        if pattern == "patchtst_models/**":
            # Check if we need to download models
            models_dir = Path(LOCAL_DIR) / "patchtst_models"
            existing_symbols = []
            if models_dir.exists():
                existing_symbols = [d.name for d in models_dir.iterdir() 
                                   if d.is_dir() and not d.name.startswith('_')]
            
            print(f"   Existing local models: {len(existing_symbols)} symbols")
        
        while attempt < MAX_RETRIES and not success:
            attempt += 1
            
            try:
                snapshot_download(
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    local_dir=LOCAL_DIR,
                    allow_patterns=pattern,
                    max_workers=2,
                    local_dir_use_symlinks=False,
                    token=hf_token,
                    resume_download=True,
                    tqdm_class=None,
                    ignore_patterns=["*.tmp", "*.log"],
                )
                
                print(f"   ✅ {pattern} (attempt {attempt})")
                total_downloaded += 1
                success = True
                
            except HfHubHTTPError as e:
                if "429" in str(e):
                    wait_time = RATE_LIMIT_WAIT
                    print(f"\n   ⚠️ Rate limited! (Attempt {attempt}/{MAX_RETRIES})")
                    
                    if attempt < MAX_RETRIES:
                        print(f"   ⏳ Waiting {wait_time//60} minutes...")
                        for remaining in range(wait_time, 0, -60):
                            mins = remaining // 60
                            print(f"      {mins} min remaining...")
                            time.sleep(60)
                    else:
                        print(f"   ❌ Max retries reached for {pattern}")
                        total_failed += 1
                        
                elif "404" in str(e):
                    print(f"   ⚠️ {pattern} (not found on HF, skipping)")
                    total_skipped += 1
                    success = True
                    
                else:
                    print(f"   ❌ {pattern} (error: {str(e)[:100]})")
                    if attempt < MAX_RETRIES:
                        time.sleep(10)
                    else:
                        total_failed += 1
                        
            except Exception as e:
                print(f"   ❌ {pattern} (error: {str(e)[:100]})")
                if attempt < MAX_RETRIES:
                    print(f"   🔄 Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    total_failed += 1
    
    # ============================
    # STEP 3: CLEANUP AGAIN (remove duplicates)
    # ============================
    print(f"\n🧹 Step 3: Cleaning duplicate checkpoints...")
    delete_old_checkpoints(keep_only_latest=True)
    delete_empty_dirs()
    
    # ============================
    # SUMMARY
    # ============================
    print(f"\n{'='*60}")
    print(f"📊 DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"   ✅ Downloaded: {total_downloaded}")
    print(f"   ⏭️ Skipped: {total_skipped}")
    print(f"   ❌ Failed: {total_failed}")
    
    csv_dir = Path(LOCAL_DIR)
    total_files = len(list(csv_dir.rglob("*"))) if csv_dir.exists() else 0
    print(f"   📂 Total files in ./csv/: {total_files}")
    
    # Count models
    models_dir = Path(LOCAL_DIR) / "patchtst_models"
    if models_dir.exists():
        symbol_count = len([d for d in models_dir.iterdir() 
                           if d.is_dir() and not d.name.startswith('_')])
        checkpoint_count = len(list(models_dir.rglob("epoch_*.pt")))
        print(f"   🧠 Models: {symbol_count} symbols, {checkpoint_count} checkpoints")
    
    print(f"{'='*60}")
    
    if total_failed == 0:
        print(f"✅ All PatchTST data ready!")
        return True
    else:
        print(f"⚠️ Some patterns failed, but training can continue")
        return True


# =========================================================
# VERIFY (Same as before)
# =========================================================

def verify_patchtst_data():
    """Verify all required files are present"""
    
    print(f"\n🔍 Verifying PatchTST data...")
    print("-" * 40)
    
    # Sector চেক - recursive glob
    sector_path = Path("./csv/sector")
    if sector_path.exists():
        sector_files = list(sector_path.rglob("*.csv"))  # ✅ recursive
        sector_ok = len(sector_files) > 0
    else:
        sector_ok = False
        sector_files = []
    
    checks = {
        "mongodb.csv": Path("./csv/mongodb.csv").exists(),
        "support_resistance.csv": Path("./csv/support_resistance.csv").exists(),
        "rsi_diver.csv": Path("./csv/rsi_diver.csv").exists(),
        "sector/": sector_ok,
    }
    
    all_ok = True
    for name, exists in checks.items():
        status = "✅" if exists else "❌"
        if not exists:
            all_ok = False
        print(f"   {status} {name}")
    
    # Sector details
    if sector_ok:
        weekly_path = sector_path / "weekly"
        daily_path = sector_path / "daily"
        
        weekly = len(list(weekly_path.glob("*.csv"))) if weekly_path.exists() else 0
        daily = len(list(daily_path.glob("*.csv"))) if daily_path.exists() else 0
        
        print(f"      📂 sector/weekly/ → {weekly} files")
        print(f"      📂 sector/daily/ → {daily} files")
        print(f"      📂 Total sector files: {len(sector_files)}")
        
        # Show sector names
        if weekly > 0:
            sectors = [f.stem.replace('_weekly', '') for f in weekly_path.glob("*_weekly.csv")]
            print(f"      🏭 Weekly sectors: {', '.join(sorted(sectors))}")
    else:
        print(f"      ⚠️ sector directory empty or missing")
    
    # PatchTST models
    models_dir = Path("./csv/patchtst_models")
    if models_dir.exists():
        symbols = [d.name for d in models_dir.iterdir() 
                  if d.is_dir() and not d.name.startswith('_')]
        if symbols:
            ckpt_count = len(list(models_dir.rglob("epoch_*.pt")))
            pt_count = len(list(models_dir.rglob("*.pt")))
            json_count = len(list(models_dir.rglob("*.json")))
            pkl_count = len(list(models_dir.rglob("*.pkl")))
            
            print(f"   ✅ patchtst_models/")
            print(f"      🧠 Symbols: {len(symbols)}")
            print(f"      📦 Checkpoints: {ckpt_count}")
            print(f"      📄 JSON files: {json_count}")
            print(f"      🥒 PKL files: {pkl_count}")
            print(f"      📊 Total model files: {pt_count + json_count + pkl_count}")
        else:
            print(f"   ℹ️ patchtst_models/ (empty directory)")
    else:
        print(f"   ℹ️ patchtst_models/ (not created yet)")
    
    # Additional checks
    print(f"\n📋 Full CSV directory stats:")
    csv_dir = Path("./csv")
    if csv_dir.exists():
        total_files = len(list(csv_dir.rglob("*")))
        total_csv = len(list(csv_dir.rglob("*.csv")))
        total_dirs = len([d for d in csv_dir.rglob("*") if d.is_dir()])
        print(f"   📁 Total files: {total_files}")
        print(f"   📊 CSV files: {total_csv}")
        print(f"   📂 Subdirectories: {total_dirs}")
    
    print(f"-" * 40)
    
    if all_ok:
        print(f"✅ Core data verified!")
        print(f"   Ready for PatchTST training!")
    else:
        print(f"⚠️ Some core files missing. Run: python tst_download.py")
        # Show specific issues
        for name, exists in checks.items():
            if not exists:
                print(f"   ❌ Missing: {name}")
    
    return all_ok

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PatchTST Data Downloader (Latest Only)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify data")
    parser.add_argument("--clean-only", action="store_true", help="Only clean old checkpoints")
    parser.add_argument("--pattern", type=str, help="Download single pattern")
    args = parser.parse_args()
    
    if args.verify_only:
        verify_patchtst_data()
    
    elif args.clean_only:
        print("🧹 Cleaning old checkpoints...")
        deleted = delete_old_checkpoints(keep_only_latest=True)
        delete_empty_dirs()
        print(f"✅ Deleted {deleted} old checkpoint files")
        verify_patchtst_data()
    
    elif args.pattern:
        print(f"📥 Downloading: {args.pattern}")
        download_single_pattern(args.pattern)
        delete_old_checkpoints(keep_only_latest=True)
        verify_patchtst_data()
    
    else:
        success = download_patchtst_data()
        
        if success:
            verify_patchtst_data()
        
        print(f"\n✅ tst_download.py complete!")


# =========================================================
# UTILITY
# =========================================================

def ensure_patchtst_data():
    """Call from other scripts to ensure data is available"""
    
    required = [
        "./csv/mongodb.csv",
        "./csv/support_resistance.csv", 
        "./csv/rsi_diver.csv",
    ]
    
    missing = [f for f in required if not Path(f).exists()]
    
    if missing or not Path("./csv/sector").exists():
        print("📥 Downloading PatchTST data...")
        return download_patchtst_data()
    else:
        print("✅ PatchTST data already exists")
        return True
