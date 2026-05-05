# ================== tst_download.py (UPDATED) ==================
# PatchTST Data Downloader from Hugging Face
# ✅ Downloads only LATEST checkpoints
# ✅ Deletes old checkpoints after download
# ✅ Rate limit handling with auto-retry
# ✅ Downloads patchtst_models/ folder completely

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
# DOWNLOAD FUNCTIONS
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
        
        # ✅ patchtst_models থাকলে count দেখান
        if pattern == "patchtst_models/**":
            models_dir = Path(LOCAL_DIR) / "patchtst_models"
            existing_symbols = []
            if models_dir.exists():
                existing_symbols = [d.name for d in models_dir.iterdir() 
                                   if d.is_dir() and not d.name.startswith('_')]
            print(f"   Existing local models: {len(existing_symbols)} symbols")
            
            if existing_symbols:
                print(f"   First few: {', '.join(existing_symbols[:5])}")
        
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
        pt_count = len(list(models_dir.rglob("patchtst_model.pt")))
        print(f"   🧠 Models: {symbol_count} symbols")
        print(f"   📦 Checkpoints: {checkpoint_count}")
        print(f"   🎯 Best Models: {pt_count}")
        
        # ✅ Sample symbols
        symbols = [d.name for d in models_dir.iterdir() 
                  if d.is_dir() and not d.name.startswith('_')]
        if symbols:
            print(f"   📋 Sample: {', '.join(symbols[:10])}")
    else:
        print(f"   ⚠️ patchtst_models/ NOT CREATED!")
    
    print(f"{'='*60}")
    
    if total_failed == 0:
        print(f"✅ All PatchTST data ready!")
        return True
    else:
        print(f"⚠️ Some patterns failed, but training can continue")
        return True


# =========================================================
# VERIFY
# =========================================================

def verify_patchtst_data():
    """Verify all required files are present"""
    
    print(f"\n🔍 Verifying PatchTST data...")
    print("-" * 40)
    
    # Sector চেক - recursive glob
    sector_path = Path("./csv/sector")
    if sector_path.exists():
        sector_files = list(sector_path.rglob("*.csv"))
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
        
        if weekly > 0:
            sectors = [f.stem.replace('_weekly', '') for f in weekly_path.glob("*_weekly.csv")]
            print(f"      🏭 Weekly sectors: {', '.join(sorted(sectors))}")
    
    # ✅ PatchTST models - DETAILED CHECK
    models_dir = Path("./csv/patchtst_models")
    if models_dir.exists():
        symbols = [d.name for d in models_dir.iterdir() 
                  if d.is_dir() and not d.name.startswith('_')]
        if symbols:
            ckpt_count = len(list(models_dir.rglob("epoch_*.pt")))
            pt_count = len(list(models_dir.rglob("patchtst_model.pt")))
            
            print(f"   ✅ patchtst_models/")
            print(f"      🧠 Symbols: {len(symbols)}")
            print(f"      📦 Checkpoints: {ckpt_count}")
            print(f"      🎯 Best Models (.pt): {pt_count}")
            print(f"      📋 First 10: {', '.join(sorted(symbols)[:10])}")
        else:
            print(f"   ⚠️ patchtst_models/ exists but EMPTY!")
            print(f"      Run download again or check HF token")
    else:
        print(f"   ❌ patchtst_models/ NOT FOUND!")
        print(f"      Models will NOT be loaded for training")
        print(f"      Run: python tst_download.py")
    
    print(f"-" * 40)
    
    if all_ok and models_dir.exists() and len(symbols) > 0:
        print(f"✅ Core data verified!")
        print(f"   Ready for PatchTST training with {len(symbols)} existing models!")
    elif all_ok:
        print(f"⚠️ Core data OK but no PatchTST models yet")
        print(f"   Training will start from scratch")
    else:
        print(f"⚠️ Some core files missing. Run: python tst_download.py")
    
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
    parser.add_argument("--force-models", action="store_true", help="Force re-download patchtst_models")
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
    
    elif args.force_models:
        print("🔄 Force downloading patchtst_models/** ...")
        download_single_pattern("patchtst_models/**")
        verify_patchtst_data()
    
    else:
        success = download_patchtst_data()
        
        if success:
            verify_patchtst_data()
        
        print(f"\n✅ tst_download.py complete!")


# =========================================================
# UTILITY
# =========================================================

def download_single_pattern(pattern, max_retries=5):
    """Download a single pattern with retry"""
    hf_token = os.getenv("HF_TOKEN", "") or os.getenv("hf_token", "")
    if not hf_token:
        print("❌ No HF_TOKEN")
        return False
    
    from huggingface_hub import snapshot_download
    
    for attempt in range(1, max_retries + 1):
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
            return True
        except Exception as e:
            if "429" in str(e):
                wait_time = RATE_LIMIT_WAIT
                print(f"\n   ⚠️ Rate limited! (Attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    print(f"   ⏳ Waiting {wait_time//60} minutes...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ Max retries reached")
                    return False
            else:
                print(f"   ❌ {str(e)[:100]}")
                if attempt < max_retries:
                    time.sleep(10)
    
    return False


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