# ================== tst_download.py ==================
# PatchTST Data Downloader from Hugging Face
# Downloads only required files for PatchTST training
# No files deleted, only downloads missing/new files
# ✅ Rate limit handling with auto-retry

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================

HF_REPO = "ahashanahmed/csv"
LOCAL_DIR = "./csv"

# Files/Folders needed for PatchTST
PATCHTST_PATTERNS = [
    "mongodb.csv",              # Main OHLCV data
    "support_resistance.csv",   # Support/Resistance levels
    "rsi_diver.csv",           # RSI Divergence
    "sector/*.csv",            # Sector features (daily + weekly)
    "patchtst_models/**",      # Existing PatchTST models (checkpoints)
]

MAX_RETRIES = 5
RATE_LIMIT_WAIT = 300  # 5 minutes


# =========================================================
# DOWNLOAD FUNCTION WITH RATE LIMIT HANDLER
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
    
    # Check HF token
    hf_token = os.getenv("HF_TOKEN", "") or os.getenv("hf_token", "")
    if not hf_token:
        print("❌ HF_TOKEN not found in environment!")
        print("   Set with: export HF_TOKEN=your_token")
        return False
    
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        print("❌ huggingface_hub not installed!")
        print("   Install: pip install huggingface_hub")
        return False
    
    # Download each pattern separately
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    
    for pattern in PATCHTST_PATTERNS:
        print(f"\n📥 Downloading: {pattern}")
        print("-" * 40)
        
        attempt = 0
        success = False
        
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
                    # Rate limited
                    wait_time = RATE_LIMIT_WAIT
                    print(f"\n   ⚠️ Rate limited! (Attempt {attempt}/{MAX_RETRIES})")
                    
                    if attempt < MAX_RETRIES:
                        print(f"   ⏳ Waiting {wait_time//60} minutes...")
                        print(f"   📊 Already downloaded files will resume")
                        
                        # Countdown
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
                    success = True  # Skip this pattern
                    
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
    # SUMMARY
    # ============================
    print(f"\n{'='*60}")
    print(f"📊 DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"   ✅ Downloaded: {total_downloaded}")
    print(f"   ⏭️ Skipped: {total_skipped}")
    print(f"   ❌ Failed: {total_failed}")
    
    # Count files
    csv_dir = Path(LOCAL_DIR)
    total_files = len(list(csv_dir.rglob("*"))) if csv_dir.exists() else 0
    print(f"   📂 Total files in ./csv/: {total_files}")
    print(f"{'='*60}")
    
    if total_failed == 0:
        print(f"✅ All PatchTST data ready!")
        return True
    else:
        print(f"⚠️ Some patterns failed, but training can continue")
        return True


# =========================================================
# QUICK DOWNLOAD (Single pattern)
# =========================================================

def download_single_pattern(pattern, max_retries=3):
    """Download a single pattern with retry"""
    
    hf_token = os.getenv("HF_TOKEN", "") or os.getenv("hf_token", "")
    
    if not hf_token:
        print("❌ No HF_TOKEN")
        return False
    
    from huggingface_hub import snapshot_download
    
    attempt = 0
    while attempt < max_retries:
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
            print(f"✅ {pattern}")
            return True
        except Exception as e:
            if "429" in str(e):
                wait_time = 300
                print(f"\n⚠️ Rate limited! (Attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    print(f"⏳ Waiting {wait_time//60} minutes...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Max retries reached")
                    return False
            else:
                print(f"❌ {pattern}: {str(e)[:50]}")
                if attempt < max_retries:
                    time.sleep(10)
                else:
                    return False
    
    return False


# =========================================================
# VERIFY DOWNLOADED DATA
# =========================================================

def verify_patchtst_data():
    """Verify all required files are present"""
    
    print(f"\n🔍 Verifying PatchTST data...")
    print("-" * 40)
    
    checks = {
        "mongodb.csv": Path("./csv/mongodb.csv").exists(),
        "support_resistance.csv": Path("./csv/support_resistance.csv").exists(),
        "rsi_diver.csv": Path("./csv/rsi_diver.csv").exists(),
        "sector/": Path("./csv/sector").exists() and len(list(Path("./csv/sector").glob("*.csv"))) > 0,
    }
    
    all_ok = True
    for name, exists in checks.items():
        status = "✅" if exists else "❌"
        if not exists:
            all_ok = False
        print(f"   {status} {name}")
    
    # PatchTST models
    models_dir = Path("./csv/patchtst_models")
    if models_dir.exists():
        model_files = list(models_dir.rglob("*.pt"))
        if model_files:
            print(f"   ✅ patchtst_models/ ({len(model_files)} model files)")
        else:
            print(f"   ℹ️ patchtst_models/ (empty)")
    else:
        print(f"   ℹ️ patchtst_models/ (not created yet)")
    
    print(f"-" * 40)
    
    if all_ok:
        print(f"✅ Core data verified!")
    else:
        print(f"⚠️ Some core files missing. Run: python tst_download.py")
    
    return all_ok


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PatchTST Data Downloader")
    parser.add_argument("--verify-only", action="store_true", help="Only verify data")
    parser.add_argument("--pattern", type=str, help="Download single pattern")
    parser.add_argument("--max-retries", type=int, default=5, help="Max download retries")
    args = parser.parse_args()
    
    if args.verify_only:
        verify_patchtst_data()
    
    elif args.pattern:
        print(f"📥 Downloading: {args.pattern}")
        download_single_pattern(args.pattern, args.max_retries)
        verify_patchtst_data()
    
    else:
        success = download_patchtst_data()
        
        if success:
            verify_patchtst_data()
        
        print(f"\n✅ tst_download.py complete!")


# =========================================================
# EASY IMPORT FOR OTHER SCRIPTS
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