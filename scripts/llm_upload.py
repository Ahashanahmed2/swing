# scripts/llm_upload.py
# 🔥 ZERO-BUG PRODUCTION VERSION

import os
from huggingface_hub import HfApi, login, create_repo, upload_file


def get_training_text():
    """
    Keep training data isolated to avoid syntax break issues
    """
    return """Symbol: RELIANCE1
Pattern: Cup and Handle detected. Cup bottom at 52, handle between 55-57.
Breakout above 58 confirmed. Target: 65. Stop loss: 54.

Symbol: KPCL
Pattern: Bull Flag. Flagpole from 45 to 55. Consolidation between 52-54.
Breakout above 55 with volume. Target: 62.

Symbol: TECHNODRUG
Pattern: Double Bottom at 30 and 31. Neckline at 35. Breakout confirmed.
Target: 40.

Symbol: APEXFOOT
Pattern: Bull Flag. Sharp move from 50 to 60, consolidation at 57-59.
Breakout above 60. Target: 70.

Symbol: SONALIANSH
Pattern: Cup and Handle. Cup bottom at 100, handle at 110-115.
Breakout above 118. Target: 135.

Symbol: VFSTDL
Pattern: Double Bottom. Bottom at 25 and 26. Neckline at 30.
Breakout above 30 confirmed. Target: 38.

Symbol: PF1STMF
Pattern: Bull Flag. Sharp rally from 80 to 95. Consolidation at 90-93.
Breakout expected above 95. Target: 110.
"""


def main():
    print("=" * 60)
    print("🚀 LLM DATA UPLOAD STARTED")
    print("=" * 60)

    # =========================================================
    # 1. TOKEN CHECK
    # =========================================================
    token = os.getenv("hf_token")

    if not token:
        raise ValueError("❌ hf_token not found in environment variables!")

    print("✅ Token found")

    # =========================================================
    # 2. LOGIN
    # =========================================================
    try:
        login(token=token)
        print("✅ Hugging Face login successful")
    except Exception as e:
        raise RuntimeError(f"❌ Login failed: {e}")

    # =========================================================
    # 3. REPO SETUP
    # =========================================================
    repo_id = "ahashanahmed/LLM_model_stock"
    api = HfApi()

    try:
        create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"✅ Repo ready: {repo_id}")
    except Exception as e:
        raise RuntimeError(f"❌ Repo creation failed: {e}")

    # =========================================================
    # 4. CREATE FILE
    # =========================================================
    file_path = "training_texts.txt"

    try:
        training_text = get_training_text()

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(training_text)

        print(f"✅ File created: {file_path}")
    except Exception as e:
        raise RuntimeError(f"❌ File creation failed: {e}")

    # =========================================================
    # 5. UPLOAD
    # =========================================================
    try:
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo="training_texts.txt",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("✅ Upload successful")
    except Exception as e:
        raise RuntimeError(f"❌ Upload failed: {e}")

    print("=" * 60)
    print("🎉 ALL DONE - ZERO BUG EXECUTION")
    print("=" * 60)


# =========================================================
# ENTRY POINT (STRICTLY CLEAN)
# =========================================================
if __name__ == "__main__":
    main()
