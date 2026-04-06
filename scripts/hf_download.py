from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ahashanahmed/csv",
    repo_type="dataset",
    local_dir="./csv",
    resume_download=True,
    max_workers=2,  # slower but more stable
    timeout=600,    # 2 minutes timeout
    etag_timeout=120,
)
print("✅ ডাউনলোড সম্পূর্ণ!")
