from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ahashanahmed/csv",
    repo_type="dataset",
    local_dir="./csv",
    max_workers=2,
    local_dir_use_symlinks=False,
)
print("✅ ডাউনলোড সম্পূর্ণ!")
