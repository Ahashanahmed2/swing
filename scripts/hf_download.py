from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ahashanahmed/csv",
    repo_type="dataset",
    local_dir="./csv"
)
print("✅ ডাউনলোড সম্পূর্ণ!")