from huggingface_hub import login, upload_folder, snapshot_download, HfApi, hf_hub_download
import os
import shutil
import time
import pandas as pd
import hashlib
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("hf_token")
USERNAME = "ahashanahmed"
REPO_NAME = "csv"
REPO_ID = f"{USERNAME}/{REPO_NAME}"

# ==================== বেসিক ফাংশন ====================

def hf_login(token=None):
    """Hugging Face লগইন"""
    if token:
        try:
            login(token=token)
            print("🔐 HF login সফল হয়েছে।")
            return True
        except Exception as e:
            print(f"❌ HF login ব্যর্থ: {e}")
            return False
    return False

def is_valid_directory(local_dir: str) -> bool:
    """ডিরেক্টরি ভ্যালিড কিনা চেক"""
    return os.path.isdir(local_dir) and len(os.listdir(local_dir)) > 0

def create_repo_if_not_exists(repo_id: str = REPO_ID, token: str = HF_TOKEN):
    """রিপোজিটরি তৈরি (যদি না থাকে)"""
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
        print(f"ℹ️ Repo '{repo_id}' আগে থেকেই আছে।")
        return True
    except Exception:
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, token=token)
            print(f"✅ নতুন Repo তৈরি হয়েছে: {repo_id}")
            return True
        except Exception as e:
            print(f"❌ Repo তৈরি ব্যর্থ: {e}")
            return False

# ==================== স্মার্ট আপলোড ক্লাস ====================

class SmartDatasetUploader:
    """স্মার্ট ডাটাসেট আপলোডার - শুধু পরিবর্তিত ফাইল আপলোড করে"""
    
    def __init__(self, repo_id=REPO_ID, token=HF_TOKEN):
        self.api = HfApi()
        self.repo_id = repo_id
        self.token = token
        self.metadata_file = ".dataset_metadata.json"
        self.stats = {
            'total_files': 0,
            'new_files': 0,
            'modified_files': 0,
            'unchanged_files': 0,
            'failed_files': 0
        }
    
    def get_file_hash(self, file_path):
        """ফাইলের MD5 হ্যাশ বের করে"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"⚠️ হ্যাশ গণনা ব্যর্থ: {e}")
            return None
    
    def get_remote_metadata(self):
        """HF থেকে মেটাডাটা ডাউনলোড"""
        try:
            # টেম্প ফাইলে মেটাডাটা ডাউনলোড
            temp_metadata = f"temp_{self.metadata_file}"
            hf_hub_download(
                repo_id=self.repo_id,
                filename=self.metadata_file,
                repo_type="dataset",
                token=self.token,
                local_path=temp_metadata
            )
            
            with open(temp_metadata, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            os.remove(temp_metadata)
            print(f"📋 মেটাডাটা পাওয়া গেছে: {len(metadata.get('files', {}))} টি ফাইলের তথ্য")
            return metadata
            
        except Exception as e:
            print(f"📋 কোন মেটাডাটা নেই। নতুন মেটাডাটা তৈরি করা হবে।")
            return {
                "files": {}, 
                "last_sync": None,
                "created_at": datetime.now().isoformat()
            }
    
    def upload_metadata(self, metadata):
        """মেটাডাটা HF-এ আপলোড"""
        try:
            # মেটাডাটা ফাইল তৈরি
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # মেটাডাটা আপলোড
            self.api.upload_file(
                path_or_fileobj=self.metadata_file,
                path_in_repo=self.metadata_file,
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token
            )
            
            os.remove(self.metadata_file)
            print(f"📋 মেটাডাটা আপলোড সফল")
            return True
            
        except Exception as e:
            print(f"⚠️ মেটাডাটা আপলোড ব্যর্থ: {e}")
            return False
    
    def merge_csv_files(self, local_path, remote_filename, unique_columns=None):
        """
        দুই CSV ফাইল মার্জ করে
        - unique_columns: যে কলামের ভিত্তিতে ডুপ্লিকেট রিমুভ হবে (যেমন: ['id', 'timestamp'])
        """
        temp_remote = f"temp_remote_{int(time.time())}.csv"
        
        try:
            # ১. লোকাল CSV পড়ি
            local_df = pd.read_csv(local_path)
            print(f"   লোকাল ডাটা: {len(local_df)} রো")
            
            # ২. রিমোট CSV ডাউনলোড করে পড়ি
            try:
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=remote_filename,
                    repo_type="dataset",
                    token=self.token,
                    local_path=temp_remote
                )
                
                remote_df = pd.read_csv(temp_remote)
                print(f"   রিমোট ডাটা: {len(remote_df)} রো")
                
                # ৩. ডাটা মার্জ
                if unique_columns and all(col in remote_df.columns for col in unique_columns):
                    # ইউনিক কলামের ভিত্তিতে মার্জ
                    combined_df = pd.concat([remote_df, local_df], ignore_index=True)
                    
                    # ডুপ্লিকেট রিমুভ (সবচেয়ে নতুনটা রাখবে)
                    if 'timestamp' in combined_df.columns:
                        combined_df = combined_df.sort_values('timestamp', ascending=False)
                    
                    combined_df = combined_df.drop_duplicates(
                        subset=unique_columns, 
                        keep='first'
                    )
                    print(f"   ডুপ্লিকেট রিমুভের পর: {len(combined_df)} রো")
                    
                else:
                    # ইউনিক কলাম না থাকলে বা না মিললে সব ডাটা রাখি
                    combined_df = pd.concat([remote_df, local_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(keep='last')
                    print(f"   সব ডাটা মার্জ: {len(combined_df)} রো")
                
                # টেম্প ফাইল ডিলিট
                if os.path.exists(temp_remote):
                    os.remove(temp_remote)
                
                return combined_df
                
            except Exception as e:
                print(f"   রিমোট ফাইল নেই, শুধু লোকাল ডাটা আপলোড হবে")
                if os.path.exists(temp_remote):
                    os.remove(temp_remote)
                return local_df
                
        except Exception as e:
            print(f"⚠️ মার্জিং ব্যর্থ: {e}")
            if os.path.exists(temp_remote):
                os.remove(temp_remote)
            return None
    
    def upload_file_with_retry(self, file_path, filename, retries=3, delay=2):
        """রিট্রাই সহ ফাইল আপলোড"""
        for attempt in range(1, retries + 1):
            try:
                self.api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=filename,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self.token
                )
                return True
            except Exception as e:
                print(f"   ⏳ আপলোড চেষ্টা {attempt} ব্যর্থ: {e}")
                if attempt < retries:
                    time.sleep(delay * attempt)
        return False
    
    def smart_upload(self, local_folder="./csv", unique_columns=None):
        """
        স্মার্ট আপলোড ফাংশন
        - unique_columns: CSV মার্জের জন্য ইউনিক কলাম (যেমন: ['id'])
        """
        
        # ০. প্রি-চেক
        if not hf_login(self.token):
            return False
        
        if not create_repo_if_not_exists(self.repo_id, self.token):
            return False
        
        if not is_valid_directory(local_folder):
            print(f"⚠️ আপলোডের জন্য ফোল্ডার খালি বা নেই: {local_folder}")
            return False
        
        print(f"\n{'='*60}")
        print(f"🚀 স্মার্ট আপলোড শুরু: {local_folder}")
        print(f"{'='*60}\n")
        
        # ১. মেটাডাটা লোড
        metadata = self.get_remote_metadata()
        remote_files = metadata.get('files', {})
        
        # ২. লোকাল ফাইল স্ক্যান
        local_files = {}
        csv_files = [f for f in os.listdir(local_folder) if f.endswith('.csv')]
        
        print(f"\n📁 লোকাল ফাইল স্ক্যানিং...")
        for filename in csv_files:
            file_path = os.path.join(local_folder, filename)
            
            if os.path.isfile(file_path):
                file_hash = self.get_file_hash(file_path)
                file_size = os.path.getsize(file_path)
                modified_time = os.path.getmtime(file_path)
                
                if file_hash:
                    local_files[filename] = {
                        'hash': file_hash,
                        'size': file_size,
                        'modified': modified_time,
                        'modified_str': datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S'),
                        'path': file_path
                    }
        
        self.stats['total_files'] = len(local_files)
        print(f"\n📊 ফাইল বিশ্লেষণ:")
        print(f"   মোট CSV ফাইল: {self.stats['total_files']}")
        
        # ৩. ফাইল তুলনা
        files_to_process = []
        
        for filename, local_info in local_files.items():
            remote_info = remote_files.get(filename, {})
            
            if filename not in remote_files:
                # সম্পূর্ণ নতুন ফাইল
                files_to_process.append(('new', filename, local_info))
                self.stats['new_files'] += 1
                print(f"   🆕 নতুন ফাইল: {filename} ({local_info['size']/1024:.1f}KB)")
                
            elif local_info['hash'] != remote_info.get('hash'):
                # ফাইল পরিবর্তিত হয়েছে
                files_to_process.append(('modified', filename, local_info))
                self.stats['modified_files'] += 1
                print(f"   📝 পরিবর্তিত: {filename} ({local_info['size']/1024:.1f}KB)")
                
            else:
                # অপরিবর্তিত ফাইল
                self.stats['unchanged_files'] += 1
        
        print(f"\n🔄 প্রক্রিয়াকরণ শুরু...\n")
        
        # ৪. ফাইল প্রক্রিয়াকরণ
        for change_type, filename, local_info in files_to_process:
            try:
                print(f"📄 {filename}:")
                
                if change_type == 'modified':
                    # মার্জিং সহ আপলোড
                    print(f"   মার্জ করছি...")
                    merged_df = self.merge_csv_files(
                        local_info['path'], 
                        filename,
                        unique_columns
                    )
                    
                    if merged_df is not None:
                        # মার্জ করা ডাটা টেম্প ফাইলে সেভ
                        temp_file = f"temp_merged_{int(time.time())}_{filename}"
                        merged_df.to_csv(temp_file, index=False, encoding='utf-8')
                        
                        # আপলোড
                        print(f"   আপলোড করছি...")
                        if self.upload_file_with_retry(temp_file, filename):
                            # মেটাডাটা আপডেট
                            metadata['files'][filename] = {
                                'hash': local_info['hash'],
                                'size': local_info['size'],
                                'modified': local_info['modified'],
                                'last_upload': datetime.now().isoformat(),
                                'merged': True
                            }
                            print(f"   ✅ মার্জ ও আপলোড সফল")
                        else:
                            self.stats['failed_files'] += 1
                            print(f"   ❌ আপলোড ব্যর্থ")
                        
                        # টেম্প ফাইল ডিলিট
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    else:
                        self.stats['failed_files'] += 1
                        print(f"   ❌ মার্জ ব্যর্থ")
                
                else:  # নতুন ফাইল
                    print(f"   সরাসরি আপলোড করছি...")
                    if self.upload_file_with_retry(local_info['path'], filename):
                        # মেটাডাটা আপডেট
                        metadata['files'][filename] = {
                            'hash': local_info['hash'],
                            'size': local_info['size'],
                            'modified': local_info['modified'],
                            'last_upload': datetime.now().isoformat(),
                            'merged': False
                        }
                        print(f"   ✅ আপলোড সফল")
                    else:
                        self.stats['failed_files'] += 1
                        print(f"   ❌ আপলোড ব্যর্থ")
                
                print()
                
            except Exception as e:
                self.stats['failed_files'] += 1
                print(f"   ❌ এরর: {str(e)}\n")
        
        # ৫. মেটাডাটা আপডেট
        metadata['last_sync'] = datetime.now().isoformat()
        self.upload_metadata(metadata)
        
        # ৬. সারসংক্ষেপ
        print(f"\n{'='*60}")
        print(f"📊 চূড়ান্ত সারসংক্ষেপ:")
        print(f"{'='*60}")
        print(f"   মোট ফাইল: {self.stats['total_files']}")
        print(f"   নতুন ফাইল: {self.stats['new_files']}")
        print(f"   পরিবর্তিত ফাইল: {self.stats['modified_files']}")
        print(f"   অপরিবর্তিত: {self.stats['unchanged_files']}")
        print(f"   ব্যর্থ: {self.stats['failed_files']}")
        print(f"{'='*60}\n")
        
        return self.stats['failed_files'] == 0

# ==================== সাধারণ আপলোড ফাংশন ====================

def simple_upload(folder_path="./csv", repo_id=REPO_ID, token=HF_TOKEN, retries=3, delay=5):
    """সাধারণ ফোল্ডার আপলোড (সব ফাইল আপলোড করে)"""
    hf_login(token)
    create_repo_if_not_exists(repo_id, token)

    if not is_valid_directory(folder_path):
        print(f"⚠️ আপলোডের জন্য ফোল্ডার খালি বা নেই: {folder_path}")
        return False

    print(f"📤 আপলোড শুরু: {folder_path} → {repo_id}")
    for attempt in range(1, retries + 1):
        try:
            upload_folder(folder_path=folder_path, repo_id=repo_id, repo_type="dataset", token=token)
            print("✅ HF আপলোড সফল হয়েছে।")
            return True
        except Exception as e:
            print(f"⏳ আপলোড চেষ্টা {attempt} ব্যর্থ: {e}")
            time.sleep(delay)

    print("❌ HF আপলোড সম্পূর্ণভাবে ব্যর্থ হয়েছে।")
    return False

# ==================== ডাউনলোড ফাংশন ====================

def download_from_hf(local_dir="./csv", repo_id=REPO_ID, token=HF_TOKEN):
    """HF থেকে ডাটাসেট ডাউনলোড"""
    create_repo_if_not_exists(repo_id, token)

    if is_valid_directory(local_dir):
        print("✅ লোকাল './csv' ফোল্ডার আগে থেকেই আছে।")
        return True

    print("🔍 লোকাল './csv' ফোল্ডার নেই, HF থেকে ডাউনলোড চেষ্টা করছি...")

    try:
        hf_login(token)
        temp_path = snapshot_download(repo_id=repo_id, repo_type="dataset", token=token)

        def copy_contents(src_dir, dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
            for item in os.listdir(src_dir):
                s = os.path.join(src_dir, item)
                d = os.path.join(dst_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                elif os.path.isfile(s):
                    shutil.copy2(s, d)

        copy_contents(temp_path, local_dir)
        print("✅ HF থেকে ডাউনলোড সফল, './csv' ফোল্ডারে কপি সম্পন্ন।")
        return True

    except Exception as e:
        print(f"⚠️ HF থেকে ডাউনলোড ব্যর্থ: {e}")
        return False

# ==================== ব্যবহারের উদাহরণ ====================

if __name__ == "__main__":
    
    # অপশন ১: স্মার্ট আপলোড (শুধু পরিবর্তিত ফাইল)
    print("\n🔧 স্মার্ট আপলোডার ব্যবহার:")
    uploader = SmartDatasetUploader(REPO_ID, HF_TOKEN)
    
    # CSV মার্জের জন্য ইউনিক কলাম নির্ধারণ (আপনার CSV অনুযায়ী পরিবর্তন করুন)
    unique_columns = ['id']  # আপনার CSV-এর প্রাইমারি কি
    
    uploader.smart_upload(
        local_folder="./csv",
        unique_columns=unique_columns  # ইউনিক কলাম সেট করুন
    )
    
    # অপশন ২: সাধারণ আপলোড (সব ফাইল আপলোড)
    # simple_upload("./csv")
    
    # অপশন ৩: ডাউনলোড
    # download_from_hf("./csv")