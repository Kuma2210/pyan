from huggingface_hub import snapshot_download
import os

# --- 新增代码：设置Hugging Face镜像地址 ---
# 这会告诉 huggingface_hub 库从 hf-mirror.com 下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ----------------------------------------

# ！！！ 替换为你自己的 Token ！！！
# 即使使用镜像，你仍然需要Token来访问门控模型
HF_AUTH_TOKEN = "YOUR_HUGGING_FACE_TOKEN_GOES_HERE"

MODEL_ID = "pyannote/speaker-diarization-3.1"
LOCAL_MODEL_DIR = MODEL_ID.replace("/", "_") 

if not os.path.exists(LOCAL_MODEL_DIR):
    os.makedirs(LOCAL_MODEL_DIR)

print(f"正在从镜像站 '{os.environ.get('HF_ENDPOINT')}' 下载模型 '{MODEL_ID}' ...")

try:
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_MODEL_DIR,
        use_auth_token=HF_AUTH_TOKEN,
        repo_type="model" 
    )
    print(f"\n模型下载成功！文件已保存到: {os.path.abspath(LOCAL_MODEL_DIR)}")
    print("下一步：请将这个完整的文件夹打包（例如 .zip），然后上传到你的公司平台。")
    
except Exception as e:
    print(f"\n下载失败: {e}")
    print(f"请检查你的 HF_AUTH_TOKEN 和网络连接。")
    print(f"同时也请确认镜像站 '{os.environ.get('HF_ENDPOINT')}' 是否可访问。")
