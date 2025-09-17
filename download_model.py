from huggingface_hub import snapshot_download
import os

# --- 配置 ---
# ！！！ 替换为你自己的 Token ！！！
HF_AUTH_TOKEN = "YOUR_HUGGING_FACE_TOKEN_GOES_HERE"

# 设置镜像站 (可选, 如果你需要)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 我们需要下载的 *所有* 模型
MODELS_TO_DOWNLOAD = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/embedding-3.0"
]
# ----------------

print(f"准备下载 {len(MODELS_TO_DOWNLOAD)} 个模型...")

for model_id in MODELS_TO_DOWNLOAD:
    LOCAL_MODEL_DIR = model_id.replace("/", "_") 
    
    if not os.path.exists(LOCAL_MODEL_DIR):
        os.makedirs(LOCAL_MODEL_DIR)

    print(f"\n--- 正在下载: {model_id} ---")
    print(f"    将保存到: ./{LOCAL_MODEL_DIR}")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=LOCAL_MODEL_DIR,
            use_auth_token=HF_AUTH_TOKEN,
            repo_type="model",
            # 忽略 .safetensors 文件（pyannote 目前主要用 .bin）
            # 这可以减少下载量
            ignore_patterns=["*.safetensors*"], 
        )
        print(f"--- {model_id} 下载成功 ---")
        
    except Exception as e:
        print(f"\n下载 {model_id} 失败: {e}")
        print("请检查你的 Token 和网络。")

print("\n所有模型下载完毕。请将以下文件夹打包上传到您的平台：")
for model_id in MODELS_TO_DOWNLOAD:
    print(f"- {model_id.replace('/', '_')}")
