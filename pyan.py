import torch
from pyannote.audio import Pipeline
import warnings

# ... (get_device 函数和警告设置保持不变) ...
DEVICE = get_device()


# --- 加载 SOTA 流水线 (离线方式) ---

# ！！！ 替换为你上传模型后，在公司平台上的真实路径 ！！！
LOCAL_MODEL_PATH = "/home/my-user/models/pyannote_speaker-diarization-3.1"

print(f"正在从本地路径加载流水线: {LOCAL_MODEL_PATH} ...")

try:
    # 关键改动：
    # 1. from_pretrained() 的第一个参数现在是本地文件夹路径
    # 2. 不再需要 use_auth_token=... 参数
    pipeline = Pipeline.from_pretrained(
        LOCAL_MODEL_PATH
    )
    pipeline.to(DEVICE)
    print("流水线从本地加载成功。")

except Exception as e:
    print(f"从本地加载流水线失败: {e}")
    print("\n--- 请确保: ---")
    print(f"1. 路径 '{LOCAL_MODEL_PATH}' 是正确的。")
    print("2. 该路径下包含了模型文件 (如 'config.yaml', 'pytorch_model.bin' 等)。")
    print("3. 你在公司平台上安装的 pyannote.audio 库版本与下载模型时兼容。")
    print("------------------\n")
    exit(1)


# ... (contains_multiple_speakers 函数 和 __main__ 部分保持不变) ...

# ... (在 __main__ 部分，你也不再需要检查 HF_AUTH_TOKEN 了) ...
