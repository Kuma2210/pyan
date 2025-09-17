import torch
from pyannote.audio import Pipeline
import warnings
import os

# --- 配置 ---

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    else: return torch.device("cpu")

DEVICE = get_device()
print(f"正在使用设备: {DEVICE}")

# --- 加载 SOTA 流水线 (离线方式) ---

# ！！！ 关键步骤 ！！！
# 仅指向 *打过补丁* 的 *主模型* config.yaml
LOCAL_MODEL_CONFIG_FILE = "/PATH/TO/pyannote_speaker-diarization-3.1/config.yaml" # <-- ！！！ 修改我 ！！！


print(f"正在从本地配置文件加载流水线: {LOCAL_MODEL_CONFIG_FILE} ...")

# 检查路径是否被修改
if "PATH/TO/" in LOCAL_MODEL_CONFIG_FILE:
    print("\n错误: 请修改 'LOCAL_MODEL_CONFIG_FILE' 变量！\n")
    sys.exit(1)

# 检查 *文件* 是否存在
if not os.path.isfile(LOCAL_MODEL_CONFIG_FILE):
    print(f"\n错误: 配置文件 '{LOCAL_MODEL_CONFIG_FILE}' 未找到。")
    sys.exit(1)

try:
    # 加载这个已被 patch_config.py 修改过的文件
    pipeline = Pipeline.from_pretrained(
        LOCAL_MODEL_CONFIG_FILE
    )
    pipeline.to(DEVICE)
    print("流水线从本地加载成功。")

except Exception as e:
    print(f"从本地加载流水线失败: {e}")
    print("如果 'patch_config.py' 已成功运行, 仍出现此错误，")
    print("请检查 pyannote.audio 库是否已在您的环境中正确安装。")
    exit(1)


def contains_multiple_speakers(audio_file_path: str) -> bool:
    print(f"\n正在处理文件: {audio_file_path}")
    try:
        diarization = pipeline(audio_file_path)
        speaker_labels = diarization.labels()
        num_speakers = len(speaker_labels)
        print(f"检测到的唯一说话人数量: {num_speakers}")
        
        print("--- 详细日志 (可选) ---")
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"时间: [{turn.start:04.1f}s - {turn.end:04.1f}s] 说话人: {speaker}")
        print("------------------------")
        return num_speakers > 1
    except Exception as e:
        print(f"处理文件 {audio_file_path} 时出错: {e}")
        return False

# --- 示例用法 ---
if __name__ == "__main__":
    test_files = [
        "/PATH/TO/YOUR/SINGLE_SPEAKER_AUDIO.wav",  # <-- ！！！ 修改我 ！！！
        "/PATH/TO/YOUR/MULTI_SPEAKER_AUDIO.wav"    # <-- ！！！ 修改我 ！！！
    ]
    
    print("\n" + "="*30)
    print("开始运行说话人检测...")
    print("="*30)
    
    for file_path in test_files:
        try:
            is_multi = contains_multiple_speakers(file_path)
            print(f"--- 最终结果 ---")
            print(f"文件: '{os.path.basename(file_path)}'")
            print(f"是否包含多个说话人? {is_multi}")
            print("="*30)
        except Exception as e:
             print(f"运行示例时发生意外错误: {e}")
             print("="*30)
