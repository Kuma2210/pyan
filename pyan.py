import torch
from pyannote.audio import Pipeline
import warnings
import os # 导入 os 库

# 忽略一些 torchaudio 相关的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# --- 配置 ---

def get_device():
    """
    自动检测可用的最佳计算设备 (GPU > Apple Silicon > CPU)
    """
    if torch.cuda.is_available():
        print("检测到 NVIDIA GPU (CUDA)，将使用 'cuda' 设备。")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("检测到 Apple Silicon (MPS)，将使用 'mps' 设备。")
        return torch.device("mps")
    else:
        print("未检测到 CUDA 或 MPS，将使用 'cpu'。")
        print("警告: 使用 CPU 进行推理会非常慢。")
        return torch.device("cpu")

DEVICE = get_device()

# --- 加载 SOTA 流水线 (离线方式) ---

# ！！！ 关键修改 ！！！
#
# 1. 路径现在必须指向 *文件夹内部* 的 'config.yaml' 文件
#
# 示例:
# 如果你的文件夹路径是: /home/your_user/models/pyannote_speaker-diarization-3.1
# 那么这里就填:
# LOCAL_MODEL_CONFIG_FILE = "/home/your_user/models/pyannote_speaker-diarization-3.1/config.yaml"
#
LOCAL_MODEL_CONFIG_FILE = "/PATH/TO/YOUR/UNZIPPED/MODEL/FOLDER/config.yaml" # <-- ！！！ 修改我 ！！！


print(f"正在从本地配置文件加载流水线: {LOCAL_MODEL_CONFIG_FILE} ...")

# 检查路径是否被修改
if "YOUR/UNZIPPED/MODEL/FOLDER" in LOCAL_MODEL_CONFIG_FILE:
    print("\n" + "="*50)
    print("错误: 请在脚本中修改第 36 行的 'LOCAL_MODEL_CONFIG_FILE' 变量！")
    print("它必须指向您解压的模型文件夹 *内部* 的 'config.yaml' 文件。")
    print("="*50 + "\n")
    exit(1)

# 检查 *文件* 是否存在
if not os.path.isfile(LOCAL_MODEL_CONFIG_FILE):
    print(f"\n错误: 配置文件 '{LOCAL_MODEL_CONFIG_FILE}' 未找到。")
    print("请确保您的路径正确，并且它指向的是 'config.yaml' 文件，*不是* 文件夹。")
    print(f"请检查这个路径：{LOCAL_MODEL_CONFIG_FILE}")
    exit(1)

try:
    # 关键改动：现在加载的是 .yaml 文件路径
    pipeline = Pipeline.from_pretrained(
        LOCAL_MODEL_CONFIG_FILE
    )
    pipeline.to(DEVICE)
    print("流水线从本地加载成功。")

except Exception as e:
    print(f"从本地加载流水线失败: {e}")
    print("\n--- 请确保: ---")
    print(f"1. 路径 '{LOCAL_MODEL_CONFIG_FILE}' 是正确的。")
    print("2. 您的 pyannote.audio 库已正确安装。")
    print("------------------\n")
    exit(1)


def contains_multiple_speakers(audio_file_path: str) -> bool:
    """
    分析一个音频文件，判断是否包含多个说话人。
    ... (此函数与上一版本完全相同) ...
    """
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

    except FileNotFoundError:
        print(f"错误: 音频文件未找到于 '{audio_file_path}'")
        return False
    except Exception as e:
        print(f"处理文件 {audio_file_path} 时出错: {e}")
        return False

# --- 示例用法 ---
if __name__ == "__main__":
    
    # *** 替换为你自己的音频文件路径 ***
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
            if "未找到" not in str(e):
                print(f"运行示例时发生意外错误: {e}")
            print("="*30)
