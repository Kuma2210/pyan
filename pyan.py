import torch
from pyannote.audio import Pipeline
import warnings
import os

# --- 配置 ---

# ！！！ 关键步骤 ！！！
# 请在这里填入您解压的 *三个模型* 的 *config.yaml* 文件的 *绝对路径*

# 1. 主流水线 (diarization-3.1) 的 config.yaml 路径
MAIN_CONFIG_PATH = "/PATH/TO/pyannote_speaker-diarization-3.1/config.yaml"

# 2. 分割模型 (segmentation-3.0) 的 config.yaml 路径
SEGMENTATION_CONFIG_PATH = "/PATH/TO/pyannote_segmentation-3.0/config.yaml"

# 3. 嵌入模型 (wespeaker) 的 config.yaml 路径 (根据您之前的反馈)
EMBEDDING_CONFIG_PATH = "/PATH/TO/pyannote_wespeaker-voxceleb-resnet34-LM/config.yaml"

# -------------------------------------------------------------------

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

def get_device():
    """自动检测最佳设备"""
    if torch.cuda.is_available():
        print("检测到 CUDA，将使用 'cuda'。")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("检测到 Apple Silicon (MPS)，将使用 'mps'。")
        return torch.device("mps")
    else:
        print("未检测到 GPU，将使用 'cpu' (较慢)。")
        return torch.device("cpu")

DEVICE = get_device()

print("--- 正在加载流水线 (离线覆盖模式) ---")

# 检查路径是否被修改
if "PATH/TO/" in MAIN_CONFIG_PATH:
    print("\n" + "="*50)
    print("错误: 请在脚本顶部修改 'MAIN_CONFIG_PATH' 等三个路径变量！")
    print("="*50 + "\n")
    exit(1)

# 检查所有文件是否存在
for path in [MAIN_CONFIG_PATH, SEGMENTATION_CONFIG_PATH, EMBEDDING_CONFIG_PATH]:
    if not os.path.isfile(path):
        print(f"\n错误: 路径配置错误，找不到文件: {path}")
        print("请确保脚本顶部的三个路径都指向了正确的 config.yaml 文件。")
        exit(1)
    else:
        print(f"找到配置文件: {path}")

try:
    # ！！！ 最终解决方案 ！！！
    # 我们加载 *原始* 的主 config.yaml 文件...
    # ...然后 *在 Python 内存中* 动态覆盖它的依赖项！
    
    pipeline = Pipeline.from_pretrained(
        MAIN_CONFIG_PATH,
        
        # --- 参数覆盖 ---
        # 这里的参数名 (segmentation, embedding)
        # 必须与主 config.yaml 中 'params:' 下的键完全一致
        
        # 强制 pyannote 使用 *本地* 的分割模型
        segmentation=SEGMENTATION_CONFIG_PATH,
        
        # 强制 pyannote 使用 *本地* 的嵌入模型
        embedding=EMBEDDING_CONFIG_PATH
        # -----------------
    )
    
    pipeline.to(DEVICE)
    print("\n流水线从本地加载成功！（已覆盖依赖项）")

except Exception as e:
    print(f"\n从本地加载流水线失败: {e}")
    print("请仔细检查脚本顶部的三个路径是否完全正确。")
    print("------------------\n")
    exit(1)


def contains_multiple_speakers(audio_file_path: str) -> bool:
    """
    分析一个音频文件，判断是否包含多个说话人。
    (此函数与上一版本完全相同)
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
