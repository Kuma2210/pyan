import torch
from pyannote.audio import Pipeline
import warnings
import os # 导入 os 库来检查文件

# 忽略一些 torchaudio 相关的警告，使输出更干净
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# --- 配置 ---

def get_device():
    """
    自动检测可用的最佳计算设备 (GPU > Apple Silicon > CPU)
    """
    if torch.cuda.is_available():
        print("检测到 NVIDIA GPU (CUDA)，将使用 'cuda' 设备。")
        return torch.device("cuda")
    # torch.backends.mps.is_available() 仅在 PyTorch 1.12+ 和 macOS 12.3+ 上可用
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("检测到 Apple Silicon (MPS)，将使用 'mps' 设备。")
        print("注意: 首次运行时，MPS 可能需要几分钟来编译内核。")
        return torch.device("mps")
    else:
        print("未检测到 CUDA 或 MPS，将使用 'cpu'。")
        print("警告: 使用 CPU 进行推理会非常慢。")
        return torch.device("cpu")

DEVICE = get_device()

# --- 加载 SOTA 流水线 (离线方式) ---

# ！！！ 关键步骤 ！！！
# 请将此路径替换为您在公司平台上解压模型后的 *绝对路径*
#
# 例如:
# 如果您解压到了 /home/your_user/models/pyannote_speaker-diarization-3.1
# 那么这里就填:
# LOCAL_MODEL_PATH = "/home/your_user/models/pyannote_speaker-diarization-3.1"
#
LOCAL_MODEL_PATH = "/PATH/TO/YOUR/UNZIPPED/MODEL/FOLDER" # <-- ！！！ 修改我 ！！！


print(f"正在从本地路径加载流水线: {LOCAL_MODEL_PATH} ...")

# 检查路径是否被修改
if LOCAL_MODEL_PATH == "/PATH/TO/YOUR/UNZIPPED/MODEL/FOLDER":
    print("\n" + "="*50)
    print("错误: 请在脚本中修改第 31 行的 'LOCAL_MODEL_PATH' 变量！")
    print("它必须指向您解压的模型文件夹的真实路径。")
    print("="*50 + "\n")
    exit(1)

# 检查路径是否存在
if not os.path.isdir(LOCAL_MODEL_PATH):
    print(f"\n错误: 路径 '{LOCAL_MODEL_PATH}' 不存在或不是一个文件夹。")
    print("请检查您的路径是否正确。")
    exit(1)

try:
    # 关键改动：
    # 1. from_pretrained() 的第一个参数现在是本地文件夹路径
    # 2. 没有 use_auth_token=... 参数，完全离线
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
    print("------------------\n")
    exit(1)


def contains_multiple_speakers(audio_file_path: str) -> bool:
    """
    分析一个音频文件，判断是否包含多个说话人。

    参数:
    audio_file_path (str): 指向音频文件的路径 (wav, mp3, flac 等)

    返回:
    bool: True 如果检测到多个说话人, False 否则。
    """
    print(f"\n正在处理文件: {audio_file_path}")
    try:
        # 运行流水线
        # 'diarization' 是一个 pyannote.core.Annotation 对象
        diarization = pipeline(audio_file_path)

        # .labels() 方法返回在此标注中找到的所有唯一标签（即说话人ID）
        # 示例: ['SPEAKER_00', 'SPEAKER_01']
        speaker_labels = diarization.labels()

        # 计算唯一说话人的数量
        num_speakers = len(speaker_labels)

        print(f"检测到的唯一说话人数量: {num_speakers}")
        
        # 打印详细的日志 (可选, 如果您想看具体时间)
        print("--- 详细日志 (可选) ---")
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"时间: [{turn.start:04.1f}s - {turn.end:04.1f}s] 说话人: {speaker}")
        print("------------------------")

        # 如果说话人数量 > 1，则返回 True
        return num_speakers > 1

    except FileNotFoundError:
        print(f"错误: 音频文件未找到于 '{audio_file_path}'")
        return False
    except Exception as e:
        print(f"处理文件 {audio_file_path} 时出错: {e}")
        # 如果出错，保守地返回 False（或根据需要抛出异常）
        return False

# --- 示例用法 ---
if __name__ == "__main__":
    
    # *** 替换为你自己的音频文件路径 ***
    # 您可以准备一个单人说话的音频
    # 和一个多人说话的音频来测试
    
    test_files = [
        "/PATH/TO/YOUR/SINGLE_SPEAKER_AUDIO.wav",  # <-- ！！！ 修改我 ！！！
        "/PATH/TO/YOUR/MULTI_SPEAKER_AUDIO.wav"    # <-- ！！！ 修改我 ！！！
    ]

    print("\n" + "="*30)
    print("开始运行说话人检测...")
    print("="*30)

    for file_path in test_files:
        try:
            # 运行主函数
            is_multi = contains_multiple_speakers(file_path)
            
            print(f"--- 最终结果 ---")
            print(f"文件: '{os.path.basename(file_path)}'")
            print(f"是否包含多个说话人? {is_multi}")
            print("="*30)

        except Exception as e:
            # 捕获 contains_multiple_speakers 中 FileNotFoundError 之外的错误
            if "未找到" not in str(e): # 避免重复打印文件未找到
                print(f"运行示例时发生意外错误: {e}")
                print("请检查音频文件路径和格式是否正确。")
            print("="*30)
