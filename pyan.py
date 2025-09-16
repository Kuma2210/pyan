import torch
from pyannote.audio import Pipeline
import warnings

# 忽略一些 torchaudio 相关的警告，使输出更干净
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# --- 配置 ---
# !!! 替换为你自己的 Hugging Face Access Token !!!
# (从 https://huggingface.co/settings/tokens 获取)
HF_AUTH_TOKEN = "YOUR_HUGGING_FACE_TOKEN_GOES_HERE"

# 自动选择设备：如果
# 1. 有NVIDIA GPU，使用 'cuda'
# 2. 有Apple Silicon (M1/M2/M3)，使用 'mps'
# 3. 否则，使用 'cpu'
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("检测到 Apple Silicon (MPS)，将使用 'mps' 设备。")
        print("注意: 首次运行时，MPS 可能需要几分钟来编译内核。")
        return torch.device("mps")
    else:
        print("未检测到 CUDA 或 MPS，将使用 'cpu'。")
        print("警告: 使用 CPU 进行推理会非常慢。")
        return torch.device("cpu")

DEVICE = get_device()

print(f"正在使用设备: {DEVICE}")

# --- 加载 SOTA 流水线 ---
# 我们使用 pyannote/speaker-diarization-3.1，这是当前的SOTA模型
# 它会自动完成：语音活动检测 -> 说话人嵌入提取 -> 聚类
print("正在加载说话人日志流水线 (pyannote/speaker-diarization-3.1)...")
try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_AUTH_TOKEN
    )
    pipeline.to(DEVICE)
    print("流水线加载成功。")
except Exception as e:
    print(f"加载流水线失败: {e}")
    print("\n--- 请确保: ---")
    print(f"1. 你的 HF_AUTH_TOKEN '{HF_AUTH_TOKEN[:4]}...{HF_AUTH_TOKEN[-4:]}' 是正确的。")
    print("2. 你已经在 Hugging Face 网站上同意了 'pyannote/speaker-diarization-3.1' 模型的使用条款。")
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
        # print("--- 详细日志 ---")
        # for turn, _, speaker in diarization.itertracks(yield_label=True):
        #     print(f"时间: [{turn.start:04.1f}s - {turn.end:04.1f}s] 说话人: {speaker}")
        # print("------------------")

        # 如果说话人数量 > 1，则返回 True
        return num_speakers > 1

    except Exception as e:
        print(f"处理文件 {audio_file_path} 时出错: {e}")
        # 如果出错，保守地返回 False（或根据需要抛出异常）
        return False

# --- 示例用法 ---
if __name__ == "__main__":
    
    # *** 替换为你自己的音频文件路径 ***
    # 您可以准备一个单人说话的音频 (test_audio_single.wav)
    # 和一个多人说话的音频 (test_audio_multi.wav) 来测试
    
    test_files = [
        "YOUR_PATH_TO_SINGLE_SPEAKER_AUDIO.wav",  # 替换我
        "YOUR_PATH_TO_MULTI_SPEAKER_AUDIO.wav"    # 替换我
    ]

    # 检查您的 HF_AUTH_TOKEN 是否已更改
    if HF_AUTH_TOKEN == "YOUR_HUGGING_FACE_TOKEN_GOES_HERE":
        print("\n" + "="*50)
        print("错误: 请在脚本顶部设置您的 HF_AUTH_TOKEN 变量！")
        print("请访问 https://huggingface.co/settings/tokens 获取。")
        print("="*50)
    else:
        for file_path in test_files:
            try:
                # 检查文件是否存在
                with open(file_path, 'rb') as f:
                    pass
                
                # 运行主函数
                is_multi = contains_multiple_speakers(file_path)
                
                print(f"结果: 文件 '{file_path}' { '包含' if is_multi else '不包含' } 多个说话人。")

            except FileNotFoundError:
                print(f"\n错误: 示例文件 '{file_path}' 未找到。")
                print("请修改脚本中的 'test_files' 列表，指向您本地的音频文件。")
            except Exception as e:
                print(f"运行示例时发生意外错误: {e}")
