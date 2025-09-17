import yaml
import os
import sys

# --- 配置 ---
# ！！！ 关键步骤 ！！！
# 请在这里填入您解压的 *三个模型* 的 *config.yaml* 文件的 *绝对路径*
# (这些路径与您在上一个脚本 v3 中填写的相同)

# 1. 主流水线 (diarization-3.1) 的 config.yaml 路径
MAIN_CONFIG_PATH = "/PATH/TO/pyannote_speaker-diarization-3.1/config.yaml"

# 2. 分割模型 (segmentation-3.0) 的 config.yaml 路径
SEGMENTATION_CONFIG_PATH = "/PATH/TO/pyannote_segmentation-3.0/config.yaml"

# 3. 嵌入模型 (wespeaker) 的 config.yaml 路径
EMBEDDING_CONFIG_PATH = "/PATH/TO/pyannote_wespeaker-voxceleb-resnet34-LM/config.yaml"
# -------------------------------------------------------------------

print("--- Pyannote 离线配置补丁脚本 ---")

# 检查路径是否被修改
if "PATH/TO/" in MAIN_CONFIG_PATH:
    print("\n" + "="*50)
    print("错误: 请在脚本顶部修改 'MAIN_CONFIG_PATH' 等三个路径变量！")
    print("="*50 + "\n")
    sys.exit(1)

# 检查所有文件是否存在
for path in [MAIN_CONFIG_PATH, SEGMENTATION_CONFIG_PATH, EMBEDDING_CONFIG_PATH]:
    if not os.path.isfile(path):
        print(f"\n错误: 路径配置错误，找不到文件: {path}")
        print("请确保脚本顶部的三个路径都指向了正确的 config.yaml 文件。")
        sys.exit(1)
    else:
        print(f"找到配置文件: {path}")

try:
    print(f"\n正在加载主配置文件: {MAIN_CONFIG_PATH}")
    # 我们需要使用 CLoader 来保留注释和顺序，但 safe_load 更标准
    # 为简单起见，我们使用 safe_load，因为 pyannote 的 config 很简单
    with open(MAIN_CONFIG_PATH, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    print("加载成功。")

    # --- 核心修改 ---
    # 我们在内存中修改 Python 字典
    print("正在应用离线路径补丁...")
    
    # 根据您之前的反馈，修改这两个键的值
    config_data['pipeline']['params']['segmentation'] = SEGMENTATION_CONFIG_PATH
    config_data['pipeline']['params']['embedding'] = EMBEDDING_CONFIG_PATH
    
    print("补丁应用成功。")
    # ------------------

    # --- 写回文件 ---
    # 我们将修改后的字典写回到 *同一个* 文件，覆盖它
    print(f"正在将修改后的配置写回: {MAIN_CONFIG_PATH}")
    with open(MAIN_CONFIG_PATH, 'w', encoding='utf-8') as f:
        # Dumper=yaml.SafeDumper 保证了良好的输出格式
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, Dumper=yaml.SafeDumper)
    
    print("\n" + "="*50)
    print("      ！！！ 成 功 ！！！")
    print("您的主 config.yaml 文件已被成功修补，已指向本地依赖。")
    print("现在请运行 'check_speakers_offline_v2.py' 脚本。")
    print("="*50)

except yaml.YAMLError as e:
    print(f"\n--- YAML 错误 ---")
    print(f"读取 {MAIN_CONFIG_PATH} 时出错: {e}")
    print("这可能是因为您之前手动修改时损坏了它。")
    print("请从 .zip 压缩包中恢复一个'干净'的 'config.yaml' 再试一次。")
except KeyError as e:
    print(f"\n--- 键错误 (KeyError) ---")
    print(f"在 config.yaml 中找不到键: {e}")
    print("这说明 config.yaml 的结构与预期不符。请确保 MAIN_CONFIG_PATH 指向的是 '...diarization-3.1' 的配置文件。")
except Exception as e:
    print(f"\n--- 发生未知错误 ---")
    print(f"{e}")
