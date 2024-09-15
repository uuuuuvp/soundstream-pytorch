from program import encode_decode_speech
import os

# 定义输入和输出文件夹路径
input_folder = 'ls16k'
output_folder = 'lsout16'

# 获取输入文件夹中的所有音频文件
audio_files = os.listdir(input_folder)

# 逐个处理音频文件
for audio_file in audio_files:
    # 构建输入和输出文件路径
    input_file = os.path.join(input_folder, audio_file)
    output_file = os.path.join(output_folder, audio_file)

    # 调用函数进行编码量化解码
    encode_decode_speech(input_file, output_file)