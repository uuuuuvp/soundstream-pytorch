from hubconf import from_ckpt_load_model
import torchaudio
import torch
# 定义一个预训练模型 prog
prog = from_ckpt_load_model();
# 定义一个函数， 编码量化解码
def encode_decode_speech(input_wav, output_wav):

    x, sr = torchaudio.load(input_wav)
    with torch.no_grad():
        y = prog.encode(x)
        z = prog.decode(y)
    torchaudio.save(output_wav, z, sr)

