from program import encode_decode_speech
from hubconf import from_ckpt_load_model
import torchaudio
import torch

prog = from_ckpt_load_model();

x, sr = torchaudio.load('908-31957-0025.wav')
x, sr = torchaudio.functional.resample(x, sr, 16000), 16000
with torch.no_grad():
    print(x)
    print(x.shape)
    # y = prog.encode(x)
    # print(y)
    y = prog.enco(x)
    print(y)
    print(y.shape)
    y = prog.quan(y)
    print(y)
    print(y.shape)
    z = prog.dequan(y)
    print(z)
    print(z.shape)
    z =prog.deco(z)
    print(z)
    print(z.shape)
torchaudio.save('output.wav', z, sr)