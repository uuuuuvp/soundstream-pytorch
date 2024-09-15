dependencies = ['torch']
# flake8: noqa: E402
from soundstream import Encoder, Decoder, ResidualVectorQuantizer
import torch
import torch.nn as nn

# python 语法， 变量名：类型
class EncoderDecoder(nn.Module):
    def __init__(
        self,
        n_channels: int = 32,   # 变量名： 类型 = 赋值
        num_quantizers: int = 8,
        num_embeddings: int = 1024,
        padding: str = "valid"  # padding: str = "valid"  不补零
    ):
        super().__init__()
        self.encoder = Encoder(n_channels, padding) # 通过调用来创建实例
        self.decoder = Decoder(n_channels, padding)
        self.quantizer = ResidualVectorQuantizer(
            num_quantizers, num_embeddings, n_channels * 16)# 调用RVQ 来实现 quantizer
        

    def forward(self, x):
        return self.encode(x)

    # 接收的是 input ,input: torch.Tensor 意为：input为 张量
    def encode(self, input: torch.Tensor) -> torch.Tensor:  # 返回类类型为 torch.Tensor 
        assert input.ndim == 2
        x = torch.unsqueeze(input, 1)   # torch.unsqueeze(input,dim)表示升维 在dim处升维
        x = self.encoder(x)
        x = torch.transpose(x, -1, -2)  # torch.transpose()交换两个维度
        _, codes, _ = self.quantizer(x)
        return codes    # 返回值为量化 结果
    def enco(self, input : torch.Tensor) -> torch.Tensor:
        assert input.ndim == 2
        x = torch.unsqueeze(input, 1)
        x = self.encoder(x)
        return x
    
    def quan(self, input: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(input, -1, -2)
        _, codes, _ = self.quantizer(x)
        return codes

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        # input: [batch_size, length, num_quantizers]
        x = self.quantizer.dequantize(input)
        x = torch.transpose(x, -1, -2)
        x = self.decoder(x)
        x = torch.squeeze(x, 1)
        return x # 返回值为 解得的 speech整数 list

    def dequan(self, input: torch.Tensor) -> torch.Tensor:
        x = self.quantizer.dequantize(input)
        return x
    
    def deco(self, input: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(input, -1, -2)
        x = self.decoder(x)
        x = torch.squeeze(x, 1)
        return x


def soundstream_16khz(pretrained=True, **kwargs): # pretrained=False
    """SoundStream encoder decoder
    
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = EncoderDecoder()
    state_dict = torch.hub.load_state_dict_from_url("https://github.com/kaiidams/soundstream-pytorch/releases/download/v1.0/soundstream_16khz-20230425.ckpt", map_location='cpu')
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model.eval()
    return model

def from_ckpt_load_model():
    # 创建自定义的 prog 模型实例
    model = EncoderDecoder()
    # 加载参数
    checkpoint = torch.load("/data0/youyubo/.cache/torch/hub/checkpoints/soundstream_16khz-20230425.ckpt", map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # 设置模型为评估模式
    model.eval()

    return model