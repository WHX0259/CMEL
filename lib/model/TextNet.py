"""
这个代码就是将形状为(batch_size,1,input_dim)的输入进行特征提取得到(batch_size, hidden_dim,1)的输出
"""
# from models.BaseModel import BaseModel
import torch
import torch.nn as nn
# from sarcopenia_data.SarcopeniaDataLoader import TEXT_COLS
from .BaseModel import BaseModel
class TextNetFeature(BaseModel):
    def __init__(self, backbone, n_channels, num_classes, pretrained, input_dim):
        super(TextNetFeature, self).__init__(backbone, n_channels, num_classes, pretrained)
        in_planes2 = input_dim
        if n_channels==0:
            self.hidden = [64]
        else:
            self.hidden = n_channels

        self.num = nn.Conv1d(in_planes2, self.hidden[-1], kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(self.hidden[-1])
        self.silu2 = nn.SiLU(inplace=True)
        self.fc = nn.Linear(self.hidden[-1], self.hidden[-1])
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, num_x2):
        num_x = self.num(num_x2.permute(0, 2, 1))
        num_x = self.silu2(self.bn2(num_x))#4, 64, 3
        num_x = num_x.squeeze(-1)
        num_x = self.fc(num_x).unsqueeze(-1)
        return num_x

if __name__ == '__main__':


    # 定义输入维度和批量大小
    input_dim = 22  # 输入特征的维度
    batch_size = 4  # 批量大小
    n_channels = 3  # 通道数
    num_classes = 2  # 分类数目

    # 创建一个 TextNetFeature 实例
    textnet = TextNetFeature('none', n_channels, num_classes, pretrained=False, input_dim=input_dim)

    # 将模型设置为评估模式（如果需要）
    textnet.eval()

    # 创建随机输入数据
    # 假设我们有 batch_size 个样本，每个样本有 input_dim 维度的特征
    random_input = torch.randn(batch_size, 1, input_dim)

    # 进行前向传播
    with torch.no_grad():  # 不需要计算梯度
        output = textnet(random_input)
        
    # 输出结果
    print("Output shape:", output.shape)  # 应该是 [batch_size, hidden[-1], 1]
    print("Output:", output)

    # 如果你知道期望的输出形状或值，你可以在这里添加断言来验证输出
    assert output.shape == (batch_size, textnet.hidden[-1], 1), "输出形状不正确"