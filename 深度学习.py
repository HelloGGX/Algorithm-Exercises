import torch
import torch.nn as nn

conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)

print("--- 侦探的随机“线索图案”(卷积核权重) ---")
print(conv_layer.weight)

# 2. 手动创建一个 5x5 的图片，包含 'X' 图案
image = torch.tensor([
    [1., 0., 1., 0., 0.],
    [0., 1., 0., 1., 0.],
    [1., 0., 1., 0., 1.],
    [0., 1., 0., 1., 0.],
    [0., 0., 1., 0., 1.]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

print("\n--- 输入图片 (5x5) ---")
print(image.squeeze())

# 3. 让图片通过卷积层，得到特征图
feature_map = conv_layer(image)

print("\n--- 输出的特征图 (3x3) ---")
print(feature_map.squeeze())


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc = nn.Linear(in_features=32 * 7 * 7, out_features=10)  # 假设输入图片是28x28

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)  # 展平
        x = self.fc(x)
        return x