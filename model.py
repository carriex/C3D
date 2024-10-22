import torch.nn as nn
import torch.nn.functional as F



class C3D(nn.Module):
    def __init__(self, num_classes):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, 3, padding=1)
        self.pool1 = nn.MaxPool3d((1, 2, 2), ceil_mode=True)
        self.conv2 = nn.Conv3d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool3d((2, 2, 2), ceil_mode=True)
        self.conv3 = nn.Conv3d(128, 256, 3, padding=1)
        self.pool3 = nn.MaxPool3d((2, 2, 2), ceil_mode=True)
        self.conv4 = nn.Conv3d(256, 256, 3, padding=1)
        self.pool4 = nn.MaxPool3d((2, 2, 2), ceil_mode=True)
        self.conv5 = nn.Conv3d(256, 256, 3, padding=1)
        self.pool5 = nn.MaxPool3d((2, 2, 2), ceil_mode=True)
        self.fc1 = nn.Linear(256 * 1 * 4 * 4, 2048)  # 2048/4096
        self.fc2 = nn.Linear(2048, 2048)
        self.out = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.init_weights()

    def forward(self, x):

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(-1, 256 * 1 * 4 * 4)
        x = F.relu(self.fc1(x))
        d1_x = self.dropout(x)
        x = F.relu(self.fc2(d1_x))
        d2_x = self.dropout(x)
        x = self.out(d2_x)

        return x

    def init_weights(self):

        for name, m in self.named_modules():
            if type(m) == nn.Conv3d:
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 1.0)
            if type(m) == nn.Linear:
                if name == 'out':
                    nn.init.constant_(m.bias, 0.0)
                    nn.init.normal_(m.weight, std=0.005)
                else:
                    nn.init.constant_(m.bias, 1.0)
                    nn.init.normal_(m.weight, std=0.005)

    def get_1x_lr_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.requires_grad:
                yield param

    def get_2x_lr_params(self):
        for name, param in self.named_parameters():
            if 'bias' in name and param.requires_grad:
                yield param
