import torch
import torch.nn as nn



class ModalEncoder(nn.Module):
    def __init__(self, num_class = 7):
        super().__init__()
        self.conv1 =self.make_block(in_channels = 1, out_channels = 64)
        self.conv2 =self.make_block(in_channels = 64, out_channels = 128)
        self.conv3 =self.make_block(in_channels = 128, out_channels = 256)
        self.conv4 =self.make_block(in_channels = 256, out_channels = 512)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features = 32768, out_features = 512),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features = 512, out_features = 1024),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features = 1024, out_features = num_class),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels= in_channels, out_channels = out_channels, kernel_size = (3,3),stride=1, padding= "same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels= out_channels, out_channels = out_channels, kernel_size = (3,3),stride=1, padding= "same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )   
    
if __name__ == "__main__":
    modal = ModalEncoder()
    input_data = torch.randn(1,1,128,128)
    if torch.cuda.is_available():
        modal = modal.cuda()
        input_data = input_data.cuda()

    result = modal(input_data)
    print(result)