import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def get_model(model_str: str):
    """모델 클래스 변수 설정
    Args:
        model_str (str): 모델 클래스명
    Note:
        model_str 이 모델 클래스명으로 정의돼야함
        `model` 변수에 모델 클래스에 해당하는 인스턴스 할당
    """
    if model_str == 'resnet':
        return ResNet
    else:
        print(model_str)
        return timm.create_model(model_str,pretrained=True)
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=True):
        super().__init__()

        layers = []
        

        ##fill it##
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias))
        if norm == "bnorm":
            layers.append(nn.BatchNorm2d(out_channels))
        
        if relu:
            layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)
        


    def forward(self, x):

        ##fill it##
        return self.conv(x)
    
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=True, norm="bnorm", short_cut=False, relu=True, init_block=False):
        super().__init__()

        layers = []

        ## Channel dimension을 늘림과 동시에 height와 width는 절반으로 줄어들어야 하므로
        ## stride=2를 가진 convolutional layer가 각 residual block 시작에 존재해야함.
        if init_block:
          init_stride = 2
        else:
          init_stride = stride

        ##fill##
        # print(in_channels, out_channels, kernel_size, init_stride, padding, bias, norm,relu)
        layers.append(ConvBlock(in_channels, out_channels, kernel_size, init_stride, padding, bias, norm,relu))
        layers.append(ConvBlock(out_channels, out_channels, kernel_size, stride, padding, bias, norm,False))

        self.resblk = nn.Sequential(*layers)
        ##Projection shortcut connection에 해당하는 1x1 convolution도 함께 정의해주세요.##
        ## stride를 init_stirde를 해야할지 2로 해야할지 결정
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=(1,1),stride = 2),
            nn.BatchNorm2d(out_channels)
        )
            
        self.relu = nn.ReLU()



    def forward(self, x, short_cut=False):

        ##fill##
        ##Residual Block의 Input과 Output의 Channel Dimension이 서로 다른 경우, Projection shortcut connection을 이용하여 차원을 맞춰주세요.##
        if short_cut:
          return F.relu(self.short_cut(x) + self.resblk(x))
        else:
          return F.relu(x + self.resblk(x))
        # if short_cut:
        #   return self.short_cut(x) + self.resblk(x)
        # else:
        #   return x + self.resblk(x)



class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm", nblk=[3,4,6,3]):
        super(ResNet, self).__init__()

        self.enc = ConvBlock(in_channels, nker, kernel_size=7, stride=2, padding=1, bias=True, norm=None, relu=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        
        
        ##fill##
        kener_size =3
        stride = 1
        padding = 1
        bias = True
        relu = True
        
        self.resblock_1_0 = ResBlock(in_channels=nker,out_channels=nker*2,kernel_size=kener_size,stride=stride, padding=padding,bias=bias,norm=norm,relu=True,init_block=True)
        self.resblock_2_0 = ResBlock(in_channels=nker*2,out_channels=nker*4,kernel_size=kener_size,stride=stride, padding=padding,bias=bias,norm=norm,relu=True,init_block=True)
        self.resblock_3_0 = ResBlock(in_channels=nker*4,out_channels=nker*8,kernel_size=kener_size,stride=stride, padding=padding,bias=bias,norm=norm,relu=True,init_block=True)
        
        self.resblock_0_1 = nn.Sequential(*[ResBlock(in_channels=nker,out_channels=nker,kernel_size=kener_size,stride=stride, padding=padding,bias=bias,norm=norm,relu=True) for _ in range(nblk[0])])
        self.resblock_1_1 = nn.Sequential(*[ResBlock(in_channels=nker*2,out_channels=nker*2,kernel_size=kener_size,stride=stride, padding=padding,bias=bias,norm=norm,relu=True) for _ in range(nblk[1]-1)])
        self.resblock_2_1 = nn.Sequential(*[ResBlock(in_channels=nker*4,out_channels=nker*4,kernel_size=kener_size,stride=stride, padding=padding,bias=bias,norm=norm,relu=True) for _ in range(nblk[2]-1)])
        self.resblock_3_1 = nn.Sequential(*[ResBlock(in_channels=nker*8,out_channels=nker*8,kernel_size=kener_size,stride=stride, padding=padding,bias=bias,norm=norm,relu=True) for _ in range(nblk[3]-1)])



        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(nker*2*2*2, 18)

    def forward(self, x):
        x = self.enc(x)
        x = self.max_pool(x)


        ##fill##
        x =self.resblock_0_1(x)
        # x = self.max_pool(x)
        
        x =self.resblock_1_0(x, short_cut=True)
        x =self.resblock_1_1(x)
        # x = self.max_pool(x)
        
        x =self.resblock_2_0(x, short_cut=True)
        x =self.resblock_2_1(x)
        # x = self.max_pool(x)
        
        x =self.resblock_3_0(x, short_cut=True)
        x =self.resblock_3_1(x)

        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        out = self.fc(x)

        return out
    
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet1(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet1, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
