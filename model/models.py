import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models


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
    elif model_str == "efficientnet_b3a":
        return timm.create_model
    elif model_str == 'pre_resnet34':
        return pre_ResNet34
    elif model_str == 'pre_resnet50':
        return pre_ResNet50
    elif model_str == 'eifficientnet_b3':
        return Efficientnet_b3
    elif model_str == 'eifficientnet_b4':
        return Efficientnet_b4
    elif model_str == 'eifficientnet_b5':
        return Efficientnet_b5
    elif model_str == 'eifficientnet_b6':
        return Efficientnet_b6
    elif model_str == 'convnext_base':
        return ConvNext_base
    elif model_str == 'convnext_large':
        return ConvNext_large 
    elif model_str == 'vit_b_16':
        return ViT_B_16  
    elif model_str == 'swintransformer_t':
        return SwinTransformer_t      
    elif model_str == 'swintransformer_b':
        return SwinTransformer_b     
    elif model_str == 'mobilenet_v3_small':
        return Mobilenet_v3_S     
    elif model_str == 'mobilenet_v3_large':
        return Mobilenet_v3_L     
    elif model_str == 'mobileone_s4':
        return Mobileone_s4  
    elif model_str == 'coatnet_0_rw_224':
        return CoAtnet_rw_224      
    elif model_str == 'tinynet_c':
        return TinyNet_c  
    elif model_str == 'tinynet_e':
        return TinyNet_E
    elif model_str == 'tf_efficientnet_b5.ns_jft_in1k':
        return tf_efficientnet_b5_ns
    elif model_str == 'tf_efficientnetv2_m.in21k_ft_in1k':
        return tf_efficientnetv2_m_in21k
    elif model_str == 'caformer_b36.sail_in22k_ft_in1k':
        return caformer_b36_sail_in22k_ft_in1k
    elif model_str == 'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k':
        return swinv2_base_window12to16_192to256_ms_in22k_ft_in1k
    elif model_str == 'caformer_s36.sail_in22k_ft_in1k':
        return caformer_s36_sail_in22k_ft_in1k
    elif model_str == 'convformer_m36.sail_in22k_ft_in1k':
        return convformer_m36_sail_in22k_ft_in1k
    elif model_str == 'tiny_vit_21m_224.dist_in22k_ft_in1k':
        return tiny_vit_21m_224_dist_in22k_ft_in1k
    elif model_str == 'convnext_small.in12k_ft_in1k':
        return convnext_small_in12k_ft_in1k

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

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    

class pre_ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(pre_ResNet34, self).__init__()
        
        self.model = models.resnet34(pretrained=True)

        pre_layer = self.model.fc.in_features
        self.model.fc = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)

    def forward(self, x):
        
        x = self.model(x)
        x = self.linear(x)

        return x
    

class pre_ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(pre_ResNet50, self).__init__()
        
        self.model = models.resnet50(pretrained=True)

        pre_layer = self.model.fc.in_features
        self.model.fc = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)

    def forward(self, x):
        
        x = self.model(x)
        x = self.linear(x)

        return x
    

class pre_ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(pre_ResNet50, self).__init__()
        
        self.model = models.resnet50(pretrained=True)

        pre_layer = self.model.fc.in_features
        self.model.fc = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)

    def forward(self, x):
        
        x = self.model(x)
        x = self.linear(x)

        return x
    

class Efficientnet_b4(nn.Module):
    def __init__(self, num_classes):
        super(Efficientnet_b4, self).__init__()
        
        self.model = models.efficientnet_b4(pretrained=True)
        pre_layer = self.model.classifier[1].in_features
        self.model.classifier = Identity()

        self.linear = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(pre_layer, num_classes)
        )

    def forward(self, x):
        
        x = self.model(x)
        x = self.linear(x)

        return x


class Efficientnet_b3(nn.Module):
    def __init__(self, num_classes):
        super(Efficientnet_b3, self).__init__()
        
        self.model = models.efficientnet_b3(pretrained=True)
        pre_layer = self.model.classifier[1].in_features
        self.model.classifier = Identity()

        self.linear = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(pre_layer, num_classes)
        )

    def forward(self, x):
        
        x = self.model(x)
        x = self.linear(x)

        return x
    
class Efficientnet_b5(nn.Module):
    def __init__(self, num_classes):
        super(Efficientnet_b5, self).__init__()
        
        self.model = models.efficientnet_b5(pretrained=True)
        pre_layer = self.model.classifier[1].in_features
        self.model.classifier = Identity()

        self.linear = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(pre_layer, num_classes)
        )

    def forward(self, x):
        
        x = self.model(x)
        x = self.linear(x)

        return x

class Efficientnet_b6(nn.Module):
    def __init__(self, num_classes):
        super(Efficientnet_b6, self).__init__()
        
        self.model = models.efficientnet_b6(pretrained=True)
        pre_layer = self.model.classifier[1].in_features
        self.model.classifier = Identity()

        self.linear = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(pre_layer, num_classes)
        )

    def forward(self, x):
        
        x = self.model(x)
        x = self.linear(x)

        return x


class ConvNext_base(nn.Module):
    def __init__(self, num_classes):
        super(ConvNext_base, self).__init__()
        
        self.model = models.convnext_base(pretrained=True)
        pre_layer = self.model.classifier[-1].in_features
        self.model.classifier = Identity()

        self.linear = nn.Sequential(
            nn.LayerNorm([pre_layer]),
            nn.Linear(pre_layer, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)        
        x = self.linear(x)
        return x


class ConvNext_large(nn.Module):
    def __init__(self, num_classes):
        super(ConvNext_large, self).__init__()
        
        self.model = models.convnext_large(pretrained=True)
        pre_layer = self.model.classifier[-1].in_features
        self.model.classifier = Identity()

        self.linear = nn.Sequential(
            nn.LayerNorm([pre_layer]),
            nn.Linear(pre_layer, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)        
        x = self.linear(x)
        return x

class ViT_B_16(nn.Module):
    def __init__(self, num_classes):
        super(ViT_B_16, self).__init__()
        
        self.model = models.vit_b_16(pretrained=True)
        pre_layer = self.model.heads[-1].in_features
        self.model.heads = Identity()

        self.linear = nn.Sequential(
            nn.Linear(pre_layer, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x


class SwinTransformer_b(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformer_b, self).__init__()
        
        self.model = models.swin_b(pretrained=True)
        pre_layer = self.model.head.in_features
        self.model.head = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x
    
class SwinTransformer_t(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformer_t, self).__init__()
        
        self.model = models.swin_t(pretrained=True)
        pre_layer = self.model.head.in_features
        self.model.head = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x
    

class Mobilenet_v3_L(nn.Module):
    def __init__(self, num_classes):
        super(Mobilenet_v3_L, self).__init__()
        
        self.model = models.mobilenet_v3_large(pretrained=True)
        pre_layer = self.model.classifier[-1].in_features
        self.model.classifier[-1] = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x

class Mobilenet_v3_S(nn.Module):
    def __init__(self, num_classes):
        super(Mobilenet_v3_S, self).__init__()
        
        self.model = models.mobilenet_v3_small(pretrained=True)
        pre_layer = self.model.classifier[-1].in_features
        self.model.classifier[-1] = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x 


class Mobileone_s4(nn.Module):
    def __init__(self, num_classes):
        super(Mobileone_s4, self).__init__()
        
        self.model = timm.create_model("mobileone_s4", pretrained=True)
        pre_layer = self.model.head.fc.in_features
        self.model.head.fc = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x
    

class CoAtnet_rw_224(nn.Module):
    def __init__(self, num_classes):
        super(CoAtnet_rw_224, self).__init__()
        
        self.model = timm.create_model("coatnet_0_rw_224", pretrained=True)
        pre_layer = self.model.head.fc.in_features
        self.model.head.fc = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x
    
class TinyNet_c(nn.Module):
    def __init__(self, num_classes):
        super(TinyNet_c, self).__init__()
        
        self.model = timm.create_model("tinynet_c", pretrained=True)
        pre_layer = self.model.classifier.in_features
        self.model.classifier = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x

class TinyNet_E(nn.Module):
    def __init__(self, num_classes):
        super(TinyNet_E, self).__init__()
        
        self.model = timm.create_model("tinynet_e", pretrained=True)
        pre_layer = self.model.classifier.in_features
        self.model.classifier = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x
    
class tf_efficientnet_b5_ns(nn.Module):
    def __init__(self, num_classes):
        super(tf_efficientnet_b5_ns, self).__init__()
        
        self.model = timm.create_model("tf_efficientnet_b5.ns_jft_in1k", pretrained=True)
        pre_layer = self.model.classifier.in_features
        self.model.classifier = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x
    
class tf_efficientnetv2_m_in21k(nn.Module):
    def __init__(self, num_classes):
        super(tf_efficientnetv2_m_in21k, self).__init__()
        
        self.model = timm.create_model("tf_efficientnetv2_m.in21k_ft_in1k", pretrained=True)
        pre_layer = self.model.classifier.in_features
        self.model.classifier = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x
    
class tf_efficientnet_b2_ns_jft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(tf_efficientnet_b2_ns_jft_in1k, self).__init__()
        
        self.model = timm.create_model("tf_efficientnet_b2.ns_jft_in1k", pretrained=True)
        pre_layer = self.model.classifier.in_features
        self.model.classifier = Identity()

        self.linear = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x
    
class caformer_b36_sail_in22k_ft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(caformer_b36_sail_in22k_ft_in1k, self).__init__()
        
        self.model = timm.create_model("caformer_b36.sail_in22k_ft_in1k", pretrained=True)
        pre_layer = self.model.head.fc.fc2.in_features
        self.model.head.fc.fc2 = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x

class swinv2_base_window12to16_192to256_ms_in22k_ft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(swinv2_base_window12to16_192to256_ms_in22k_ft_in1k, self).__init__()
        
        self.model = timm.create_model("swinv2_base_window12to16_192to256.ms_in22k_ft_in1k", pretrained=True)
        pre_layer = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class caformer_s36_sail_in22k_ft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(caformer_s36_sail_in22k_ft_in1k, self).__init__()
        
        self.model = timm.create_model("caformer_s36.sail_in22k_ft_in1k", pretrained=True)
        pre_layer = self.model.head.fc.fc2.in_features
        self.model.head.fc.fc2 = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class convformer_m36_sail_in22k_ft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(convformer_m36_sail_in22k_ft_in1k, self).__init__()
        
        self.model = timm.create_model("convformer_m36.sail_in22k_ft_in1k", pretrained=True)
        pre_layer = self.model.head.fc.fc2.in_features
        self.model.head.fc.fc2 = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
    
class convnext_small_in12k_ft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(convnext_small_in12k_ft_in1k, self).__init__()
        
        self.model = timm.create_model("convnext_small.in12k_ft_in1k", pretrained=True)
        pre_layer = self.model.head.fc.fc2.in_features
        self.model.head.fc.fc2 = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class tiny_vit_21m_224_dist_in22k_ft_in1k(nn.Module):
    def __init__(self, num_classes):
        super(tiny_vit_21m_224_dist_in22k_ft_in1k, self).__init__()
        
        self.model = timm.create_model("tiny_vit_21m_224.dist_in22k_ft_in1k", pretrained=True)
        pre_layer = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(pre_layer, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x