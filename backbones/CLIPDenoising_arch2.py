import torch
from torch import nn

from backbones.CLIPEncoder_util2 import (ModifiedResNet,
                                                      UNetUpBlock,
                                                      UNetUpBlock_nocat,
                                                      FSASUpBlock,
                                                      FSASUpBlock_nocat,
                                                      conv3x3)

class ModifiedResNet_RemoveFinalLayer(ModifiedResNet):
   
    def __init__(self, layers, in_chn=3, width=64):
        super().__init__(layers, in_chn, width)

    def forward(self, x):
        out = []

        x = x.type(self.conv1.weight.dtype); out.append(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x))); out.append(x)
        x = self.avgpool(x)
        
        x = self.layer1(x); out.append(x)
        x = self.layer2(x); out.append(x)
        x = self.layer3(x); out.append(x)
        x = self.layer4(x); 

        return out

class CLIPDenoising(nn.Module):
    def __init__(self, num_blocks, inp_channels=3, out_channels=3, depth=5, wf=64, slope=0.2,
                       bias=True, model_path=None, aug_level=0.025, pretrain=False, reg_max: int = 18, use_fsas=True):

        super(CLIPDenoising, self).__init__()
        
        self.sigmas = [aug_level * i for i in range(1, 5)]
        self.pretrain = pretrain
        self.reg_max = reg_max
        self.inp_channels = inp_channels
        if inp_channels == 1: # used for 1 channel input, eg. CT images
            self.first = nn.Conv2d(inp_channels, 3, kernel_size=1, bias=bias)
            inp_channels = 3

        self.encoder = ModifiedResNet_RemoveFinalLayer(num_blocks, inp_channels, width=wf)
        self.encoder.load_pretrain_model(model_path)
            
        for params in self.encoder.parameters():
            params.requires_grad = False

        # learnable decoder
        self.up_path = nn.ModuleList()
        prev_channels = wf * 2 ** (len(num_blocks))
        
        UpBlockWithCat = FSASUpBlock if use_fsas else UNetUpBlock
        UpBlockNoCat = FSASUpBlock_nocat if use_fsas else UNetUpBlock_nocat
        
        for i in range(depth):
            if i == 0:
                self.up_path.append(UpBlockNoCat(prev_channels, prev_channels//2, slope, bias))
                prev_channels = prev_channels//2
            elif i == depth - 2: #3
                self.up_path.append(UpBlockWithCat(prev_channels*3//2, prev_channels//2, slope, bias))
                prev_channels = prev_channels//2
            elif i == depth - 1: # introduce noisy image as a dense feature 4
                self.up_path.append(UpBlockWithCat(prev_channels+inp_channels, prev_channels, slope, bias))
            else:
                self.up_path.append(UpBlockWithCat(prev_channels*2, prev_channels//2, slope, bias))
                prev_channels = prev_channels//2

        self.last = conv3x3(prev_channels, out_channels, bias=bias)
        self.__determineOutputLayer(64, -0.2, 0.2, 6, 0., 1.)




    def __determineOutputLayer(self, mid_ch, y_0, y_n, pretrain_out_ch, norm_range_max, norm_range_min):
        if self.pretrain:
            self.pretrain_out_ch = pretrain_out_ch
            self.output = nn.Conv2d(mid_ch, pretrain_out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            # Distribution Regression Layer
            self.output = nn.Conv2d(mid_ch, self.reg_max + 1, kernel_size=3, stride=1, padding=1, bias=True)
            # Bias is initialized to 1.0, which helps the model to accelerate convergence
            self.output.bias.data[:] = 1.0
            proj = torch.linspace(y_0, y_n, self.reg_max + 1, dtype=torch.float) / (norm_range_max - norm_range_min)
            self.register_buffer('proj', proj, persistent=False)
            self.hu_interval = (y_n - y_0) / self.reg_max

    def forward(self, x, Step=1):
        if self.inp_channels == 1:
            re1 = x
        else:
            re1 = x[:,0].unsqueeze(1)

        if self.inp_channels == 1:
            x = self.first(x) #b,1,h,w to b,3,h,w
            
        out = self.encoder(x) 
        #16,3,128,128 16,64,64,64 16,256,32,32 16,512,16,16 16,1024,8,8
        # progressive feature augmentation 
        if self.training:
            for idx in range(len(out)):
                if idx == 0: continue # 
                alpha = torch.randn_like(out[idx]) * self.sigmas[idx-1] + 1.0
                out[idx] = out[idx] * alpha
                
        x = out[-1]
         
        for i, up in enumerate(self.up_path):
            if i != 0: 
                x = up(x, out[-i-1])
            else:
                x = up(x)

        if Step == 1:
            return self.last(x), None, None
        
        output = self.last(x)
        if self.pretrain:
            # In pre-training, the model does not use dfl and directly predicts before
            # and after images instead of predicting bias.
            x = self.output(x)
            return x
        else:
            # Distribution Regression Layer
            x = self.output(x)
            out_dist = x.permute(0, 2, 3, 1)

            softmax_out = out_dist.softmax(3)
            proj_view = self.proj.view([-1, 1])

            x = softmax_out.matmul(proj_view)
                    
            x = x.permute(0, 3, 1, 2)

            x = x + output

            x = torch.clamp(x, 0.0, 1.0)

            return x, output, out_dist
        
