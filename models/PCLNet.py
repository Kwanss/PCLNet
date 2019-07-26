import numpy as np
import torch
import torch.nn as nn

import models.resnetM as resnetM
from .CLSTM import ConvLSTM


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    if type(in_planes) == np.int64:
        in_planes = np.asscalar(in_planes)
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    if type(in_planes) == np.int64:
        in_planes = np.asscalar(in_planes)
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    if type(in_planes) == np.int64:
        in_planes = np.asscalar(in_planes)
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class PCLNet(nn.Module):
    """
    PCLNet: Unsupervised Learning for Optical Flow Estimation Using Pyramid Convolution LSTM
    Author: Shuosen Guan
    """

    def __init__(self, args):
        
        super(PCLNet, self).__init__()
        self.args = args

        snippet_len = args.snippet_len
        self.feature_net = getattr(resnetM, args.backbone)(pretrained=True, num_classes=args.class_num)
        if args.freeze_vgg:
            for p in self.feature_net.parameters():
                p.required_grad = False
            print("[>>>> Feature head frozen.<<<<]")

        # Motion Encoding
        # in_size: 1/2
        self.clstm_encoder_1 = ConvLSTM(input_channels=64, hidden_channels=[64],
                                        kernel_size=3, step=snippet_len, effective_step=list(range(snippet_len))).cuda()
        # in_size: 1/4
        self.clstm_encoder_2 = ConvLSTM(input_channels=64, hidden_channels=[64],
                                        kernel_size=3, step=snippet_len, effective_step=list(range(snippet_len))).cuda()
        # in_size: 1/8
        self.clstm_encoder_3 = ConvLSTM(input_channels=128, hidden_channels=[128],
                                        kernel_size=3, step=snippet_len, effective_step=list(range(snippet_len))).cuda()
        # in_size: 1/16
        self.clstm_encoder_4 = ConvLSTM(input_channels=256, hidden_channels=[256],
                                        kernel_size=3, step=snippet_len, effective_step=list(range(snippet_len))).cuda()

        self.conv_B1    = conv(64, 64, stride=1, kernel_size=3, padding=1)  
        self.conv_S1_1  = conv(64, 64, stride=1, kernel_size=3, padding=1)
        self.conv_S1_2  = conv(64, 64, stride=1, kernel_size=3, padding=1)
        self.conv_D1    = conv(64, 64, stride=2)
        self.Pool1      = nn.MaxPool2d(8, 8)

        self.conv_B2    = conv(64, 64, stride=1, kernel_size=3, padding=1)  
        self.conv_S2_1  = conv(64 + 64, 128, stride=1, kernel_size=3, padding=1)
        self.conv_S2_2  = conv(128, 128, stride=1, kernel_size=3, padding=1)
        self.conv_D2    = conv(128, 64, stride=2)
        self.Pool2      = nn.MaxPool2d(4, 4)

        self.conv_B3    = conv(128, 128, stride=1, kernel_size=3, padding=1) 
        self.conv_S3_1  = conv(128 + 64, 128, stride=1, kernel_size=3, padding=1)
        self.conv_S3_2  = conv(128, 128, stride=1, kernel_size=3, padding=1)
        self.conv_D3    = conv(128, 64, stride=2)
        self.Pool3      = nn.MaxPool2d(2, 2)

        self.conv_B4    = conv(256, 128, stride=1, kernel_size=3, padding=1) 
        self.conv_S4_1  = conv(128 + 64, 128, stride=1, kernel_size=3, padding=1)
        self.conv_S4_2  = conv(128, 128, stride=1, kernel_size=3, padding=1)

        # Motion feature
        self.conv_M = conv((64 + 128 + 128 + 128), 256, stride=1, kernel_size=3, padding=1)

        # Motion reconstruction
        if self.args.couple:
            rec_in_size = [0, 64 + 64 + 2, 128 + 128 + 2, 128 + 196 + 2, 128 + 256]
        else:
            rec_in_size = [0, 64 + 2, 128 + 2, 196 + 2, 256]

        self.conv_4     = conv(rec_in_size[4], 256)
        self.pred_flow4 = predict_flow(256)
        self.up_flow4   = deconv(2, 2)
        self.up_feat4   = deconv(256, 196)

        self.conv_3     = conv(rec_in_size[3], 196)
        self.pred_flow3 = predict_flow(196)
        self.up_flow3   = deconv(2, 2)
        self.up_feat3   = deconv(196, 128)

        self.conv_2     = conv(rec_in_size[2], 96)
        self.pred_flow2 = predict_flow(96)
        self.up_flow2   = conv(2, 2)
        self.up_feat2   = conv(96, 64)

        self.conv_1     = conv(rec_in_size[1], 64)
        self.pred_flow1 = predict_flow(64)

        self.dc_conv1 = conv(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv2 = conv(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = conv(64, 64, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = conv(64, 64, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = conv(64, 64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv7 = predict_flow(32)

    
    def forward(self, x):

        if x.dim() == 6:    # (batch_size, K, snippet_len, channel, H, W)
            batch_size, K, snippet_len, channel, H, W = x.size()
        elif x.dim() == 5:  # (batch_size, snippet_len, channel, H, W)
            batch_size, snippet_len, channel, H, W = x.size()
            K = 1
        elif x.dim() == 4:  # (batch_size, channel * snippet_len, H, W)
            batch_size, _channels, H, W = x.size()
            K, channel = 1, 3
            snippet_len = _channels // channel
        else:
            raise RuntimeError('Input format not suppored!')

        x = x.contiguous().view(-1, channel, H, W)

        la1, la2, la3, la4, _ = self.feature_net(x)  

        la1 = la1.view((-1, snippet_len) + la1.size()[1:])
        la2 = la2.view((-1, snippet_len) + la2.size()[1:])
        la3 = la3.view((-1, snippet_len) + la3.size()[1:])
        la4 = la4.view((-1, snippet_len) + la4.size()[1:])
        # la5 = la5.view((-1, snippet_len) + la5.size()[1:])

        h1, _ = self.clstm_encoder_1(la1)  
        h2, _ = self.clstm_encoder_2(la2)  
        h3, _ = self.clstm_encoder_3(la3) 
        h4, _ = self.clstm_encoder_4(la4)  
        # list for each step (batch_size * K, channel, H, W)

        # (batch_size * K*(snippet_len -1), channel, H, W)
        h1 = torch.stack(h1[1:], 1).view((-1,) + h1[0].size()[-3:])
        h2 = torch.stack(h2[1:], 1).view((-1,) + h2[0].size()[-3:])
        h3 = torch.stack(h3[1:], 1).view((-1,) + h3[0].size()[-3:])
        h4 = torch.stack(h4[1:], 1).view((-1,) + h4[0].size()[-3:])

        x1 = self.conv_B1(h1)
        x1 = self.conv_S1_2(self.conv_S1_1(x1))

        x2 = torch.cat((self.conv_B2(h2), self.conv_D1(x1)), 1)
        x2 = self.conv_S2_2(self.conv_S2_1(x2))

        x3 = torch.cat((self.conv_B3(h3), self.conv_D2(x2)), 1)
        x3 = self.conv_S3_2(self.conv_S3_1(x3))

        x4 = torch.cat((self.conv_B4(h4), self.conv_D3(x3)), 1)
        x4 = self.conv_S4_2(self.conv_S4_1(x4))

        xm = self.conv_M(torch.cat((self.Pool1(x1), self.Pool2(x2), self.Pool3(x3), x4), 1))

        rec_x4 = torch.cat((x4, xm), 1) if self.args.couple else xm
        x = self.conv_4(rec_x4)
        flow4 = self.pred_flow4(x)  
        up_flow4 = self.up_flow4(flow4)
        up_feat4 = self.up_feat4(x)

        rec_x3 = torch.cat((x3, up_feat4, up_flow4), 1) if self.args.couple else torch.cat((up_feat4, up_flow4), 1)
        x = self.conv_3(rec_x3)
        flow3 = self.pred_flow3(x) 
        up_flow3 = self.up_flow3(flow3)
        up_feat3 = self.up_feat3(x)

        rec_x2 = torch.cat((x2, up_feat3, up_flow3), 1) if self.args.couple else torch.cat((up_feat3, up_flow3), 1)
        x = self.conv_2(rec_x2)
        flow2 = self.pred_flow2(x) 
        up_flow2 = self.up_flow2(flow2)
        up_feat2 = self.up_feat2(x)

        rec_x1 = torch.cat((x1, up_feat2, up_flow2), 1) if self.args.couple else torch.cat((up_feat2, up_flow2), 1)
        x = self.conv_1(rec_x1)
        flow1 = self.pred_flow1(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow1 += self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        # output size: (batch_size, K, snippet_len -1 , C,H,W)
        flow_pyramid = [flo.view((batch_size, K, snippet_len - 1,) + flo.size()[-3:])
                        for flo in [flow1, flow2, flow3, flow4]]
        re_dict = {}
        re_dict['flow_pyramid'] = flow_pyramid

        return re_dict



class arg_example():
    def __init__(self):
        self.snippet_len = 2
        self.backbone = 'resnet18'
        self.class_num = 101
        self.freeze_vgg = True
        self.couple = False



if __name__ == "__main__":
    args = arg_example()
    model = PCLNet(args)
    n = sum(p.numel() for p in  model.parameters())
    print(n/ 1e6) 
