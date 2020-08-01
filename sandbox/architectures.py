# Created by Darvin

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class Inception(nn.Module):
  def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, dropout=0.3):
    super(Inception, self).__init__()
    #1x1 conv branch
    self.b1 = nn.Sequential(nn.Conv2d(in_planes, n1x1, kernel_size=1),
                            nn.BatchNorm2d(n1x1),
                            nn.ReLU(True),
                           )
    #1x1 conv -> 3x3 conv branch
    self.b2 = nn.Sequential(nn.Conv2d(in_planes, n3x3red, kernel_size=1),
                            nn.BatchNorm2d(n3x3red),
                            nn.ReLU(True),
                            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
                            nn.BatchNorm2d(n3x3),
                            nn.ReLU(True),
                           )
    self.b3 = nn.Sequential(nn.Conv2d(in_planes, n5x5red, kernel_size=1),
                            nn.BatchNorm2d(n5x5red),
                            nn.ReLU(True),
                            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
                            nn.BatchNorm2d(n5x5),
                            nn.ReLU(True),
                            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
                            nn.BatchNorm2d(n5x5),
                            nn.ReLU(True),
                           )
    self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
                            nn.BatchNorm2d(pool_planes),
                            nn.ReLU(True),
                           )
    self.do = nn.Dropout2d(p=dropout)
  def forward(self, x):
    y1 = self.b1(x)
    y2 = self.b2(x)
    y3 = self.b3(x)
    y4 = self.b4(x)
    y  = torch.cat([y1, y2, y3, y4], 1)
    y  = self.do(y)
    return y

class GoogLeNet(nn.Module):
  def __init__(self, in_chan=3, out_chan=1, dropout=0.3):
    super(GoogLeNet, self).__init__()
    self.pre_layers = nn.Sequential(nn.Conv2d(in_chan, 64, kernel_size=7, stride=1, padding=3),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(True),
                                    nn.Conv2d(64, 192, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(192),
                                    nn.ReLU(True),
                                    nn.Dropout2d(p=dropout),
                                   )
    self.a3 = Inception(192,  64,  96, 128, 16, 32, 32, dropout=dropout)
    self.b3 = Inception(256, 128, 128, 192, 32, 96, 64, dropout=dropout)

    self.a4 = Inception(480, 192,  96, 208, 16,  48,  64, dropout=dropout)
    self.b4 = Inception(512, 160, 112, 224, 24,  64,  64, dropout=dropout)
    self.c4 = Inception(512, 128, 128, 256, 24,  64,  64, dropout=dropout)
    self.d4 = Inception(512, 112, 144, 288, 32,  64,  64, dropout=dropout)
    self.e4 = Inception(528, 256, 160, 320, 32, 128, 128, dropout=dropout)

    self.a5 = Inception(832, 256, 160, 320, 32, 128, 128, dropout=dropout)
    self.b5 = Inception(832, 384, 192, 384, 48, 128, 128, dropout=dropout)

    self.mp = nn.MaxPool2d(3, stride=2, padding=1)

    self.cT5 = nn.ConvTranspose2d(1024, out_chan, kernel_size=8, stride=4, padding=2)
    self.cT4 = nn.ConvTranspose2d(832,  out_chan, kernel_size=4,  stride=2, padding=1)
    #self.cT3 = nn.ConvTranspose2d(480,  out_chan, kernel_size=4,  stride=2, padding=1)
    self.cT3 = nn.Conv2d(480, out_chan, kernel_size=3, stride=1, padding=1)

    self.mp = nn.MaxPool2d(3, stride=2, padding=1)

  def forward(self, x):
    l2 = self.pre_layers(x)
    l3 = self.b3(self.a3(l2))
    l4 = self.e4(self.d4(self.c4(self.b4(self.a4(self.mp(l3))))))
    l5 = self.b5(self.a5(self.mp(l4)))
    out3 = self.cT3(l3)
    out4 = self.cT4(l4)
    out5 = self.cT5(l5)
    return out3+out4+out5

class Bottleneck(nn.Module):
  expansion = 4
  
  def __init__(self, in_planes, planes, stride=1, dropout=0.3):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
    self.bn1   = nn.BatchNorm2d(planes)
    self.do1   = nn.Dropout2d(p=dropout)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2   = nn.BatchNorm2d(planes)
    self.do2   = nn.Dropout2d(p=dropout)
    self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
    self.bn3   = nn.BatchNorm2d(self.expansion*planes)
    self.do3   = nn.Dropout2d(p=dropout)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion*planes:
      self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                                    nn.BatchNorm2d(self.expansion*planes),
                                    nn.Dropout2d(p=dropout)
                                   )
    
  def forward(self, x):
    out = F.relu(self.do1(self.bn1(self.conv1(x))))
    out = F.relu(self.do2(self.bn2(self.conv2(out))))
    out = self.do3(self.bn3(self.conv3(out)))
    out += self.shortcut(x)
    out = F.relu(out)
    return out

def agg_node(in_planes, out_planes, dropout=0.3):
  return nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),
                       nn.BatchNorm2d(in_planes),
                       nn.Dropout2d(p=dropout),
                       nn.ReLU(),
                       nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
                       nn.BatchNorm2d(out_planes),
                       nn.Dropout2d(p=dropout),
                       nn.ReLU(),
                      )

def upshuffle(in_planes, out_planes, upscale_factor):
  return nn.Sequential(nn.Conv2d(in_planes, out_planes*upscale_factor**2, kernel_size=3, stride=1, padding=1),
                       nn.BatchNorm2d(out_planes*upscale_factor**2),
                       nn.PixelShuffle(upscale_factor),
                       nn.ReLU(),
                      )
'''
  return nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes, upscale_factor*2, stride=upscale_factor, padding=int(upscale_factor/2),bias=False),
                       nn.BatchNorm2d(out_planes),
                       nn.ReLU(),
                      )
'''

def upshuffle_old(in_planes, out_planes, upscale_factor):
  return nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes, upscale_factor*2, stride=upscale_factor, padding=int(upscale_factor/2),bias=False),
                       nn.BatchNorm2d(out_planes),
                       nn.ReLU(),
                      )

def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dropout=0.3):
  return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
                       nn.BatchNorm2d(out_planes),
                       nn.Dropout2d(p=dropout),
                       nn.ReLU(),
                      )

class FPN(nn.Module):
  def __init__(self, num_blocks=[2,4,23-17,3], in_chan=3, out_chan=1, dropout=0.3, block=Bottleneck):
    super(FPN, self).__init__()
    #num_blocks=[2,2,2,2]
    self.in_planes = 64

    self.conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1   = nn.BatchNorm2d(64)

    # Bottom-up layers
    self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1, dropout=dropout)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout=dropout)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout=dropout)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dropout=dropout)

    # Top Layer
    self.toplayer = conv_bn(2048, 256, kernel_size=1, stride=1, padding=0, dropout=dropout) # Reduce channels

    # Smooth layers
    self.smooth1 = conv_bn(256, 256, kernel_size=3, stride=1, padding=1, dropout=dropout)
    self.smooth2 = conv_bn(256, 256, kernel_size=3, stride=1, padding=1, dropout=dropout)
    self.smooth3 = conv_bn(256, 256, kernel_size=3, stride=1, padding=1, dropout=dropout)

    # Lateral layers
    self.latlayer1 = conv_bn(1024, 256, kernel_size=1, stride=1, padding=0, dropout=dropout)
    self.latlayer2 = conv_bn( 512, 256, kernel_size=1, stride=1, padding=0, dropout=dropout)
    self.latlayer3 = conv_bn( 256, 256, kernel_size=1, stride=1, padding=0, dropout=dropout)

    # Aggregate layers
    self.agg1 = agg_node(256, 128, dropout=dropout)
    self.agg2 = agg_node(256, 128, dropout=dropout)
    self.agg3 = agg_node(256, 128, dropout=dropout)
    self.agg4 = agg_node(256, 128, dropout=dropout)

    # Upshuffle layers
    self.up1 = upshuffle(128, 128, 8)
    self.up2 = upshuffle(128, 128, 4)
    self.up3 = upshuffle(128, 128, 2)
    self.up4 = upshuffle(128, 128, 1)

    # Prediction
    self.predict1 = conv_bn(512, 128, kernel_size=3, stride=1, padding=1, dropout=dropout)
    self.predict2 = nn.Conv2d(128, out_chan, kernel_size=3, stride=1, padding=1)
    #self.predict_tot = nn.Conv2d(512, out_chan, kernel_size=3, stride=1, padding=1)
    #self.predictT1    = nn.ConvTranspose2d(256, out_chan, kernel_size=16, stride=8, padding=4)
    #self.predictT2    = nn.ConvTranspose2d(256, out_chan, kernel_size=8, stride=4, padding=2)
    #self.predictT3    = nn.ConvTranspose2d(256, out_chan, kernel_size=4, stride=2, padding=1)
    #self.predictT4    = nn.ConvTranspose2d(256, out_chan, kernel_size=4, stride=2, padding=1)

  def _make_layer(self, block, planes, num_blocks, stride, dropout):
    strides = [stride] + [1]*(num_blocks - 1)
    layers  = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride, dropout))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def _upsample_add(self, x, y):
    _,_,H,W = y.size()
    return F.interpolate(x, size=(H,W), mode='bilinear') + y

  def forward(self, x):
    _,_,H0,W0 = x.size()
    # Bottom-up
    c1 = F.relu(self.bn1(self.conv1(x)))
    c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
    c2 = self.layer1(c1)
    c3 = self.layer2(c2)
    c4 = self.layer3(c3)
    c5 = self.layer4(c4)
    # Top-down
    p5 = self.toplayer(c5)
    p4 = self._upsample_add(p5, self.latlayer1(c4))
    p3 = self._upsample_add(p4, self.latlayer2(c3))
    p2 = self._upsample_add(p3, self.latlayer3(c2))
    # Smooth
    p4 = self.smooth1(p4)
    p3 = self.smooth2(p3)
    p2 = self.smooth3(p2)
    # Aggregate
    a5 = self.agg1(p5)
    a4 = self.agg2(p4)
    a3 = self.agg3(p3)
    a2 = self.agg4(p2)
    # Upsampling
    d5 = self.up1(a5)
    d4 = self.up2(a4)
    d3 = self.up3(a3)
    d2 = self.up4(a2)
    # Resizing and Combining
    _,_,H,W = d2.size()
    d5 = F.interpolate(d5, size=(H,W), mode='bilinear')
    d4 = F.interpolate(d4, size=(H,W), mode='bilinear')
    d3 = F.interpolate(d3, size=(H,W), mode='bilinear')
    vol = torch.cat( [d5,d4,d3,d2], dim=1  )
    # Predicting
    out = self.predict1(vol)
    out = self.predict2(out)
    out = F.interpolate(out, size=(H0,W0), mode='bilinear')
    #out  = F.interpolate(self.predict_tot(vol), size=(H0,W0), mode='bilinear')
    #out5 = F.interpolate(self.predictT1(p5),     size=(H0,W0), mode='bilinear')
    #out4 = F.interpolate(self.predictT2(p4),     size=(H0,W0), mode='bilinear')
    #out3 = F.interpolate(self.predictT3(p3),     size=(H0,W0), mode='bilinear')
    #out2 = F.interpolate(self.predictT4(p2),     size=(H0,W0), mode='bilinear')
    return out#+out5+out4+out3+out2

class FPN_old(nn.Module):
  def __init__(self, num_blocks=[2,4,23,3], in_chan=3, out_chan=1, dropout=0.3, block=Bottleneck):
    super(FPN_old, self).__init__()
    #num_blocks=[2,2,2,2]
    self.in_planes = 64

    self.conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1   = nn.BatchNorm2d(64)

    # Bottom-up layers
    self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1, dropout=dropout)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1, dropout=dropout)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout=dropout)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dropout=dropout)

    # Top Layer
    self.toplayer = conv_bn(2048, 256, kernel_size=1, stride=1, padding=0, dropout=dropout) # Reduce channels

    # Smooth layers
    self.smooth1 = conv_bn(256, 256, kernel_size=3, stride=1, padding=1, dropout=dropout)
    self.smooth2 = conv_bn(256, 256, kernel_size=3, stride=1, padding=1, dropout=dropout)
    self.smooth3 = conv_bn(256, 256, kernel_size=3, stride=1, padding=1, dropout=dropout)

    # Lateral layers
    self.latlayer1 = conv_bn(1024, 256, kernel_size=1, stride=1, padding=0, dropout=dropout)
    self.latlayer2 = conv_bn( 512, 256, kernel_size=1, stride=1, padding=0, dropout=dropout)
    self.latlayer3 = conv_bn( 256, 256, kernel_size=1, stride=1, padding=0, dropout=dropout)

    # Aggregate layers
    self.agg1 = agg_node(256, 128, dropout=dropout)
    self.agg2 = agg_node(256, 128, dropout=dropout)
    self.agg3 = agg_node(256, 128, dropout=dropout)
    self.agg4 = agg_node(256, 128, dropout=dropout)

    # Upshuffle layers
    self.up1 = upshuffle_old(128, 128, 8)
    self.up2 = upshuffle_old(128, 128, 4)
    self.up3 = upshuffle_old(128, 128, 2)
    self.up4 = upshuffle_old(128, 128, 2)

    # Prediction
    self.predict1 = conv_bn(512, 128, kernel_size=3, stride=1, padding=1, dropout=dropout)
    self.predict2 = nn.Conv2d(128, out_chan, kernel_size=3, stride=1, padding=1)
    #self.predict_tot = nn.Conv2d(512, out_chan, kernel_size=3, stride=1, padding=1)
    self.predictT1    = nn.ConvTranspose2d(256, out_chan, kernel_size=16, stride=8, padding=4)
    self.predictT2    = nn.ConvTranspose2d(1024, out_chan, kernel_size=8, stride=4, padding=2)
    self.predictT3    = nn.ConvTranspose2d(512, out_chan, kernel_size=4, stride=2, padding=1)
    #self.predictT4    = nn.ConvTranspose2d(256, out_chan, kernel_size=4, stride=2, padding=1)

  def _make_layer(self, block, planes, num_blocks, stride, dropout):
    strides = [stride] + [1]*(num_blocks - 1)
    layers  = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride, dropout))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def _upsample_add(self, x, y):
    _,_,H,W = y.size()
    return F.interpolate(x, size=(H,W), mode='bilinear') + y

  def forward(self, x):
    _,_,H0,W0 = x.size()
    # Bottom-up
    c1 = F.relu(self.bn1(self.conv1(x)))
    #c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
    c2 = self.layer1(c1)
    c3 = self.layer2(c2)
    c4 = self.layer3(c3)
    c5 = self.layer4(c4)
    # Top-down
    p5 = self.toplayer(c5)
    p4 = self._upsample_add(p5, self.latlayer1(c4))
    p3 = self._upsample_add(p4, self.latlayer2(c3))
    p2 = self._upsample_add(p3, self.latlayer3(c2))
    # Smooth
    p4 = self.smooth1(p4)
    p3 = self.smooth2(p3)
    p2 = self.smooth3(p2)
    # Aggregate
    a5 = self.agg1(p5)
    a4 = self.agg2(p4)
    a3 = self.agg3(p3)
    a2 = self.agg4(p2)
    # Upsampling
    d5 = self.up1(a5)
    d4 = self.up2(a4)
    d3 = self.up3(a3)
    d2 = self.up4(a2)
    # Resizing and Combining
    _,_,H,W = d2.size()
    d5 = F.interpolate(d5, size=(H,W), mode='bilinear')
    d4 = F.interpolate(d4, size=(H,W), mode='bilinear')
    d3 = F.interpolate(d3, size=(H,W), mode='bilinear')
    vol = torch.cat( [d5,d4,d3,d2], dim=1  )
    # Predicting
    out = self.predict1(vol)
    out = self.predict2(out)
    out = F.interpolate(out, size=(H0,W0), mode='bilinear')
    #out  = F.interpolate(self.predict_tot(vol), size=(H0,W0), mode='bilinear')
    out5 = self.predictT1(p5)#F.interpolate(self.predictT1(p5),     size=(H0,W0), mode='bilinear')
    out4 = self.predictT2(c4)#F.interpolate(self.predictT2(p4),     size=(H0,W0), mode='bilinear')
    out3 = self.predictT3(c3)#F.interpolate(self.predictT3(p3),     size=(H0,W0), mode='bilinear')
    #out2 = self.predictT4(p2)#F.interpolate(self.predictT4(p2),     size=(H0,W0), mode='bilinear')
    return out+out5+out4+out3#+out2

class FPN_ppd(nn.Module):
  def __init__(self, num_blocks=[2,4,23-17,3], in_chan=3, out_chan=1, dropout=0.3, block=Bottleneck):
    super(FPN_ppd, self).__init__()
    self.in_planes = 64

    self.conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1   = nn.BatchNorm2d(64)

    # Bottom-up layers
    self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1, dropout=dropout)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1, dropout=dropout)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout=dropout)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dropout=dropout)

    # Top Layer
    self.toplayer = conv_bn(2048, 256, kernel_size=1, stride=1, padding=0, dropout=dropout) # Reduce channels

    # Smooth layers
    self.smooth1 = conv_bn(256, 256, kernel_size=3, stride=1, padding=1, dropout=dropout)
    self.smooth2 = conv_bn(256, 256, kernel_size=3, stride=1, padding=1, dropout=dropout)
    self.smooth3 = conv_bn(256, 256, kernel_size=3, stride=1, padding=1, dropout=dropout)

    # Lateral layers
    self.latlayer1 = conv_bn(1024, 256, kernel_size=1, stride=1, padding=0, dropout=dropout)
    self.latlayer2 = conv_bn( 512, 256, kernel_size=1, stride=1, padding=0, dropout=dropout)
    self.latlayer3 = conv_bn( 256, 256, kernel_size=1, stride=1, padding=0, dropout=dropout)

    # Aggregate layers
    self.agg1 = agg_node(256, 128, dropout=dropout)
    self.agg2 = agg_node(256, 128, dropout=dropout)
    self.agg3 = agg_node(256, 128, dropout=dropout)
    self.agg4 = agg_node(256, 128, dropout=dropout)

    # Upshuffle layers
    self.up1 = upshuffle(128, 128, 8)
    self.up2 = upshuffle(128, 128, 4)
    self.up3 = upshuffle(128, 128, 2)
    self.up4 = upshuffle(128, 128, 2)

    # Prediction
    self.predict1 = conv_bn(512, 128, kernel_size=3, stride=1, padding=1, dropout=dropout)
    self.predict2 = nn.Conv2d(128, out_chan, kernel_size=3, stride=1, padding=1)
    #self.predict_tot = nn.Conv2d(512, out_chan, kernel_size=3, stride=1, padding=1)
    self.predictT1    = nn.ConvTranspose2d(256, out_chan, kernel_size=16, stride=8, padding=4)
    self.predictT2    = nn.ConvTranspose2d(1024, out_chan, kernel_size=8, stride=4, padding=2)
    self.predictT3    = nn.ConvTranspose2d(512, out_chan, kernel_size=4, stride=2, padding=1)
    #self.predictT4    = nn.ConvTranspose2d(256, out_chan, kernel_size=4, stride=2, padding=1)

  def _make_layer(self, block, planes, num_blocks, stride, dropout):
    strides = [stride] + [1]*(num_blocks - 1)
    layers  = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride, dropout))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def _upsample_add(self, x, y):
    _,_,H,W = y.size()
    return F.interpolate(x, size=(H,W), mode='bilinear') + y

  def forward(self, x):
    _,_,H0,W0 = x.size()
    x1,x2 = torch.split(x, int(x.size(1)/2), dim=1)
    # Bottom-up
    c11,c12 = F.relu(self.bn1(self.conv1(x1))),F.relu(self.bn1(self.conv1(x2)))
    #c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
    c21,c22 = self.layer1(c11),self.layer1(c12)
    c31,c32 = self.layer2(c21),self.layer2(c22)
    c41,c42 = self.layer3(c31),self.layer3(c32)
    c51,c52 = self.layer4(c41),self.layer4(c42)
    c5 = c52-c51
    c4 = c42-c41
    c3 = c32-c31
    c2 = c22-c21
    c1 = c12-c11
    # Top-down
    p5 = self.toplayer(c5)
    p4 = self._upsample_add(p5, self.latlayer1(c4))
    p3 = self._upsample_add(p4, self.latlayer2(c3))
    p2 = self._upsample_add(p3, self.latlayer3(c2))
    # Smooth
    p4 = self.smooth1(p4)
    p3 = self.smooth2(p3)
    p2 = self.smooth3(p2)
    # Aggregate
    a5 = self.agg1(p5)
    a4 = self.agg2(p4)
    a3 = self.agg3(p3)
    a2 = self.agg4(p2)
    # Upsampling
    d5 = self.up1(a5)
    d4 = self.up2(a4)
    d3 = self.up3(a3)
    d2 = self.up4(a2)
    # Resizing and Combining
    _,_,H,W = d2.size()
    d5 = F.interpolate(d5, size=(H,W), mode='bilinear')
    d4 = F.interpolate(d4, size=(H,W), mode='bilinear')
    d3 = F.interpolate(d3, size=(H,W), mode='bilinear')
    vol = torch.cat( [d5,d4,d3,d2], dim=1  )
    # Predicting
    out = self.predict1(vol)
    out = self.predict2(out)
    out = F.interpolate(out, size=(H0,W0), mode='bilinear')
    #out  = F.interpolate(self.predict_tot(vol), size=(H0,W0), mode='bilinear')
    out5 = self.predictT1(p5)#F.interpolate(self.predictT1(p5),     size=(H0,W0), mode='bilinear')
    out4 = self.predictT2(c4)#F.interpolate(self.predictT2(p4),     size=(H0,W0), mode='bilinear')potentiallyc
    out3 = self.predictT3(c3)#F.interpolate(self.predictT3(p3),     size=(H0,W0), mode='bilinear')potentiallyc
    #out2 = self.predictT4(p2)#F.interpolate(self.predictT4(p2),     size=(H0,W0), mode='bilinear')
    return out+out5+out4+out3#+out2


























































class Inception_3d(nn.Module):
  def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, dropout=0.3):
    super(Inception_3d, self).__init__()
    # 1x1x1 conv branch
    self.b1 = nn.Sequential(nn.Conv3d(in_planes, n1x1, kernel_size=1),
                            nn.BatchNorm3d(n1x1),
                            nn.Dropout3d(p=dropout),
                            nn.ReLU(True),
                           )
    # 1x1x1 conv -> 3x3x3 conv branch
    self.b2 = nn.Sequential(nn.Conv3d(in_planes, n3x3red, kernel_size=1),
                            nn.BatchNorm3d(n3x3red),
                            nn.Dropout3d(p=dropout),
                            nn.ReLU(True),
                            nn.Conv3d(n3x3red, n3x3, kernel_size=3, padding=1),
                            nn.BatchNorm3d(n3x3),
                            nn.Dropout3d(p=dropout),
                            nn.ReLU(True),
                           )
    # 1x1x1 conv -> 5x5x5 conv branch
    self.b3 = nn.Sequential(nn.Conv3d(in_planes, n5x5red, kernel_size=1),
                            nn.BatchNorm3d(n5x5red),
                            nn.Dropout3d(p=dropout),
                            nn.ReLU(True),
                            nn.Conv3d(n5x5red, n5x5, kernel_size=5, padding=2),
                            nn.BatchNorm3d(n5x5),
                            nn.Dropout3d(p=dropout),
                            nn.ReLU(True),
                           )
    # 3x3 pool -> 1x1 conv branch
    self.b4 = nn.Sequential(nn.MaxPool3d(3, stride=1, padding=1),
                            nn.Conv3d(in_planes, pool_planes, kernel_size=1),
                            nn.BatchNorm3d(pool_planes),
                            nn.Dropout3d(p=dropout),
                            nn.ReLU(True),
                           )
  def forward(self, x):
    y1 = self.b1(x)
    y2 = self.b2(x)
    y3 = self.b3(x)
    y4 = self.b4(x)
    return torch.cat([y1,y2,y3,y4], 1)

class GoogLeNet_3d(nn.Module):
  def __init__(self, in_chan=3, out_chan=1, dropout=0.3):
    super(GoogLeNet_3d, self).__init__()
    self.pre_layers = nn.Sequential(nn.Conv3d(in_chan, 64, kernel_size=3, padding=1, stride=2),
                                    nn.BatchNorm3d(64),
                                    nn.Dropout3d(p=dropout),
                                    nn.ReLU(True),
                                    nn.Conv3d(64, 192, kernel_size=3, padding=1),
                                    nn.BatchNorm3d(192),
                                    nn.Dropout3d(p=dropout),
                                    nn.ReLU(True),
                                   )
    # 3rd Block
    self.a3 = Inception_3d(192, 64, 96, 128, 16, 32, 32, dropout=dropout)
    self.b3 = Inception_3d(256, 128, 128, 192, 32, 96, 64, dropout=dropout)
    # 4th Block
    self.a4 = Inception_3d(480, 192, 96, 208, 16, 48, 64, dropout=dropout)
    self.b4 = Inception_3d(512, 160, 112, 224, 24, 64, 64, dropout=dropout)
    self.c4 = Inception_3d(512, 128, 128, 256, 24, 64, 64, dropout=dropout)
    self.d4 = Inception_3d(512, 112, 144, 288, 32, 64, 64, dropout=dropout)
    self.e4 = Inception_3d(528, 256, 160, 320, 32, 128, 128, dropout=dropout)
    # 5th Block
    self.a5 = Inception_3d(832, 256, 160, 320, 32, 128, 128, dropout=dropout)
    self.b5 = Inception_3d(832, 384, 192, 384, 48, 128, 128, dropout=dropout)
    # Conv Transpose
    self.cT  = nn.ConvTranspose3d(1024, out_chan, kernel_size=16, stride=8, padding=4)
    self.cT1 = nn.ConvTranspose3d(832, out_chan, kernel_size=8, stride=4, padding=2)
    self.cT2 = nn.ConvTranspose3d(480, out_chan, kernel_size=4, stride=2, padding=1)
    # Pooling
    self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
  def forward(self, x):
    x = self.pre_layers(x) #/2
    x = self.b3(self.a3(x))
    out2 = self.cT2(x)
    x = self.maxpool(x) #/2
    x = self.e4(self.d4(self.c4(self.b4(self.a4(x)))))
    out1 = self.cT1(x)
    x = self.maxpool(x) #/2
    x = self.b5(self.a5(x))
    x = self.cT(x)
    return (x,out1,out2)

