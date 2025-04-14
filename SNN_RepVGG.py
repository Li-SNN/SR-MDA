
"""
@CreatedDate:   2024/04
@Author: lyh
"""
import math

import torch
import torch.nn as nn
import numpy as np
# from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

# from qiskit import QuantumCircuit, transpile, assemble

# from qiskit_aer import Aer
act_sum1 = 0
act_sum2 = 0
act_sum3 = 0
act_sum4 = 0
act_sum5 = 0
act_sum6 = 0
act_sum7 = 0
act_sum8 = 0
act_sum9 = 0
act_sum10 = 0
act_sum11 = 0
act_sum12 = 0
act_sum13 = 0
act_sum14 = 0
act_sum15 = 0
act_sum16 = 0
act_sum17 = 0
act_sum18 = 0
act_sum19 = 0
act_sum20 = 0
act_sum21 = 0


class Surrogate_BP_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        temp = torch.abs(1 - torch.abs(torch.arcsin(input))) < 0.7
        # temp = (1 / 2.5) * torch.sign(abs(input) < 2.5)
        return grad_input * temp.float()

def channel_shuffle(x, groups: int):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x

class TGRS(nn.Module):
    def __init__(self, num_steps, leak_mem, img_size, num_cls, input_dim):
        super(TGRS, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        # (">>>>>>>>>>>>>>>>>>> SNN Direct Coding For TGRS >>>>>>>>>>>>>>>>>>>>>>")
        bias_flag = False
        # 192+8192+16384
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)

        self.conv1_w = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=bias_flag)
        self.BN1 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=bias_flag)

        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=bias_flag)
        self.BN2 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=bias_flag)

        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=bias_flag)
        self.BN3=nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=bias_flag)

        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear(4096, self.num_cls, bias=False)  
        self.softmax = nn.LogSoftmax(dim=-1)

        self.conv_list = [self.conv1, self.conv1_w,self.conv2, self.conv3,self.conv4, self.conv5,
                          self.conv6,
                          self.conv7, self.conv8,self.conv9, self.conv10]

        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=5)
            elif isinstance(m, nn.Linear):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=5)

    def forward(self, input):
        batch_size = input.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv1_w = torch.zeros(batch_size, 128, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 128, self.img_size, self.img_size).cuda()
        mem_conv3 = torch.zeros(batch_size, 128, self.img_size, self.img_size).cuda()
        mem_conv4 = torch.zeros(batch_size, 256, self.img_size, self.img_size).cuda()
        mem_conv5 = torch.zeros(batch_size, 256, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv6 = torch.zeros(batch_size, 256, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv7 = torch.zeros(batch_size, 256, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv8 = torch.zeros(batch_size, 256, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv9 = torch.zeros(batch_size, 256, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv10 = torch.zeros(batch_size, 256, self.img_size // 2, self.img_size // 2).cuda()
        mem_fc1 = torch.zeros(batch_size, self.num_cls).cuda()

        mem_conv_list = [mem_conv1,mem_conv1_w,mem_conv2, mem_conv3,mem_conv4, mem_conv5, mem_conv6,mem_conv7,mem_conv8,
                         mem_conv9,mem_conv10]
        global act_sum1, act_sum2, act_sum3, act_sum4, act_sum5, act_sum6, act_sum7, act_sum8, act_sum9, act_sum10, act_sum11, act_sum12, act_sum13, act_sum14, act_sum15, act_sum16, act_sum17, act_sum18, act_sum19, act_sum20, act_sum21
        static_input1 = self.conv1(input)
        def forward(self, index, input_value):
            mem_conv_list[index] = self.leak_mem * mem_conv_list[index] + (1 - self.leak_mem) * self.conv_list[index](input_value) # 总分支
            mem_thr = mem_conv_list[index] - 1
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv_list[index]).cuda()
            rst[mem_thr > 0] = 1
            mem_conv_list[index] = mem_conv_list[index] - rst
            out_prev = out.clone()
            return out_prev

        for t in range(self.num_steps):
            mem_conv_list[0] = self.leak_mem * mem_conv_list[0] + (1 - self.leak_mem) * static_input1  # 总分支
            mem_thr = mem_conv_list[0] - self.conv_list[0].threshold
            out = self.spike_fn(mem_thr)
            # Soft reset
            rst = torch.zeros_like(mem_conv_list[0]).cuda()
            rst[mem_thr > 0] = self.conv_list[0].threshold
            mem_conv_list[0] = mem_conv_list[0] - rst
            out1 = out.clone()

            with torch.no_grad():
                act_sum1 = act_sum1 + torch.sum(torch.abs(out1)).cuda()

            value1_w = forward(self,1,out1)
            with torch.no_grad():
                act_sum2 = act_sum2 + torch.sum(torch.abs(value1_w)).cuda()

            value2 = forward(self, 2, value1_w)
            with torch.no_grad():
                act_sum3 = act_sum3 + torch.sum(torch.abs(value2)).cuda()
            # bn1 = self.BN1(out1)

            value3 = forward(self, 3, value1_w)
            with torch.no_grad():
                act_sum4 = act_sum4 + torch.sum(torch.abs(value3)).cuda()

            res1 = value1_w+value3+value2
            with torch.no_grad():
                act_sum5 = act_sum5 + torch.sum(torch.abs(res1)).cuda()

            value4 = forward(self, 4, res1)
            with torch.no_grad():
                act_sum6 = act_sum6 + torch.sum(torch.abs(value4)).cuda()

            pool1 = self.pool1(value4)
            with torch.no_grad():
                act_sum7 = act_sum7 + torch.sum(torch.abs(pool1)).cuda()

            value5 = forward(self, 5, pool1)
            with torch.no_grad():
                act_sum8 = act_sum8 + torch.sum(torch.abs(value5)).cuda()
            # bn2=self.BN2(pool1)

            value6 = forward(self, 6, pool1)
            with torch.no_grad():
                act_sum9 = act_sum9 + torch.sum(torch.abs(value6)).cuda()

            res3 = value6+value5+pool1
            with torch.no_grad():
                act_sum10 = act_sum10 + torch.sum(torch.abs(res3)).cuda()
            value7 = forward(self, 7, res3)

            value8 = forward(self, 8, value7)
            with torch.no_grad():
                act_sum11 = act_sum11 + torch.sum(torch.abs(value8)).cuda()
            # bn3=self.BN3(value7)

            value9 = forward(self, 9, value7)
            with torch.no_grad():
                act_sum12 = act_sum12 + torch.sum(torch.abs(value9)).cuda()

            res4 = value7+value8+value9
            with torch.no_grad():
                act_sum13 = act_sum13 + torch.sum(torch.abs(res4)).cuda()

            value10 = forward(self,10,res4)
            with torch.no_grad():
                act_sum14 = act_sum14 + torch.sum(torch.abs(value10)).cuda()

            pool2 = self.pool2(value10)
            with torch.no_grad():
                act_sum15 = act_sum15 + torch.sum(torch.abs(pool2)).cuda()

            out_prev = pool2.reshape(batch_size, -1)
            mem_fc1 = mem_fc1 + self.fc1(out_prev)

        out_voltage = mem_fc1 / self.num_steps
        print('l1:%20d' % act_sum1)
        print('l2:%20d' % act_sum2)
        print('l3:%20d' % act_sum3)
        print('l4:%20d' % act_sum4)
        print('l5:%20d' % act_sum5)
        print('l6:%20d' % act_sum6)
        print('l7:%20d' % act_sum7)
        print('l8:%20d' % act_sum8)
        print('l9:%20d' % act_sum9)
        print('l10:%20d' % act_sum10)
        print('l11:%20d' % act_sum11)
        print('l12:%20d' % act_sum12)
        print('l13:%20d' % act_sum13)
        print('l14:%20d' % act_sum14)
        print('l15:%20d' % act_sum15)


        return out_voltage
