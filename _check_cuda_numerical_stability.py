'''
用于检测cuda运算错误
'''

import torch
import torch.nn as nn
from torch.backends import cudnn
import argparse
import time
import math


def ConvBnAct(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation),
                         nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9),
                         act)


def DeConvBnAct(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1, dilation=1):
    return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation),
                         nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9),
                         act)


class RevSequential(nn.ModuleList):
    '''
    功能大部分与ModuleList重叠
    '''
    def __init__(self, modules=None):
        super().__init__(modules)

    def append(self, module):
        assert hasattr(module, 'invert') and callable(module.invert)
        super().append(module)

    def extend(self, modules):
        for m in modules:
            self.append(m)

    def forward(self, x1, x2):
        y1, y2 = x1, x2
        for m in self:
            y1, y2 = m(y1, y2)
        return y1, y2

    def invert(self, y1, y2):
        x1, x2 = y1, y2
        for m in list(self)[::-1]:
            x1, x2 = m.invert(x1, x2)
        return x1, x2


class RevGroupBlock(RevSequential):
    '''
    当前只支持输入通道等于输出通道，并且不允许下采样
    '''
    def __init__(self, in_ch, out_ch, stride, act, block_type, blocks, **kwargs):
        assert in_ch == out_ch
        assert stride == 1
        mods = []
        for _ in range(blocks):
            mods.append(block_type(in_ch=in_ch, out_ch=out_ch, stride=1, act=act, **kwargs))
        # self.extend(mods)
        super().__init__(mods)


class RevBlockC(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act, **kwargs):
        super().__init__()
        inter_ch = in_ch // 2
        self.conv1 = ConvBnAct(in_ch, inter_ch, ker_sz=3, stride=1, pad=1, act=act)
        self.conv2 = ConvBnAct(inter_ch, inter_ch, ker_sz=5, stride=1, pad=2, act=act, group=inter_ch)
        self.conv3 = ConvBnAct(in_ch, in_ch, ker_sz=1, stride=1, pad=0, act=nn.Identity())

    def func(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y = torch.cat([y1, y2], dim=1)
        y = self.conv3(y)
        return y

    def forward(self, x1, x2):
        y = x1 + self.func(x2)
        return x2, y

    def invert(self, y1, y2):
        x2, y = y1, y2
        x1 = y - self.func(x2)
        return x1, x2


if __name__ == '__main__':
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.set_grad_enabled(False)

    parse = argparse.ArgumentParser(description='Used to detect CUDA numerical stability problems.')
    parse.add_argument('-i', type=int, help='card id. Which cuda card do you want to test. default: 0', default=0)
    parse.add_argument('-t', type=int, help='minute. Test duration. When the setting is less than or equal to 0, it will not stop automatically. defaule: 30', default=30)
    parse.add_argument('-bs', type=int, help='Test batch size when testing. defaule: 20', default=20)
    parse = parse.parse_args()

    duration = parse.t * 60
    if duration <= 0:
        duration = math.inf

    card_id = parse.i
    if card_id == -1:
        # 使用cpu测试理论上是永远不会报错的
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{card_id}')

    batch_size = parse.bs
    assert batch_size > 0

    start_time = time.time()
    test_count = 0

    act = nn.ELU()
    rvb = RevGroupBlock(128, 128, 1, act, RevBlockC, 32).to(device)
    rvb.eval()

    is_no_error = True

    print('CUDA numerical stability test begin.')
    while is_no_error:
        cur_time = time.time()
        if cur_time - start_time > duration:
            break
        test_count += 1

        if test_count % 50 == 0:
            # 每50次迭代后，刷新一次网络权重
            rvb = RevGroupBlock(128, 128, 1, act, RevBlockC, 32).to(device)
            rvb.eval()

        a1 = torch.randn(batch_size, 128, 128, 128, device=device)
        b1, b2 = rvb(a1, a1)
        o_a1, o_a2 = rvb.invert(b1, b2)
        max_diff_1 = torch.abs(o_a1 - o_a2).max()
        max_diff_2 = torch.abs(a1 - o_a1).max()

        line = f'elapsed/total: {int(cur_time-start_time)}/{duration} card_id: {card_id} count: {test_count} max_diff_1: {max_diff_1:.8f} max_diff_2: {max_diff_2:.8f}'
        print(line)
        if max_diff_1 > 1e-3 or max_diff_2 > 1e-3:
            print(f'A large numerical error was found!')
            is_no_error = False

    if is_no_error:
        print(f'Test passed. Card ID: {card_id}')
    else:
        print(f'Test failed. Card ID: {card_id}')
