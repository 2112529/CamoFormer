import os, argparse, imageio

import jittor as jt
from jittor import nn
import numpy as np

from model.camoformer import CamoFormer
from utils.dataset import test_dataset

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--testroot', type=str, default="/zhaotingfeng/datasets/test/", help='testing size')
parser.add_argument('--pth_path', type=str, default='snapshot/camoformer_pvtv2b4.pkl')
opt = parser.parse_args()

model = CamoFormer(opt)
model.eval()

for _data_name in ['CAMO', 'COD10K', 'CHAMELEON', 'NC4K']:
    data_path = opt.testroot+'{}/'.format(_data_name)
    save_path = 'result/{}/{}/'.format(opt.pth_path.split('/')[(- 2)], _data_name)

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}images/'.format(data_path)
    gt_root = '{}GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)\
        .set_attrs(batch_size=1, shuffle=False)

    for image, gt, name, _ in test_loader:
        gt /= (gt.max() + 1e-08)
        res5, res4, res3, res2,res = model(image)
        # pred = torch.sigmoid(res[0,0]).cpu().numpy()*255
        c, h, w = gt.shape
        res = nn.upsample(res, size=(h, w), mode='bilinear')
        res = res.sigmoid().data.squeeze()
        res = ((res - res.min()) / ((res.max() - res.min()) + 1e-08))
        res = (res * 255).clip(0, 255).astype(np.uint8)
        print('> {} - {}'.format(_data_name, name))
        imageio.imwrite((save_path + name[0]), res)