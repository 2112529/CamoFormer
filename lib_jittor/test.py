import os, argparse, imageio

import jittor as jt
from jittor import nn

from model.camoformer import CamoFormer
from utils.dataset import test_dataset

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--testroot', type=str, default="/zhaotingfeng/data/test/", help='testing size')
parser.add_argument('--pth_path', type=str, default='snapshot/Net_epoch_best.pkl')
opt = parser.parse_args()


for _data_name in ['CAMO', 'COD10K', 'CHAMELEON', 'NC4K']:
    data_path = opt.testroot+'{}/'.format(_data_name)
    save_path = 'res/{}/{}/'.format(opt.pth_path.split('/')[(- 2)], _data_name)
    model = CamoFormer()
    model.load(opt.pth_path)
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)\
        .set_attrs(batch_size=1, shuffle=False)

    for image, gt, name, _ in test_loader:
        gt /= (gt.max() + 1e-08)
        (res5, res4, res3, res2,res) = model(image)

        res = res2
        c, h, w = gt.shape
        upsample = nn.upsample(res, size=(h, w), mode='bilinear')
        res = res.sigmoid().data.squeeze()
        res = ((res - res.min()) / ((res.max() - res.min()) + 1e-08))
        print('> {} - {}'.format(_data_name, name))
        imageio.imwrite((save_path + name[0]), res)