import argparse
from models import hsm
import numpy as np
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from models.submodule import *
from utils.model import load_model
#cudnn.benchmark = True
cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='HSM')
    parser.add_argument('modelpath', default=None,
                        help='input model path')
    parser.add_argument('--outdir', default='.',
                        help='output dir')
    args = parser.parse_args()
    
    model, _, pretrained_dict = load_model(args.modelpath, max_disp=128, clean=-1.0, cuda=True, data_parallel_model=True)

    # dry run
    multip = 48
    imgL = np.zeros((1,3,24*multip,32*multip))
    imgR = np.zeros((1,3,24*multip,32*multip))
    imgL = Variable(torch.FloatTensor(imgL).cuda())
    imgR = Variable(torch.FloatTensor(imgR).cuda())
    with torch.no_grad():
        model.eval()
        model(imgL,imgR)

    model_name, ext = os.path.splitext(os.path.basename(args.modelpath))
    output_file_path = os.path.join(args.outdir, '%s-notdp%s'% (model_name, ext))
    torch.save({
        'iters': pretrained_dict['state_dict'],
        'state_dict': model.module.state_dict(),
        'train_loss': pretrained_dict['train_loss'],
        }, output_file_path)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()


