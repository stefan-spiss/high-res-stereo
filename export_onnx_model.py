import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils.model import *
from utils.inference import *
from models.submodule import *
from utils.eval import save_pfm
#cudnn.benchmark = True
cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='HSM')
    parser.add_argument('modelpath', default=None,
                        help='input model path')
    parser.add_argument('leftimg', default='./data-mbtest/Insta360Pro/im0.png',
                        help='left input image')
    parser.add_argument('rightimg', default='./data-mbtest/Insta360Pro/im1.png',
                        help='right input image')
    parser.add_argument('--outdir', default='output',
                        help='output dir')
    parser.add_argument('--outfilename', default='output',
                        help='output file name')
    parser.add_argument('--maxdisp', type=float, default=128,
                        help='maximum disparity to search for')
    parser.add_argument('--clean', type=float, default=-1,
                        help='clean up output using entropy estimation')
    parser.add_argument('--resscale', type=float, default=1.0,
                        help='resolution scale')
    parser.add_argument('--level', type=int, default=1,
                        help='output level of output, default is level 1 (stage 3),\
                              can also use level 2 (stage 2) or level 3 (stage 1), only affects the inference, scripting uses level 1.')
    parser.add_argument('--cuda', default=True, action=argparse.BooleanOptionalAction,
                        help='use cuda if available')
    args = parser.parse_args()

    left_input_img = args.leftimg
    right_input_img = args.rightimg

    # load model
    if args.cuda and torch.cuda.is_available():
        run_cuda = True
    else:
        run_cuda = False
    
    model, _, _ = load_model(model_path=args.modelpath, max_disp=args.maxdisp, clean=args.clean, cuda=run_cuda)
    model.eval()

    if run_cuda:
        module = model.module
    else:
        module = model

    # load images
    imgL, imgR, img_size_in, img_size_in_scaled, img_size_net_in = load_image_pair(left_input_img, right_input_img, args.resscale)

    if run_cuda:
        imgL = torch.FloatTensor(imgL).cuda()
        imgR = torch.FloatTensor(imgR).cuda()
    else:
        imgL = torch.FloatTensor(imgL)
        imgR = torch.FloatTensor(imgR)


    onnx_model_name = 'highresnet_model-%s-%s.onnx'% (('cuda' if run_cuda else 'cpu'), args.outfilename)
    export_onnx_model(module, imgL, imgR, '%s/%s'%(args.outdir, onnx_model_name), ['left', 'right'], ['disparity', 'entropy'])

    # run model inference and measure time
    print("pytorch model - run 1:")
    perform_inference(model, imgL, imgR, run_cuda)
    print("pytorch model - run 2:")
    pred_disp, entropy, _ = perform_inference(model, imgL, imgR, run_cuda)

    pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()
    entropy = torch.squeeze(entropy).data.cpu().numpy()
    top_pad   = img_size_net_in[0]-img_size_in_scaled[0]
    left_pad  = img_size_net_in[1]-img_size_in_scaled[1]
    pred_disp = pred_disp[top_pad:,:pred_disp.shape[1]-left_pad]
    entropy = entropy[top_pad:,:pred_disp.shape[1]-left_pad]

    # resize to highres
    pred_disp = cv2.resize(pred_disp/args.resscale,(img_size_in[1],img_size_in[0]),interpolation=cv2.INTER_LINEAR)

    # clip while keep inf
    invalid = np.logical_or(pred_disp == np.inf,pred_disp!=pred_disp)
    pred_disp[invalid] = np.inf

    torch.cuda.empty_cache()

    disp_vis = pred_disp/pred_disp[~invalid].max()*255
    ent_vis = entropy/entropy.max()*255
    # save predictions

    cv2.imshow('disp', disp_vis.astype(np.uint8))
    cv2.imshow('ent', ent_vis.astype(np.uint8))
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
