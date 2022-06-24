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
    parser.add_argument('--saveoutputimgs', default=False, action=argparse.BooleanOptionalAction,
                        help='if provided, output images are saved')
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


    script_module = create_script_model(module, imgL, imgR)
    script_module_name = 'highresnet_script-%s-%s.pt'% (('cuda' if run_cuda else 'cpu'), args.outfilename)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    script_module.save('%s/%s'%(args.outdir, script_module_name))

    # run model inference and traced model inference and measure time
    print("pytorch model - run 1:")
    perform_inference(model, imgL, imgR, run_cuda)
    print("pytorch model - run 2:")
    pred_disp_m, entropy_m, _ = perform_inference(model, imgL, imgR, run_cuda)

    script_model = script_module;
    if run_cuda:
        script_model = nn.DataParallel(script_model, device_ids=[0])
        script_model.cuda().eval()
        script_model.module.set_level(args.level)
    else:
        script_model.eval()
        script_model.set_level(args.level)

    print("scripted model - run 1:")
    perform_inference(script_model, imgL, imgR, run_cuda)
    print("scripted model - run 2:")
    perform_inference(script_model, imgL, imgR, run_cuda)
    print("scripted model - run 3:")
    pred_disp, entropy, _ = perform_inference(script_model, imgL, imgR, run_cuda)

    print("Resulting disparities are the same: " + str(torch.allclose(pred_disp_m, pred_disp, rtol=1e-4, atol=1e-4, equal_nan=True)))
    print("Resulting entropies are the same: " + str(torch.allclose(entropy_m, entropy, rtol=1e-4, atol=1e-4, equal_nan=True)))

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
    if args.saveoutputimgs:
        idxname = '%s-max_disp_%s-clean_%s-res_scale_%s-level_%s'% (args.outfilename, module.maxdisp, args.clean, args.resscale, args.level)
        np.save('%s/%s-disp.npy'% (args.outdir, idxname),(pred_disp))
        np.save('%s/%s-ent.npy'% (args.outdir, idxname),(entropy))
        cv2.imwrite('%s/%s-disp.png'% (args.outdir, idxname),pred_disp)
        cv2.imwrite('%s/%s-dispVisualize.png'% (args.outdir, idxname),disp_vis)
        cv2.imwrite('%s/%s-ent.png'% (args.outdir, idxname),ent_vis)

        with open('%s/%s/disp0HSM.pfm'% (args.outdir, idxname),'w') as f:
            save_pfm(f,pred_disp[::-1,:])

    cv2.imshow('disp', disp_vis.astype(np.uint8))
    cv2.imshow('ent', ent_vis.astype(np.uint8))
    cv2.waitKey(0)

if __name__ == '__main__':
    main()


