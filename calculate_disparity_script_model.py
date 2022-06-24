import argparse
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from utils.inference import *
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
    parser.add_argument('--maxdisp', type=int, default=None,
                        help='maximum disparity to search for, should be None or disparity used during tracing if model was created from tracing')
    parser.add_argument('--clean', type=float, default=None,
                        help='clean up output using entropy estimation, needs to be None if model was created from tracing')
    parser.add_argument('--resscale', type=float, default=1.0,
                        help='resolution scale')
    parser.add_argument('--level', type=int, default=1,
                        help='level of output, default is level 1 (stage 3),\
                              can also use level 2 (stage 2) or level 3 (stage 1), needs to be None if model was created from tracing')
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
        device = torch.device('cuda')
    else:
        run_cuda = False
        device = torch.device('cpu')
    
    model = torch.jit.load(args.modelpath, map_location=device)

    if args.maxdisp:
        model.set_max_disp(args.maxdisp)

    if args.clean:
        model.set_clean(args.clean)

    if run_cuda:
        # model = nn.DataParallel(model, device_ids=[0])
        model.cuda().eval()
    else:
        model.eval()

    if args.clean:
        model.set_clean(args.clean)
    if args.level:
        model.set_level(args.level)

    # load images
    imgL, imgR, img_size_in, img_size_in_scaled, img_size_net_in = load_image_pair(left_input_img, right_input_img, args.resscale)

    if run_cuda:
        imgL = Variable(torch.FloatTensor(imgL).cuda())
        imgR = Variable(torch.FloatTensor(imgR).cuda())
    else:
        imgL = Variable(torch.FloatTensor(imgL))
        imgR = Variable(torch.FloatTensor(imgR))


    # run traced model inference and measure time
    print("traced model - run 1:")
    pred_disp, entropy, _ = perform_inference(model, imgL, imgR, run_cuda)

    print("traced model - run 2:")
    pred_disp, entropy, _ = perform_inference(model, imgL, imgR, run_cuda)

    print("traced model - run 3:")
    pred_disp, entropy, _ = perform_inference(model, imgL, imgR, run_cuda)

    print("traced model - run 4:")
    pred_disp, entropy, _ = perform_inference(model, imgL, imgR, run_cuda)

    # load images
    imgL, imgR, img_size_in, img_size_in_scaled, img_size_net_in = load_image_pair(left_input_img, right_input_img, args.resscale)

    if run_cuda:
        imgL = Variable(torch.FloatTensor(imgL).cuda())
        imgR = Variable(torch.FloatTensor(imgR).cuda())
    else:
        imgL = Variable(torch.FloatTensor(imgL))
        imgR = Variable(torch.FloatTensor(imgR))

    print("traced model - run 5:")
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
    if args.saveoutputimgs:
        idxname = '%s-max_disp_%s-clean_%s-res_scale_%s-level_%s'% (args.outfilename, model.maxdisp, args.clean, args.resscale, args.level)
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
