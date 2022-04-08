import cv2
import numpy as np
# import skimage.io
import torch
import time
from models.submodule import *
from utils.preprocess import get_transform

def load_image_pair(left_img_path, right_img_path, scale_factor):
    processed = get_transform()

    print("load left img: " + left_img_path)
    imgL_o = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
    imgL_o = (cv2.cvtColor(imgL_o, cv2.COLOR_BGR2RGB)).astype(np.float32)
    # imgL_o = (skimage.io.imread(left_img_path).astype('float32'))[:,:,:3]
    print(imgL_o.shape)

    print("load right img: " + right_img_path)
    imgR_o = cv2.imread(right_img_path, cv2.IMREAD_COLOR)
    imgR_o = cv2.cvtColor(imgR_o, cv2.COLOR_BGR2RGB).astype(np.float32)
    # imgR_o = (skimage.io.imread(right_img_path).astype('float32'))[:,:,:3]
    print(imgR_o.shape)

    input_img_size = imgL_o.shape[:2]
    
    # resize
    imgL_o = cv2.resize(imgL_o,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_CUBIC)
    imgR_o = cv2.resize(imgR_o,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_CUBIC)

    input_img_size_scaled = imgL_o.shape[:2]

    imgL = processed(imgL_o).numpy()
    imgR = processed(imgR_o).numpy()

    imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
    imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

    ##fast pad
    h_img_net_in = int(imgL.shape[2] // 64 * 64)
    w_img_net_in = int(imgL.shape[3] // 64 * 64)
    if h_img_net_in < imgL.shape[2]: h_img_net_in += 64
    if w_img_net_in < imgL.shape[3]: w_img_net_in += 64

    top_pad = h_img_net_in-imgL.shape[2]
    left_pad = w_img_net_in-imgL.shape[3]
    imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

    return imgL, imgR, input_img_size, input_img_size_scaled, (h_img_net_in, w_img_net_in)


def perform_inference(model, imgL, imgR, cuda):
    pred_disp = None
    entropy = None
    with torch.no_grad():
        if cuda:
            torch.cuda.synchronize()
        start_time = time.time()
        pred_disp, entropy = model(imgL, imgR)
        if cuda:
            torch.cuda.synchronize()
        ttime = (time.time() - start_time); print('runtime = %.2f' % (ttime*1000) )
    return pred_disp, entropy, ttime
