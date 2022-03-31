from models import hsm
import torch
import torch.nn as nn
from models.submodule import *

def load_model(model_path, max_disp, clean, level, cuda=True, data_parallel_model=False):
    # construct model
    model = hsm(max_disp,clean,level=level)
    device = None
    if cuda:
        device = torch.device('cuda')
        if data_parallel_model:
            model = nn.DataParallel(model, device_ids=[0])
            model.cuda()
        else:
            model.cuda()
    else:
        device = torch.device('cpu')

    pretrained_dict = None
    if model_path is not None:
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
        model.load_state_dict(pretrained_dict['state_dict'],strict=False)
    else:
        print('run with random init')
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if not data_parallel_model:
        if cuda:
            model = nn.DataParallel(model, device_ids=[0])
            model.cuda()
        else:
            model.cpu()

    if max_disp>0:
        max_disp = int(max_disp)
    else:
        max_disp = int(128)

    if cuda:
        module = model.module
    else:
        module = model

    ## change max disp
    tmpdisp = int(max_disp//64*64)
    if (max_disp/64*64) > tmpdisp:
        module.maxdisp = tmpdisp + 64
    else:
        module.maxdisp = tmpdisp
    if module.maxdisp == 64: module.maxdisp=128
    
    if cuda:
        module.disp_reg8 =  disparityregression(module.maxdisp,16).cuda()
        module.disp_reg16 = disparityregression(module.maxdisp,16).cuda()
        module.disp_reg32 = disparityregression(module.maxdisp,32).cuda()
        module.disp_reg64 = disparityregression(module.maxdisp,64).cuda()
    else:
        module.disp_reg8 =  disparityregression(module.maxdisp,16)
        module.disp_reg16 = disparityregression(module.maxdisp,16)
        module.disp_reg32 = disparityregression(module.maxdisp,32)
        module.disp_reg64 = disparityregression(module.maxdisp,64)

    return model, device, pretrained_dict


def trace_model(module, imgL, imgR):
    traced_script_module = None
    with torch.no_grad():
        exampleInput = [imgL, imgR]
        traced_script_module = torch.jit.trace(module, exampleInput)
    return traced_script_module

