from models import hsm
import torch
import torch.nn as nn
from models.submodule import *

def load_model(model_path, max_disp, clean, cuda=True, data_parallel_model=False):
    # construct model
    model = hsm(max_disp, clean=clean)
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

    return model, device, pretrained_dict


def trace_model(module, imgL, imgR):
    print("Creating script module of traced model")
    traced_script_module = None
    with torch.no_grad():
        exampleInput = (imgL, imgR)
        # there is an error with the graphs being different if check_trace is True, not sure what's the problem -> check
        # disabled
        traced_script_module = torch.jit.trace(module, example_inputs=exampleInput, optimize=True, check_trace=False)
    return traced_script_module

def create_script_model(module, imgL, imgR):
    print("Creating script module")
    script_module = None
    # with torch.no_grad():
    # with torch.jit.optimized_execution(True):
    example_input = [(imgL, imgR)]
    script_module = torch.jit.script(module, optimize=True, example_inputs=example_input)
    return script_module

