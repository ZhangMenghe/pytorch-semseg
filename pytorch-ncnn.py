import torch.onnx
import os
os.sys.path.append('../../')
from ptsemseg.models.pspnet import pspnet
from ptsemseg.models import get_model
# Setup device
import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
n_classes = 14
model = get_model({"arch": "pspnet"}, n_classes).to(device)

from ptsemseg.utils import convert_state_dict
model_path="../pspnet_nyuv2_best_model.pkl"
state = convert_state_dict(torch.load(model_path)["model_state"])
model.load_state_dict(state)

from torch.autograd import Variable
import torch.onnx
import torchvision
dummy_input = Variable(torch.randn(1, 3, 480, 640))

torch.onnx.export(model, dummy_input, "pspnet.onnx")
