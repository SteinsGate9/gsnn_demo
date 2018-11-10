import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.gsnn import finetune

# set vgg 16
# set vgg 16
vgg16 = models.vgg16(pretrained=True)
pretrained_dict = vgg16.state_dict()
model = finetune()
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
print(list(map(id, vgg16.parameters())))