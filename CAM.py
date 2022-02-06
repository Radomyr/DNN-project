import io

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json

# input image
#image_file = './original dataset/images/maksssksksss848.png'
file = "269853120.png"
image_file = "./in/" + file

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 6
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'
elif model_id == 4:
    net = models.googlenet(pretrained=True)
    finalconv_name = 'inception5b'
elif model_id == 5: # Radomyr's best-model
    net = torch.load('best-model.pt')
    finalconv_name = 'inception5b'
elif model_id == 6: # best model from training whole network
    checkpoint_path = f"checkpoints/exercise=2C/loss_type=SGD/batch_size=32/20_iterations_best_acc80.pth"
    net = models.googlenet(pretrained=True)
    n_inputs = net.fc.in_features
    last_layer = nn.Linear(n_inputs, 3, bias=True)
    net.fc = last_layer
    device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model'])
    finalconv_name = 'inception5b'
elif model_id == 7: # best model from training whole network
    checkpoint_path = f"checkpoints/exercise=2A/loss_type=SGD/batch_size=32/40_iterations_best_acc64.pth"
    net = models.googlenet(pretrained=True)
    n_inputs = net.fc.in_features
    last_layer = nn.Linear(n_inputs, 3, bias=True)
    net.fc = last_layer
    device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model'])
    finalconv_name = 'inception5b'

net = net.to(torch.device("cpu"))

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())

weight_softmax = np.squeeze(params[-2].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

# load test image
img_pil = Image.open(image_file).convert('RGB')
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
logit = net(img_variable)

# load the imagenet category list


h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()

# output the prediction

# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

# render the CAM and output

img = cv2.imread(image_file)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite("CAM.png", result)