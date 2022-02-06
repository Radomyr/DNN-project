import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import tqdm
import scipy.ndimage as nd
from utilz import deprocess, preprocess, clip



def dream(image, model, iterations, lr):
    """ Updates the image to maximize outputs for n iterations """
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor
    image = Variable(Tensor(image), requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model(image)
        loss = out.norm()
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        image.grad.data.zero_()
    return image.cpu().data.numpy()


def deep_dream(image, model, iterations, lr, octave_scale, num_octaves):
    """ Main deep dream method """
    image = preprocess(image).unsqueeze(0).cpu().data.numpy()

    # Extract image representations for each octave
    octaves = [image]
    for _ in range(num_octaves - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        # Add deep dream detail from previous octave to new base
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image, model, iterations, lr)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image)


if __name__ == "__main__":
    # networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
    model_id = 1
    if model_id == 1:
        net = models.googlenet(pretrained=True)
        finalconv_name = 'inception5b'
    elif model_id == 2:  # Radomyr's best-model
        net = torch.load('best-model.pt')
    elif model_id == 3:  # worst model from 2a
        checkpoint_path = f"checkpoints/exercise=2A/loss_type=SGD/batch_size=32/40_iterations_best_acc64.pth"
        net = models.googlenet(pretrained=True)
        n_inputs = net.fc.in_features
        last_layer = nn.Linear(n_inputs, 3, bias=True)
        net.fc = last_layer
        device = torch.device('cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint['model'])
    elif model_id == 4:  # best model from training whole network
        checkpoint_path = f"checkpoints/exercise=2C/loss_type=SGD/batch_size=32/20_iterations_best_acc80.pth"
        net = models.googlenet(pretrained=True)
        n_inputs = net.fc.in_features
        last_layer = nn.Linear(n_inputs, 3, bias=True)
        net.fc = last_layer
        device = torch.device('cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint['model'])

    network = net.to(torch.device("cpu"))
    network.eval()

    iterations = 100
    at_layer = 1 # layer at which we modify image to maximize outputs
    lr = 0.005 # learning rate
    octave_scale = 1.4 # image scale between octaves 1.4 and 1.1 nice results
    num_octaves = 10 # number of octaves

    # input image
    #input_image = './original dataset/images/maksssksksss0.png'
    #input_image = './test_data/mask_weared_incorrect/8.png'
    #input_image = './test_data/with_mask/14.png'
    input_image = './pobrane.png'
    image = Image.open(input_image).convert("RGB")

    # Define the model
    #layers = list(network.features.children())
    layers = list(network.children())[:50]


    for at_layer in range(len(layers) - 1):
        # layers = []
        # layers.append(network._modules.get('conv3'))
        model = nn.Sequential(*layers[: (at_layer + 1)])
        if torch.cuda.is_available:
            model = model.cuda()
        print(network)

        # Extract deep dream image
        dreamed_image = deep_dream(
            image,
            model,
            iterations=iterations,
            lr=lr,
            octave_scale=octave_scale,
            num_octaves=num_octaves,
        )
        im = Image.fromarray((dreamed_image * 255).astype(np.uint8))
        im.save("./dreams/" + str(at_layer) + ".png")



        '''
        # Save and plot image
        os.makedirs("outputs", exist_ok=True)
        filename = input_image.split("/")[-1]
        # plt.figure(figsize=(20, 20))
        plt.imshow(dreamed_image)
        # plt.imsave(f"outputs/output_{filename}", dreamed_image)
        plt.show()
        '''

