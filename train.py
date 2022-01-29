import copy
import logging
import os
import time

import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torchvision import models
import process_data_test
from torchsummary import summary
from utils.config import batch_size
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image

EXERCISE = "2B"


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def train_model(train_data, validate_data, num_epochs=30):
    checkpoints_dir = os.path.join(".", 'checkpoints', 'exercise={}'.format(EXERCISE),
                                   'loss_type={}'.format("SGD"),
                                   'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device='cpu'
    old_model = models.googlenet(pretrained=True)
    old_model = old_model.to(device)
    n_inputs = old_model.fc.in_features  # refer to the fully connected layer only
    # use_cuda = torch.cuda.is_available()
    # if use_cuda:
    #     model = model.cuda()
    # Add last linear layer (n_inputs -> 4 classes). In this case the ouput is 4 classes
    # New layer automatically has requires_grad = True
    # for child in old_model.children():
    #     print(child)
    # print(summary(old_model, (3, 226, 226)))
    old_model.inception5a = nn.Sequential()
    old_model.inception5b = nn.Sequential()

    last_layer = nn.Linear(832, len(process_data_test.options.items()), bias=True)
    old_model.fc = last_layer
    model = old_model
    model.to(device)
    # old_model.to(device)
    # print(summary(old_model, (3, 226, 226)))
    # model = nn.Sequential(*(list(old_model.children()))[:14])
    # model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    # model.dropout = nn.Dropout(0.2)
    # model.fc = nn.Linear(1664, 3)
    # model.to(device)
    # print(summary(model, (3, 226, 226)))
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1.1, 0.0437, 0.239]))
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            if phase == "train":
                batches_number = 0
                for inputs, labels in train_data:
                    batches_number += 1
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, dim=1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase

                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data)

                scheduler.step()
                epoch_loss = running_loss / batches_number  # TODO
                epoch_acc = running_corrects.double() / batches_number

            else:
                valid_batches_number = 0
                for valid_inputs, valid_labels in validate_data:
                    valid_batches_number += 1
                    valid_inputs = valid_inputs.to(device)
                    valid_labels = valid_labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        valid_outputs = model(valid_inputs)
                        _, valid_preds = torch.max(valid_outputs, 1)
                        loss = criterion(valid_outputs, valid_labels)

                    # statistics
                    running_loss += loss.item()
                    running_corrects += torch.sum(valid_preds == valid_labels.data)

                epoch_loss = running_loss / valid_batches_number  # TODO
                epoch_acc = running_corrects.double() / valid_batches_number

            print('{} Loss: {:.4f} '.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        # Save model
        if epoch % 10 == 0 and epoch > 0:
            checkpoint = {
                'iteration': epoch,
                'model': model.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(epoch))

            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    checkpoint = {
        'iteration': epoch,
        'model': best_model_wts}

    checkpoint_path = os.path.join(
        checkpoints_dir, '_best_model.pth')

    torch.save(checkpoint, checkpoint_path)
    logging.info('Model saved to {}'.format(checkpoint_path))

    return model


def test_model(iteration):
    checkpoint_path = f"checkpoints/exercise={EXERCISE}/loss_type=SGD/batch_size=32/{iteration}_iterations.pth"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = models.googlenet(pretrained=True)

    n_inputs = model.fc.in_features  # refer to the fully connected layer only
    # model.inception5a = nn.Sequential()
    # model.inception5b = nn.Sequential()

    # last_layer = nn.Linear(832, len(process_data_test.options.items()), bias=True)
    # model.fc = last_layer
    last_layer = nn.Linear(n_inputs, len(process_data_test.options.items()), bias=True)
    model.fc = last_layer

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(model)
    # Parallel
    if 'cuda' in str(device):
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
    else:
        print('Using CPU.')

    directory = "test_data"
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    subfolders = [os.path.relpath(x, directory) for x in subfolders]
    y = []
    options = {'mask_weared_incorrect': 0, 'with_mask': 1, 'without_mask': 2}
    my_transform = transforms.Compose([transforms.Resize((226, 226)),
                                       transforms.ToTensor()])
    y_pred = []
    for i in subfolders:
        label = i
        for x in os.listdir(os.path.join(directory, i)):
            y.append(options[label])
            img = Image.open(os.path.join(directory, i, x)).convert('RGB')
            img_t = my_transform(img)
            img_t = img_t.to(device)
            batch_t = torch.unsqueeze(img_t, 0).cuda()
            with torch.no_grad():
                model.eval()
                out = model(batch_t)
                probabilities = torch.nn.functional.softmax(out, dim=1)
                y_pred.append(probabilities)

    binary_labels = []
    for labels in y_pred:
        a = torch.argmax(labels)
        binary_labels.append(int(a))

    class_names = ['mask weared\nincorrect', 'with mask', 'without mask']
    class_nam = [0, 1, 2]
    accuracy = accuracy_score(y, binary_labels)
    f1 = f1_score(y, binary_labels, average='weighted')
    print('Accuracy : ', "%.2f" % (accuracy * 100))
    print('F1 : ', "%.2f" % (f1 * 100))
    cm = confusion_matrix(y, binary_labels, labels=class_nam)
    plt.figure(figsize=(15,15))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.title(f"Acc={round(accuracy*100, 2)}, F1={round(f1*100,2)}")
    plt.savefig(f'CM/confiusion_matrix_exercise_{EXERCISE}_iteration_{iteration}.png')
    plt.close()


if __name__ == '__main__':
    # train_dataset, valid_dataset = process_data_test.splitting_dataset_training_test()
    # model = train_model(train_dataset, valid_dataset, 1000)
    for i in range(10, 100, 10):
        test_model(i)
