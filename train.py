import copy
import logging
import os
import time

import torch
from torch import nn, optim
from torchvision import models
import process_data_test
from utils.config import batch_size
from torch.optim import lr_scheduler

EXERCISE = "2A"


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def train_model(train_data, validate_data, num_epochs=200):
    checkpoints_dir = os.path.join(".", 'checkpoints', 'exercise={}'.format(EXERCISE),
                                   'loss_type={}'.format("SGD"),
                                   'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)

    model = models.googlenet(pretrained=True)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    print(model)
    # for param in model.parameters():
    #     param.requires_grad = False

    # model.param.requires_grad = True
    ct = 0

    for child in model.children():
        ct += 1
        if ct < 16:
            for param in child.parameters():
                param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = True

    # Define n_inputs takes the same number of inputs from pre-trained model
    n_inputs = model.fc.in_features  # refer to the fully connected layer only

    # Add last linear layer (n_inputs -> 4 classes). In this case the ouput is 4 classes
    # New layer automatically has requires_grad = True
    last_layer = nn.Linear(n_inputs, len(process_data_test.options.items()), bias=True)
    model.fc = last_layer
    # If GPU is available, move the model to GPU
    if use_cuda:
        model = model.cuda()

    print(model.fc.out_features)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
                        _, preds = torch.max(outputs, 1)
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

            print('{} Loss: {:.4f} Good predictions per batch: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        # Save model
        if epoch % 50 == 0 and epoch > 0:
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


if __name__ == '__main__':
    values, Items, img_names = process_data_test.extractions_images_and_annotatons()
    dataset = process_data_test.dataset_creation(img_names)
    train_dataset, valid_dataset = process_data_test.splitting_dataset_training_test(dataset)
    model = train_model(train_dataset, valid_dataset, 500)
