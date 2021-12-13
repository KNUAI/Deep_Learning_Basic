import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
import os
import torch.backends.cudnn as cudnn
import torchvision.models as models

seed = 2021
if seed is not None:
    import random
    import numpy as np
    import torch
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

os.environ["CUDA_VISIBLE_DEVICES"] = '0'                # GPU Number 
start_time = time.time()
batch_size = 256
learning_rate = 1e-2
epochs = 150
root_dir = 'drive/app/cifar10/'
default_directory = 'drive/app/torch/save_models'

# Data Augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),               # Random Position Crop
    transforms.RandomHorizontalFlip(),                  # right and left flip
    transforms.ToTensor(),                              # change [0,255] Int value to [0,1] Float value
    transforms.Normalize(mean=(0.4914, 0.4824, 0.4467), # RGB Normalize MEAN
                         std=(0.2471, 0.2436, 0.2616))  # RGB Normalize Standard Deviation
])

transform_test = transforms.Compose([
    transforms.ToTensor(),                              # change [0,255] Int value to [0,1] Float value
    transforms.Normalize(mean=(0.4914, 0.4824, 0.4467), # RGB Normalize MEAN
                         std=(0.2471, 0.2436, 0.2616))  # RGB Normalize Standard Deviation
])

# automatically download
train_dataset = datasets.CIFAR10(root=root_dir,
                                 train=True,
                                 transform=transform_train,
                                 download=True)

test_dataset = datasets.CIFAR10(root=root_dir,
                                train=False,
                                transform=transform_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,            # at Training Procedure, Data Shuffle = True
                                           num_workers=4)           # CPU loader number

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,            # at Test Procedure, Data Shuffle = False
                                          num_workers=4)            # CPU loader number

# model
model = models.resnet152(pretrained=True)

# optimizer
optimizer = optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# loss
criterion = nn.CrossEntropyLoss()

if torch.cuda.device_count() > 0:
    print("USE", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model).cuda()
    cudnn.benchmark = True
else:
    print("USE ONLY CPU!")


def train(epoch):
    model.train()
    train_loss = 0 
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)

        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        if batch_idx % 10 == 0:
            print('Epoch: {} | Batch_idx: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)

        outputs = model(data)
        loss = criterion(outputs, target)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
    print('# TEST : Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
          .format(test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return test_loss / (batch_idx + 1), 100. * correct / total


def save_checkpoint(directory, state, filename=f'latest_{batch_size}_{learning_rate}_{epochs}.tar.gz'):

    if not os.path.exists(directory):
        os.makedirs(directory)

    model_filename = os.path.join(directory, filename)
    torch.save(state, model_filename)
    print("=> saving checkpoint")

loss_list = []
acc_list = []
for epoch in range(epochs):

    if epoch < 50:
        lr = learning_rate
    elif epoch < 100:
        lr = learning_rate * 0.1
    else:
        lr = learning_rate * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    train(epoch)
    save_checkpoint(default_directory, {
        'epoch': epoch,
        'model': model,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    })
    out_loss, out_acc = test()
    loss_list.append(out_loss)
    acc_list.append(out_acc)

now = time.gmtime(time.time() - start_time)
print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))
'''
import os
import matplotlib.pyplot as plt
if not os.path.exists('./picture/'):
    os.makedirs('./picture/')
plt.plot(range(len(loss_list)), loss_list)
plt.xlabel('Test Epoch')
plt.ylabel('Test Loss')
plt.savefig('./picture/loss.png')
plt.close()

plt.plot(range(len(acc_list)), acc_list)
plt.xlabel('Test Epoch')
plt.ylabel('Test Accuracy')
plt.savefig('./picture/acc.png')
plt.close()
'''