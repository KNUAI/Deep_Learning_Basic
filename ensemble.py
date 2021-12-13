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
batch_size = 128
learning_rate = 0.001
root_dir = 'drive/app/cifar10/'
default_directory = 'drive/app/torch/save_models'

# Data Augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),                              # change [0,255] Int value to [0,1] Float value
    transforms.Normalize(mean=(0.4914, 0.4824, 0.4467), # RGB Normalize MEAN
                         std=(0.2471, 0.2436, 0.2616))  # RGB Normalize Standard Deviation
])

# automatically download
test_dataset = datasets.CIFAR10(root=root_dir,
                                train=False,
                                transform=transform_test)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,            # at Test Procedure, Data Shuffle = False
                                          num_workers=4)            # CPU loader number

# model
model1 = models.resnet152()
model2 = models.resnet152()
model3 = models.resnet152()
model1 = nn.DataParallel(model1).cuda()
model2 = nn.DataParallel(model2).cuda()
model3 = nn.DataParallel(model3).cuda()
cudnn.benchmark = True

# loss
criterion = nn.CrossEntropyLoss()

# load
model1.load_state_dict(torch.load('./drive/app/torch/save_models/latest_64_0.001_150.tar.gz')['state_dict'])
model2.load_state_dict(torch.load('./drive/app/torch/save_models/latest_128_0.01_150.tar.gz')['state_dict'])
model3.load_state_dict(torch.load('./drive/app/torch/save_models/latest_256_0.01_150.tar.gz')['state_dict'])

def test():
    model1.eval()
    model2.eval()
    model3.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)

        output1 = model1(data)
        output2 = model2(data)
        output3 = model3(data)
        outputs = (output1 + output2 + output3) / 3
        loss = criterion(outputs, target)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
    print('# TEST : Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
          .format(test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

test()

now = time.gmtime(time.time() - start_time)
print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))