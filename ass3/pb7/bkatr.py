import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.nn.functional as F
import os
from torchvision.utils import save_image
from PIL import Image


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # (in_channel (# feature maps per input image), out_channel (# feature maps per output image), size)
        self.pool = nn.MaxPool2d(2, 2) # 2*2 kernel size with stride=(2,2)
        self.avgp = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5) # 16 groups of filters changing channel number to 16, with kernel size 5*5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch (size flatten and channel flatten)
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_data(args):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # transform像设置？干啥用的？？

    trainset = torchvision.datasets.CIFAR10(root=args.dataroot, train=True,
                                            download=True, transform=transform) 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train,
                                              shuffle=True, num_workers=2) # shuffle?? num_workers???

    testset = torchvision.datasets.CIFAR10(root=args.dataroot, train=False,  
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test,
                                             shuffle=False, num_workers=2) # shuffle?? num_workers???

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

def train(args):
    k=200
  
    trainloader, testloader, classes = load_data(args)
    
    device = torch.device("cuda:0" if args.is_gpu else "cpu")
    
    net = Net()
    net.to(device)
    
    # HHH loss函数可改
    criterion = nn.CrossEntropyLoss()
    # HHH 学习率可动态改
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum) # momentum???

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        print("starting epoch: ", epoch)
        running_loss = 0.0

        # iterating per batch for training
        # batch_size=4 => 一次输入四张图片
        for i, data in enumerate(trainloader, 0): # i: batch number
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs) # forward propagation
            loss = criterion(outputs, labels) # calculate loss

            # 这两步在做啥？？
            loss.backward() 
            optimizer.step()

            # print statistics
            running_loss += loss.item() # 为啥是加而不是直接更新赋值？？？

            # HHH 累积batch误差数量可改
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[epoch num: %d, batch num: %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0 # 重置loss？？
                
        # training finished
        # testing starts, only 1 epoch
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device) # input image pixels, label
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1) # output包含data和分类结果
                total += labels.size(0)
                correct += (predicted == labels).sum().item() # sum没用吧 一张张图迭代的话
                if (k > 0):
                  k = k - 1
                  for i in range(4):
                    temp = 255*((inputs[i] + 1)/2)
                    im = Image.fromarray(temp)
					im.save(filename=("/group12/cloudComputing2021/CODE/Image_Classification/clares/pg" + str(predicted[i]) + str(labels[i] + ".png")))
                    #save_image(tensor=temp, filename=("/group12/cloudComputing2021/CODE/Image_Classification/clares/pg" + str(predicted[i]) + str(labels[i] + ".png")), normalize=True)
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001 , help='learning rate')
    parser.add_argument('--epochs', type=int, default=5 , help='number of training epoch')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size_train', type=int, default=4)
    parser.add_argument('--batch_size_test', type=int, default=4)

    parser.add_argument('--dataroot', type=str, default='/group12/cloudComputing2021/DATA/Cifar-10')
    parser.add_argument('--is_gpu', type=bool, default=False)
    args = parser.parse_args()
    print(args)
    train(args)

# HH 更换优化器
# HH 设置bias
# HH 改stride 
# HH 做dilation
# HH 加上bias

