import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.nn.functional as F
import os

# 网络优化思路：
# dataset(aug...), conv output channel up, pooling method change, optimizer change 

# max pool + avg pool, middle layer uses 24
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(24, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.avgpool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_data(args):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=args.dataroot, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=args.dataroot, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

def train(args):
    
    trainloader, testloader, classes = load_data(args)
    
    device = torch.device("cuda:0" if args.is_gpu else "cpu")
    
    net = Net()
    net.to(device)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        print("starting epoch: ", epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
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
    parser.add_argument('--is_gpu', type=bool, default=True)
    args = parser.parse_args()
    print(args)
    train(args)





# NN class and diagram.
# experiment: epoch=5, 10, 20; MaxMax MaxAvg AvgAvg; lr=0.0001, 0.0003, 0.0005; conv output channel=6, 12, 24
# default prog: training time, accuracy, classifying result (of testing dataset, 10 classes) 
# 
#