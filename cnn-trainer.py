
import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation

import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
from network import Net


def imshow(img):
  ''' function to show image '''
  img = img / 2 + 0.5 # unnormalize
  npimg = img.numpy() # convert to numpy objects
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()
   
def run():
    
    #print(torch.cuda.is_available())
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
   
    # Atencao: para instalar com suporte a CUDA -> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    # fique atendo a versao do seu drive. Maiores detalhes: https://telin.ugent.be/telin-docs/windows/pytorch/

    
    torch.multiprocessing.freeze_support()
    # python image library of range [0, 1] 
    # transform them to tensors of normalized range[-1, 1]
    transform = transforms.Compose( # composing several transforms together
        [transforms.ToTensor(), # to tensor object
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean = 0.5, std = 0.5

    # set batch_size
    batch_size = 4

    # set number of workers
    num_workers = 2

    # load train data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    # put 10 classes into a set
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get random training images with iter function
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # call function on our images
    imshow(torchvision.utils.make_grid(images))

    # print the class of the image
    print(' '.join('%s' % classes[labels[j]] for j in range(batch_size)))

    net = Net()
    print(net)
    
    criterion = nn.CrossEntropyLoss()
    # optim.SGD → Implements stochastic gradient descent
    # lr = learning rate
    # momentum helps accelerate gradient vectors in the right directions, which leads to faster converging
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 
    
    for epoch in range(3):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            # “backward” is PyTorch’s way to perform backpropagation by computing the gradient based on the loss.
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    
    # save
    PATH = './cnn-demo-2.pth'
    torch.save(net.state_dict(), PATH)   


if __name__ == '__main__':
    run()