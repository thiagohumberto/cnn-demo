
import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation

import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
from torchvision import datasets
from network import Net
    
def imshow2(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    plt.show()    
   
def run():
   
    torch.multiprocessing.freeze_support()
    
    # possibible classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # load model into our network
    PATH = './cnn-demo.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))
            
    root_dir = "data/testclass" 
    transform = transforms.Compose([transforms.Resize((32, 32)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.3, 0.3, 0.3), (0.3, 0.3, 0.3))])
    
    dataset = datasets.ImageFolder(root_dir, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=32, 
                                            shuffle=True)
    
    images,label = next(iter(dataloader))
    imshow2(images[0], normalize=False) 
    
    # Executing network
    outputs = net(images) 

    # Collect the max outout
    _,predicted = torch.max(outputs, 1)
    
    # Get class for it
    #print(predicted[0])
    print('Predicted: ', (classes[predicted[0]]))

if __name__ == '__main__':
    run()