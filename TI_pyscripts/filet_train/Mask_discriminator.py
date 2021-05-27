import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader



train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)



from torch.utils.data import DataLoader


size = (28,28)
conv_params = []
conv_params.append([1,64,5,2])
conv_params.append([64,32,3,2])
h_in,w_in = size
dims = np.array(size)
Cout = 0
for params in conv_params:
        dims = np.floor((dims - params[2] ) / params[3] + 1)
        print(dims)
        Cout = params[1]
class Net(nn.Module):
    def __init__(self):
        #do ModuleDict
        super(Net, self).__init__()
        # 1 input image channel, 64 output channels, 5x5 square convolution
        # kernel
        self.conv_layers = []
        for i,params in enumerate(conv_params):
            self.cl = nn.Conv2d(*params)
            self.conv_layers.append(self.cl)
        self.do1 = nn.Dropout()
        self.fc1 = nn.Linear(Cout*int(np.product(dims)),200)  # 5*5 from image dimension
        self.do2 = nn.Dropout(.25)
        self.fc2 = nn.Linear(200, 200)
        self.do3 = nn.Dropout(.25)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, x):
        for layer in self.conv_layers:
            x = F.relu(layer(x))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(self.do1(x)))
        x = F.relu(self.fc2(self.do2(x)))
        x = torch.sigmoid(self.fc3(self.do3(x)))
        return x

#loop:
is_training = True
train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True,num_workers=2)
train_iter = train_dataloader.__iter__()

loss_fun = nn.BCELoss()
net = Net()
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.6,nesterov=True)

for itr in range(1):
  batch,targets = next(train_iter)
  targets=targets.float()
  with torch.set_grad_enabled(is_training):
      out=net(batch)
      targets = targets.unsqueeze(1)
      print(targets)
      loss = loss_fun(out,targets)
  print(loss)
  if is_training:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()