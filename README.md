# CIFAR 10 Image Classification
Reference and baseline: [Training a Classifier](http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

## Architecture

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, WIDTH, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(WIDTH, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

## Hyperparamters

* batch_size = 16
* lr = 0.001
* momentum = 0.9
* epoch = 20
* WIDTH = {6, 12, 18}


## Tuning the width of neural network

| Epoch | 6 | 12 | 18 |
|---|---|---|---|
|1||||
|2||||
|3||||
|4||||
|5||||
|6||||
|7||||
|8||||
|9||||
|10||||
|11||||
|12||||
|13||||
|14||||
|15||||
|16||||
|17||||
|18||||
|19||||
|20||||
