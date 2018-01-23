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
* WIDTH = {6, 12, 18, 24}


## Tuning the width of neural network

Performance is measured by the accuracy(%) on 10,000 test images.

| Epoch | 6 | 12 | 18 | 24 |
|---|---|---|---|---|
|1|||39.15|41.29|
|2|||49.24|48.91|
|3|||55.23|53.65|
|4|||59.82|58.46|
|5|||61.24|63.30|
|6|||64.43|63.85|
|7|||64.33|65.21|
|8|||65.32|68.66|
|9|||67.08|68.71|
|10|||66.84|69.51|
|11|||67.96|69.05|
|12|||68.19|69.03|
|13|||70.17|70.18|
|14|||68.52|70.04|
|15|||69.08|68.76|
|16|||68.64|70.99|
|17|||69.13|70.09|
|18|||69.09|71.89|
|19|||69.32|71.53|
|20|||69.42|71.21|
