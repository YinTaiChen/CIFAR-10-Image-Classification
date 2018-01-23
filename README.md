# CIFAR 10 Image Classification
Reference and baseline: [Training a Classifier](http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

## Experiment-1: Architecture

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

## Experiment-1: Hyperparamters

* batch_size = 16
* lr = 0.001
* momentum = 0.9
* epoch = 20
* WIDTH = {1, 3, 6, 12, 18, 24, 30}


## Experiment-1: Tuning the width of neural network

Performance is measured by the accuracy(%) on 10,000 test images.

| Epoch | 1 | 3 | 6 | 12 | 18 | 24 | 30 |
|---|---|---|---|---|---|---|---|
|1|29.56|34.40|||39.15|41.29|40.78|
|2|34.39|44.99|||49.24|48.91|48.84|
|3|40.89|49.67|||55.23|53.65|56.78|
|4|43.00|51.03|||59.82|58.46|60.03|
|5|44.98|53.08|||61.24|63.30|63.43|
|6|47.58|56.38|||64.43|63.85|65.38|
|7|50.16|56.31|||64.33|65.21|66.52|
|8|49.33|57.75|||65.32|68.66|66.30|
|9|51.27|57.68|||67.08|68.71|67.32|
|10|52.43|58.81|||66.84|69.51|67.96|
|11|53.50|59.85|||67.96|69.05|69.21|
|12|53.59|58.48|||68.19|69.03|68.46|
|13|53.42|59.52|||70.17|70.18|70.39|
|14|53.87|59.34|||68.52|70.04|69.03|
|15|54.43|58.95|||69.08|68.76|70.55|
|16|54.39|60.14|||68.64|70.99|69.84|
|17|52.69|60.22|||69.13|70.09|70.81|
|18|53.51|59.95|||69.09|71.89|70.55|
|19|54.36|58.21|||69.32|71.53|71.19|
|20|54.25|60.00|||69.42|71.21|70.62|
