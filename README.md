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
|1|29.56|34.40|37.38|39.56|39.15|41.29|40.78|
|2|34.39|44.99|44.62|47.76|49.24|48.91|48.84|
|3|40.89|49.67|52.57|51.74|55.23|53.65|56.78|
|4|43.00|51.03|53.69|57.77|59.82|58.46|60.03|
|5|44.98|53.08|53.40|60.05|61.24|63.30|63.43|
|6|47.58|56.38|60.84|61.95|64.43|63.85|65.38|
|7|50.16|56.31|61.75|63.72|64.33|65.21|66.52|
|8|49.33|57.75|62.12|65.35|65.32|68.66|66.30|
|9|51.27|57.68|63.60|66.53|67.08|68.71|67.32|
|10|52.43|58.81|64.93|65.82|66.84|69.51|67.96|
|11|53.50|59.85|64.63|67.54|67.96|69.05|69.21|
|12|53.59|58.48|64.57|68.99|68.19|69.03|68.46|
|13|53.42|59.52|66.43|67.28|70.17|70.18|70.39|
|14|53.87|59.34|64.96|67.44|68.52|70.04|69.03|
|15|54.43|58.95|65.67|69.02|69.08|68.76|70.55|
|16|54.39|60.14|66.44|68.51|68.64|70.99|69.84|
|17|52.69|60.22|69.95|68.40|69.13|70.09|70.81|
|18|53.51|59.95|65.40|68.19|69.09|71.89|70.55|
|19|54.36|58.21|64.92|68.23|69.32|71.53|71.19|
|20|54.25|60.00|66.09|68.35|69.42|71.21|70.62|
