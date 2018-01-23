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

## Experiment-2: Architecture

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, WIDTH_1, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(WIDTH_1, WIDTH_2, 5)
            self.fc1 = nn.Linear(WIDTH_2 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, WIDTH_2 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

## Experiment-2: Hyperparamters

* batch_size = 16
* lr = 0.001
* momentum = 0.9
* epoch = 20
* (WIDTH_1, WIDTH_2) = {(1, 11), (3, 13), (6, 16), (12, 22), (18, 28), (24, 34), (30, 40)}

Note that WIDTH_1 = WIDTH_2 - 10

## Experiment-2: Tuning the widths of neural network

Performance is measured by the accuracy(%) on 10,000 test images.

| Epoch | (1,11) | (3,13) | (6,16) | (12,22) | (18,28) | (24,34) | (30,40) |
|---|---|---|---|---|---|---|---|
|1|28.93|37.19||41.57|39.93|43.02|43.58|
|2|33.00|44.42||50.17|50.18|51.74|52.26|
|3|38.46|48.29||53.49|55.04|57.39|59.48|
|4|41.60|50.45||58.46|59.65|61.36|62.89|
|5|41.92|51.43||60.31|61.97|62.72|64.92|
|6|44.61|53.16||62.93|65.46|64.80|66.15|
|7|47.25|55.73||64.69|67.31|67.39|68.77|
|8|46.21|56.16||66.21|67.17|68.87|68.74|
|9|48.43|57.08||66.45|68.74|70.29|70.29|
|10|48.98|57.18||68.16|69.33|70.74|70.68|
|11|49.93|57.97||67.40|70.03|69.68|70.36|
|12|50.25|59.31||68.20|68.95|71.16|71.87|
|13|49.98|59.31||68.74|69.99|71.67|71.46|
|14|49.62|59.56||68.03|69.50|70.92|72.13|
|15|49.47|60.33||67.71|69.39|70.60|72.12|
|16|51.55|60.28||68.35|69.92|69.69|71.15|
|17|50.60|58.96||67.96|70.15|71.40|71.66|
|18|51.96|60.99||68.62|69.77|70.91|71.10|
|19|51.04|60.74||67.55|69.93|71.37|70.50|
|20|51.03|60.53||68.42|69.48|70.30|70.42|

## Experiment-3: Architecture

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, WIDTH_1, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(WIDTH_1, WIDTH_2, 5)
            self.fc1 = nn.Linear(WIDTH_2 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, WIDTH_2 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

## Experiment-3: Hyperparamters

* batch_size = 16
* lr = 0.001
* momentum = 0.9
* epoch = 20
* (WIDTH_1, WIDTH_2) = {(1, 6), (3, 10), (6, 16), (12, 28), (18, 40), (24, 52), (30, 64)}

Note that WIDTH_1 = WIDTH_2 * 2 + 4

## Experiment-3: Tuning the widths of neural network

Performance is measured by the accuracy(%) on 10,000 test images.

| Epoch | (1,6) | (3,10) | (6,16) | (12,28) | (18,40) | (24,52) | (30,64) |
|---|---|---|---|---|---|---|---|
|1|16.6|||||||
|2|34.04|||||||
|3|39.21|||||||
|4|40.86|||||||
|5|43.09|||||||
|6|43.79|||||||
|7|44.90|||||||
|8|42.72|||||||
|9|46.57|||||||
|10|46.99|||||||
|11|47.02|||||||
|12|47.02|||||||
|13|48.17|||||||
|14|48.09|||||||
|15|49.02|||||||
|16|47.81|||||||
|17|49.30|||||||
|18|48.69|||||||
|19|49.76|||||||
|20|48.80|||||||
