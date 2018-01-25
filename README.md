# CIFAR 10 Image Classification
Reference: [Training a Classifier](http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

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

Note that WIDTH_2 = WIDTH_1 * 2 + 4

## Experiment-3: Tuning the widths of neural network

Performance is measured by the accuracy(%) on 10,000 test images.

| Epoch | (1,6) | (3,10) | (6,16) | (12,28) | (18,40) | (24,52) | (30,64) |
|---|---|---|---|---|---|---|---|
|1|16.6|35.76||39.76|42.83|42.27|40.84|
|2|34.04|44.45||49.93|49.81|51.40|52.96|
|3|39.21|47.48||54.75|55.51|57.91|58.79|
|4|40.86|49.49||57.78|63.02|62.19|60.61|
|5|43.09|52.90||61.78|64.92|65.24|64.90|
|6|43.79|56.39||64.66|66.47|67.92|67.90|
|7|44.90|56.95||63.44|68.01|69.30|70.77|
|8|42.72|57.39||66.56|69.03|69.54|71.22|
|9|46.57|57.69||66.83|69.56|70.81|71.43|
|10|46.99|59.44||67.20|71.01|70.56|72.42|
|11|47.02|57.97||67.87|70.20|72.25|71.91|
|12|47.02|59.57||68.37|69.96|70.89|72.07|
|13|48.17|59.04||68.63|71.00|70.64|72.08|
|14|48.09|59.46||68.34|70.92|71.31|72.66|
|15|49.02|58.77||68.80|70.43|71.92|71.68|
|16|47.81|60.23||68.82|70.44|71.84|72.23|
|17|49.30|58.90||68.79|70.58|71.71|70.92|
|18|48.69|59.25||68.52|70.90|71.89|71.44|
|19|49.76|59.15||68.57|69.68|71.66|71.17|
|20|48.80|59.92||67.76|70.23|70.49|71.43|

## Experiment-4: Architecture

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, WIDTH_1, 3, padding=1) 
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(WIDTH_1, WIDTH_2, 3, padding=1)
            self.conv3 = nn.Conv2d(WIDTH_2, WIDTH_3, 3, padding=1)
            self.fc1 = nn.Linear(WIDTH_3 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, WIDTH_3 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

## Experiment-4: Hyperparamters

* batch_size = 16
* lr = 0.001
* momentum = 0.9
* epoch = 20
* (WIDTH_1, WIDTH_2, WIDTH_3) = {(1, 2, 4), (2, 4, 8), (4, 8, 16), (8, 16, 32), (16, 32, 64), (32, 64, 128), (64, 128, 256)}

## Experiment-4: Tuning the widths of neural network

Performance is measured by the accuracy(%) on 10,000 test images.

| Epoch | (1,2,4) | (2,4,8) | (4,8,16) | (8,16,32) | (16,32,64) | (32,64,128) | (64,128,256) |
|---|---|---|---|---|---|---|---|
|1|10.00|10.00|19.48|26.73|32.62|36.20|40.26|
|2|11.96|12.69|35.99|39.56|46.84|48.67|50.40|
|3|31.08|32.16|42.65|46.98|54.76|58.38|58.98|
|4|36.01|36.53|48.06|55.56|57.53|62.22|65.04|
|5|39.99|40.19|51.34|58.78|64.15|66.82|69.91|
|6|41.92|44.96|54.17|61.17|66.03|68.44|71.40|
|7|43.53|47.02|56.22|63.00|68.02|71.57|73.15|
|8|44.22|48.64|54.43|65.51|69.63|72.02|76.01|
|9|45.84|50.78|58.58|68.18|70.59|74.16|75.91|
|10|44.12|51.09|59.27|67.47|71.36|73.34|76.30|
|11|45.82|52.40|60.10|68.59|71.05|75.48|76.24|
|12|45.22|52.88|61.26|69.23|71.69|74.86|75.45|
|13|47.04|51.58|61.32|68.99|71.51|74.60|76.83|
|14|46.41|53.48|62.64|69.59|72.77|75.13|76.86|
|15|48.31|53.70|63.41|70.28|71.76|75.04|76.76|
|16|48.21|53.83|63.92|70.36|72.43|74.18|76.55|
|17|46.27|54.66|63.70|70.66|72.89|75.76|75.51|
|18|47.56|55.18|62.84|70.37|72.02|75.25|76.02|
|19|46.56|54.77|63.07|70.13|71.88|75.18|76.48|
|20|47.62|54.52|63.84|71.19|72.74|74.52|76.90|

## Experiment-5: Architecture

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, WIDTH_1, 3, padding=1) 
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(WIDTH_1, WIDTH_2, 3, padding=1)
            self.conv3 = nn.Conv2d(WIDTH_2, WIDTH_3, 3, padding=1)
            self.conv4 = nn.Conv2d(WIDTH_3, WIDTH_4, 3, padding=1)
            self.fc1 = nn.Linear(WIDTH_4 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            x = x.view(-1, WIDTH_4 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

## Experiment-5: Hyperparamters

* batch_size = 16
* lr = 0.001
* momentum = 0.9
* epoch = 20
* (WIDTH_1, WIDTH_2, WIDTH_3, WIDTH_4) = {(1, 2, 4, 8), (2, 4, 8, 16), (4, 8, 16, 32), (8, 16, 32, 64), (16, 32, 64, 128), (32, 64, 128, 256), (64, 128, 256, 512)}

## Experiment-5: Tuning the widths of neural network

Performance is measured by the accuracy(%) on 10,000 test images.

| Epoch | (1,2,4,8) | (2,4,8,16) | (4,8,16,32) | (8,16,32,64) | (16,32,64,128) | (32,64,128,256) | (64,128,256,512) |
|---|---|---|---|---|---|---|---|
|1|10.01|10.00|19.01|11.68|30.98|29.30|38.13|
|2|10.72|17.62|39.40|33.73|43.04|45.19|50.57|
|3|30.59|33.10|45.53|43.84|52.56|56.12|59.61|
|4|38.96|42.05|51.67|50.29|59.73|61.71|67.29|
|5|42.23|47.48|56.98|57.65|66.39|66.45|72.17|
|6|45.93|48.81|58.46|60.58|69.75|70.62|73.80|
|7|45.96|51.27|60.93|63.88|71.40|73.43|74.02|
|8|48.95|53.96|64.02|67.86|72.87|74.59|77.31|
|9|49.26|57.84|65.53|69.37|74.75|72.68|77.67|
|10|50.86|58.57|66.59|69.78|75.52|75.45|75.50|
|11|52.89|59.62|68.67|71.60|75.18|76.60|78.06|
|12|53.50|60.01|67.73|71.11|74.73|76.67|77.64|
|13|54.87|62.11|69.48|71.58|75.89|75.50|77.93|
|14|54.35|61.54|67.56|72.02|74.64|74.88|77.04|
|15|55.98|63.47|69.86|72.65|75.24|74.76|77.34|
|16|55.73|62.78|70.19|72.92|75.93|76.37|78.34|
|17|56.31|64.36|69.75|72.21|75.42|75.77|77.72|
|18|56.95|63.90|70.60|72.90|75.65|76.38|77.27|
|19|58.14|63.78|69.77|72.06|73.99|76.26|78.59|
|20|55.34|64.13|69.64|73.37|74.74|75.93|76.21|

## Experiment-6: Architecture

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, WIDTH_1, 3, padding=1) 
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(WIDTH_1, WIDTH_2, 3, padding=1)
            self.conv3 = nn.Conv2d(WIDTH_2, WIDTH_3, 3, padding=1)
            self.conv4 = nn.Conv2d(WIDTH_3, WIDTH_4, 3, padding=1)
            self.conv4 = nn.Conv2d(WIDTH_4, WIDTH_5, 3, padding=1)
            self.fc1 = nn.Linear(WIDTH_5 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            x = self.pool(F.relu(self.conv5(x)))
            x = x.view(-1, WIDTH_5 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

## Experiment-6: Hyperparamters

* batch_size = 16
* lr = 0.001
* momentum = 0.9
* epoch = 20
* (WIDTH_1, WIDTH_2, WIDTH_3, WIDTH_4, WIDTH_5) = {(1, 2, 4, 8, 16), (2, 4, 8, 16, 32), (4, 8, 16, 32, 64), (8, 16, 32, 64, 128), (16, 32, 64, 128, 256), (32, 64, 128, 256, 512), (64, 128, 256, 512, 1024)}

## Experiment-6: Tuning the widths of neural network

Performance is measured by the accuracy(%) on 10,000 test images.

| Epoch | (1,2,4,8,16) | (2,4,8,16,32) | (4,8,16,32,64) | (8,16,32,64,128) | (16,32,64,128,256) | (32,64,128,256,512) | (64,128,256,512,1024) |
|---|---|---|---|---|---|---|---|
|1|10.00|10.00|10.00|10.00|10.00|26.15|29.28|
|2|10.00|10.00|10.00|22.17|29.64|42.70|43.31|
|3|10.00|27.40|14.51|39.31|44.78|52.81|54.66|
|4|10.00|40.56|30.07|47.78|51.16|62.07|64.44|
|5|10.00|44.91|39.13|54.95|59.66|67.21|69.30|
|6|10.00|50.53|46.03|58.47|65.65|70.81|72.40|
|7|10.00|54.81|50.75|64.08|65.47|73.35|74.09|
|8|10.00|57.34|54.45|67.38|70.79|73.41|74.90|
|9|10.00|61.48|57.70|69.79|71.80|73.16|75.70|
|10|10.00|64.30|62.65|71.03|72.33|72.57|76.13|
|11|10.00|63.34|65.47|69.59|72.07|73.30|75.74|
|12|10.00|67.22|65.49|72.10|73.04|73.69|74.87|
|13|10.00|66.46|68.13|71.27|73.11|73.98|75.61|
|14|10.00|67.16|70.51|72.47|71.89|75.20|76.13|
|15|10.07|68.19|70.31|71.28|71.97|74.65|75.95|
|16|26.12|68.70|70.72|71.33|73.05|74.68|75.64|
|17|28.13|68.27|71.77|71.91|71.93|75.78|76.99|
|18|33.42|68.43|70.86|73.02|73.02|75.27|76.72|
|19|35.99|67.97|72.50|71.66|72.95|75.91|77.38|
|20|40.88|69.02|72.43|71.46|72.59|76.20|76.86|

## Experiment-7: Architecture

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 4 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 10),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 256 * 4 * 4)
            x = self.classifier(x)
            return x

## Experiment-7: Hyperparamters

* batch_size = 16
* lr = 0.001
* momentum = 0.9
* epoch = 20
* (WIDTH_1, WIDTH_2, WIDTH_3, WIDTH_4) = {(1,3,6,4), (2,6,12,8), (4,12,24,16), (8,24,48,32), (16,48,96,64), (32,96,192,128), (64,192,384,256)}

## Experiment-7: Tuning the widths of neural network

Performance is measured by the accuracy(%) on 10,000 test images.

| Epoch | (1,3,6,4) | (2,6,12,8) | (4,12,24,16) | (8,24,48,32) | (16,48,96,64) | (32,96,192,128) | (64,192,384,256) |
|---|---|---|---|---|---|---|---|
|1|16.86||||||22.39|
|2|23.04||||||39.50|
|3|27.14||||||48.57|
|4|27.27||||||54.25|
|5|30.16||||||58.65|
|6|30.40||||||62.60|
|7|31.30||||||65.27|
|8|32.20||||||68.53|
|9|32.53||||||71.33|
|10|32.78||||||71.56|
|11|32.97||||||73.37|
|12|32.46||||||73.22|
|13|33.78||||||75.29|
|14|34.56||||||76.05|
|15|35.89||||||76.65|
|16|35.36||||||74.91|
|17|36.22||||||76.65|
|18|35.63||||||74.73|
|19|34.82||||||76.05|
|20|36.43||||||76.77|

## Experiment-8: Architecture

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=5),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 4 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 10),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 256 * 4 * 4)
            x = self.classifier(x)
            return x

## Experiment-8: Hyperparamters

* batch_size = 16
* lr = 0.001
* momentum = 0.9
* epoch = 20

## Experiment-8: Result

Performance is measured by the accuracy(%) on 10,000 test images.

| Epoch | Performance |
|---|---|
|1|10.19|
|2|9.96|
|3|10.19|
|4|10.12|
|5|10.13|
|6|10.04|
|7|10.00|
|8|10.00|
|9|10.00|
|10|10.08|
|11|9.99|
|12|10.37|
|13|10.01|
|14|10.00|
|15|9.97|
|16|10.00|
|17|9.94|
|18|10.11|
|19|10.00|
|20|9.91|

## Experiment-9: Architecture

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 4 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 10),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 256 * 4 * 4)
            x = self.classifier(x)
            return x

## Experiment-9: Hyperparamters

* batch_size = 16
* lr = 0.001
* momentum = 0.9
* epoch = 20

## Experiment-9: Result

Performance is measured by the accuracy(%) on 10,000 test images.

| Epoch | Performance |
|---|---|
|1|10.00|
|2|10.01|
|3|10.00|
|4|10.00|
|5|10.00|
|6|9.98|
|7|10.00|
|8|10.02|
|9|11.59|
|10|27.68|
|11|38.61|
|12|42.76|
|13|54.07|
|14|56.17|
|15|64.90|
|16|68.12|
|17|70.16|
|18|71.54|
|19|73.29|
|20|75.30|
|21|76.95|
|22|76.05|
|23|76.95|
|24|77.33|
|25|79.07|
|26|77.41|
|27|79.28|
|28|80.34|
|29|77.94|
|30|80.14|
|31|80.23|
|32|79.37|
|33|81.00|
|34|79.88|
|35|79.78|
|36|79.24|
|37|80.97|
|38|80.85|
|39|80.48|
|40|80.96|
|41|80.72|
|42|80.49|
|43|79.56|
|44|80.41|
|45|80.54|
|46|81.08|
|47|80.76|
|48|79.76|
|49|80.81|
|50|81.34|
|51|81.04|
|52|80.30|
|53|80.47|
|54|81.45|
|55|80.77|
|56|80.84|
|57|82.32|
|58|81.67|
|59|82.30|
|60|81.08|

## Experiment-10: Architecture

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512 * 4 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 10)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 512 * 4 * 4)
            x = self.classifier(x)
            return x

## Experiment-10: Hyperparamters

* batch_size = 16
* lr = 0.001
* momentum = 0.9
* epoch = 20

## Experiment-10: Result

Performance is measured by the accuracy(%) on 10,000 test images.

| Epoch | Performance |
|---|---|
|1|12.39|
|2|27.44|
|3|43.43|
|4|52.84|
|5|60.99|
|6|64.16|
|7|68.07|
|8|72.50|
|9|74.16|
|10|76.33|
|11|76.21|
|12|77.79|
|13|78.88|
|14|77.25|
|15|78.82|
|16|79.62|
|17|80.99|
|18|80.26|
|19|80.86|
|20|80.85|

## Experiment-11: Architecture

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.feature_1 = nn.Sequential(
                nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
            )
            self.classifier_1 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(1 * 4 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 10)
            )
            self.feature_2 = nn.Sequential(
                nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
            )
            self.classifier_2 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(2 * 4 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 10)
            )
            self.feature_3 = nn.Sequential(
                nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
            )
            self.classifier_3 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(4 * 4 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 10)
            )
            self.feature_4 = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
            )
            self.classifier_4 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(8 * 4 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 10)
            )
            self.feature_5 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
            )
            self.classifier_5 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(16 * 4 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 10)
            )
            self.feature_6 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
            )
            self.classifier_6 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(32 * 4 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 10)
            )
            self.feature_7 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
            )
            self.classifier_7 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(64 * 4 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 10)
            )
            self.feature_8 = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
            )
            self.classifier_8 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(128 * 4 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 10)
            )
            self.feature_9 = nn.Sequential(
                nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
            )
            self.classifier_9 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 4 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 10)
            )
            self.feature_10 = nn.Sequential(
                nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
            )
            self.classifier_10 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512 * 4 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 10)
            )

        def forward(self, x):
            x_1 = self.feature_1(x)
            x_1 = x_1.view(x_1.size(0), 1 * 4 * 4)
            x_1 = self.classifier_1(x_1)
            x_2 = self.feature_2(x)
            x_2 = x_2.view(x_2.size(0), 2 * 4 * 4)
            x_2 = self.classifier_2(x_2)
            x_3 = self.feature_3(x)
            x_3 = x_3.view(x_3.size(0), 4 * 4 * 4)
            x_3 = self.classifier_3(x_3)
            x_4 = self.feature_4(x)
            x_4 = x_4.view(x_4.size(0), 8 * 4 * 4)
            x_4 = self.classifier_4(x_4)
            x_5 = self.feature_5(x)
            x_5 = x_5.view(x_5.size(0), 16 * 4 * 4)
            x_5 = self.classifier_5(x_5)
            x_6 = self.feature_6(x)
            x_6 = x_6.view(x_6.size(0), 32 * 4 * 4)
            x_6 = self.classifier_6(x_6)
            x_7 = self.feature_7(x)
            x_7 = x_7.view(x_7.size(0), 64 * 4 * 4)
            x_7 = self.classifier_7(x_7)
            x_8 = self.feature_8(x)
            x_8 = x_8.view(x_8.size(0), 128 * 4 * 4)
            x_8 = self.classifier_8(x_8)
            x_9 = self.feature_9(x)
            x_9 = x_9.view(x_9.size(0), 256 * 4 * 4)
            x_9 = self.classifier_9(x_9)
            x_10 = self.feature_10(x)
            x_10 = x_10.view(x_10.size(0), 512 * 4 * 4)
            x_10 = self.classifier_10(x_10)
            x = (x_1+x_2+x_3+x_4+x_5+x_6+x_7+x_8+x_9+x_10)/10
            return x

## Experiment-11: Hyperparamters

* batch_size = 16
* lr = 0.001
* momentum = 0.9
* epoch = 20

## Experiment-11: Result

Performance is measured by the accuracy(%) on 10,000 test images.

| Epoch | Performance |
|---|---|
|1|13.43|
|2|28.68|
|3|37.15|
|4|41.56|
|5|45.12|
|6|48.09|
|7|48.46|
|8|51.73|
|9|54.75|
|10|58.17|
|11|59.51|
|12|61.88|
|13|64.83|
|14|66.67|
|15|68.90|
|16|69.00|
|17|70.71|
|18|71.73|
|19|72.86|
|20|73.03|
|21|74.56|
|22|74.05|
|23|74.94|
|24|75.66|
|25|75.00|
|26|76.33|
|27|76.00|
|28|76.85|
|29|77.27|
|30|77.24|
|31|76.77|
|32|77.20|
|33|77.67|
|34|77.30|
|35|77.70|
|36|77.51|
|37|76.86|
|38|78.24|
|39|77.76|
|40|78.47|

## Experiment-12: Architecture

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512 * 4 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 10)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 512 * 4 * 4)
            x = self.classifier(x)
            return x

## Experiment-12: Hyperparamters

* batch_size = 16
* lr = 0.001
* momentum = 0.9
* epoch = 20

## Experiment-12: Result

Performance is measured by the accuracy(%) on 10,000 test images.

| Epoch | Performance |
|---|---|
|1||
|2||
|3||
|4||
|5||
|6||
|7||
|8||
|9||
|10||
|11||
|12||
|13||
|14||
|15||
|16||
|17||
|18||
|19||
|20||
