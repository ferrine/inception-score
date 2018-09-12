import numpy, os

import torchvision.datasets

import inception_score

ROOT = "data"

def get_cifar10(download):
    download = int(download)
    train = torchvision.datasets.CIFAR10(root=ROOT, train=True, download=download)
    data = numpy.transpose(train.train_data, axes=[0, 3, 1, 2]).astype(numpy.float32)
    data = (data/255.0-0.5)*2 # scale between -1 and 1
    return data, train.train_labels

def compute_inception(download):
    data, labs = get_cifar10(download)
    output = inception_score.get_inception_score(data)
    print("Mean [std dev]: %s [%s]" % output)

def main():
    import sys
    
    DEFAULT = {
        "download": 0
    }
    
    args = [a.split("=") for a in sys.argv[1:]]
    
    DEFAULT.update(args)
    
    compute_inception(**DEFAULT)

if __name__ == "__main__":
    main()
