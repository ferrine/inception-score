import numpy, os

import torchvision.datasets

import inception_score

def get_cifar10():
    train = torchvision.datasets.CIFAR10(root=ROOT, train=True, download=True)
    return train.train_data, train.train_labels

def compute_inception():
    data, labs = get_cifar10()
    output = inception_score.get_inception_score(data)
    print("Mean [std dev]: %s [%s]" % output)

def main():
    compute_inception()

if __name__ == "__main__":
    main()
