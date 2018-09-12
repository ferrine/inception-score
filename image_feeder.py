'''

Accepted commands:

    [1] To download and run it on CIFAR10 data
    >> python3 image_feeder.py download=1
    
    [2] To run it on CIFAR10 data
    >> python3 image_feeder.py
    
    [3] To run it on a folder of images
    >> python3 image_feeder.py folder=PATH_TO_FOLDER_OF_IMAGES

'''

import numpy, os, tqdm

import scipy.misc
import torchvision.datasets

import inception_score

ROOT = "data"

def get_cifar10(folder=None, download=False):
    if folder is None:
        download = int(download)
        train = torchvision.datasets.CIFAR10(root=ROOT, train=True, download=download)
        data = train.train_data
        
    else:
        images = []
        for f in tqdm.tqdm(os.listdir(folder), desc="Loading images", ncols=80):
            fname = os.path.join(folder, f)
            im = scipy.misc.imread(fname)
            images.append(im)
        data = numpy.array(images)
        
    data = numpy.transpose(data, axes=[0, 3, 1, 2])
    data = data.astype(numpy.float32)
    data = (data/255.0-0.5)*2
    return data

def compute_inception(folder, download):
    data = get_cifar10(folder, download)
    output = inception_score.get_inception_score(data)
    print("Mean [std dev]: %s [%s]" % output)

def main():
    import sys
    
    DEFAULT = {
        "download": 0,
        "folder": None
    }
    
    args = [a.split("=") for a in sys.argv[1:]]
    
    DEFAULT.update(args)
    
    compute_inception(**DEFAULT)

if __name__ == "__main__":
    main()
