# Exploiting loss smoothness to improve query efficiency of black-box adversarial attack

This repository is the official implementation of the BABIES algorithm (*Black-box Attack Based on IntErpolation Scheme*) developed in the paper **Exploiting the local parabolic landscapes of adversarial losses to accelerate black-box adversarial attack**, published in ECCV 2022. 



## Requirements

Our codes were tested on GPU with:

- Python 3.7.9, 
- PyTorch 1.7.1,
- torchvision 0.8.2,
- PIL 5.4.1. 



## Datasets

We perform the evaluation on three sets of MNIST, CIFAR-10 and ImageNet images. We include these sets in the repository.  

For **MNIST**, we randomly select 1,000 correctly labeled images from the MNIST testing sets. In targeted attack, the target labels are uniformly sampled at random, and the same target labels are used for all methods in evaluation. The attacked images, their correct and targeted labels will be loaded from file `mnist_testset.pth` in the folder `MNIST/data`. 

For **CIFAR-10**, we randomly select 1,000 correctly labeled images from the CIFAR testing sets. The attacked images, their correct and targeted labels will be loaded from file `cifar_testset.pth` in the folder `CIFAR10/data`. For convenience, we also include these images in `CIFAR10/data/imgs` folder, as well as their labels in `.txt` files. 

For **ImageNet**, the attacks on standard models are performed on a set of 200 correctly labeled images from the ImageNetV2. The attacked images will be loaded from folder `ImageNet/data/imgs`. Their correct and targeted labels are from files `class2image.txt` and `target_label.txt` respectively. Since the classification accuracy drops on robust models, we test the defended models on 100 images subsampled from this set, which are correctly labeled by them.  



## Models

We use our method and the other baselines to attack eight pre-trained classifiers (four standard and four l<sub>2</sub>-robust).

The pre-trained model for **MNIST** is an l<sub>2</sub>-robust CNN at radius 0.005, trained by TRADES ([https://github.com/yaodongyu/TRADES](https://github.com/yaodongyu/TRADES)). The model is provided at `MNIST/models/smallCNN_l2_eps0.005.pt`.

For **CIFAR-10**, we test two standard classifiers: inception_v3 and vgg13_bn. The pre-trained models were acquired from [https://github.com/huyvnphan/PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10). Due to their large size, we do not include these in the repository. For the codes to run properly, the users can download the model files `inception_v3.pt` and `vgg13_bn.pt` from the above source and put them into the folder `CIFAR10/models/state_dicts`. We also test one pre-trained  l<sub>2</sub>-robust Resnet50 model at radius 1, which was acquired from [https://github.com/MadryLab/robustness](https://github.com/MadryLab/robustness). To run this experiment, the users can download and extract the model weights from the file `cifar_l2_1_0.pt` in this source, save the weights as `resnet50_l2_eps1.pt` and put this file into the folder `CIFAR10/models/state_dicts`. 

For **ImageNet**, we test two standard classifiers: inception_v3 and resnet50. The pretrained classifiers are acquired from `torchvision.models` and will be downloaded automatically once the codes are run. We also test two defended models:  l<sub>2</sub>-robust ResNet18 and  l<sub>2</sub>-robust ResNet 50 model at radius 3, which was acquired from [https://github.com/microsoft/robust-models-transfer](https://github.com/microsoft/robust-models-transfer). To run this experiment, the users can download the files `resnet18_l2_eps3.ckpt` and `resnet50_l2_eps3.ckpt` and extract the model weights, save them as `resnet18_l2_eps3.pt` and `resnet50_l2_eps3.pt` and put theses files into the folder `ImageNet/models/`.   



## Evaluations

Example commands to evaluate our BABIES algorithm are:

```
python run_BABIES.py --model='vgg13_cifar' --total_size=1000 --batch_size=1000 --max_iters=3072 --rho=2.4 --eps=2.0

python run_BABIES.py --model='resnet50' --total_size=200 --batch_size=50 --max_iters=25000 --rho=12.0 --eps=3.0  --targeted
```

Users can customize the test by adjusting the flags: 

- `model`: names of target model
- `total_size`: the size of image set to attack 
- `batch_size`: batch size
- `max_iters`: maximum number of iterations
- `rho`: maximum allowable distortion in  l<sub>2</sub> norm
- `eps`: querying step  
- `targeted`: do targeted attack with this flag (otherwise untargeted) 

The target classifiers, experiment parameters and algorithm parameters to reproduce Tables 2-5 in our paper are shown in the below table. The first numbers are for untargeted attacks. Numbers in parentheses are for targeted attacks.

| `model`                      | `total_size` | `max_iters`  | `rho`   | `eps`     |
| ---------------------------- | ------------ | ------------ | ------- | --------- |
| `inception_v3`               | 200          | 5000 (25000) | 5 (12)  | 2 (3)     |
| `resnet50`                   | 200          | 5000 (25000) | 5 (12)  | 2 (3)     |
| `inception_v3_cifar`         | 1000         | 3072 (3072)  | 2.4 (4) | 2 (2)     |
| `vgg13_cifar`                | 1000         | 3072 (3072)  | 2.4 (4) | 2 (2)     |
| `resnet18_l2_eps3`           | 100          | 5000 (25000) | 12 (32) | 8 (8)     |
| `resnet50_l2_eps3`           | 100          | 5000 (25000) | 12 (32) | 8 (8)     |
| `resnet50_l2_eps1_cifar`     | 1000         | 3072 (3072)  | 2 (3)   | 0.5 (0.5) |
| `smallCNN_l2_eps0.005_mnist` | 1000         | 5000 (5000)  | 1 (2)   | 0.5 (0.5) |

Specifically, users can also use the following command lines: 

```
## MNIST (robust model)
#python run_BABIES.py --model='smallCNN_l2_eps0.005_mnist' --total_size=1000 --batch_size=1000 --max_iters=5000 --rho=1.0 --eps=0.5
#python run_BABIES.py --model='smallCNN_l2_eps0.005_mnist' --total_size=1000 --batch_size=1000 --max_iters=5000 --rho=2.0 --eps=0.5 --targeted

## CIFAR-10 (robust model)
#python run_BABIES.py --model='resnet50_l2_eps1_cifar' --total_size=1000 --batch_size=250 --max_iters=3072 --rho=2.0 --eps=0.5
#python run_BABIES.py --model='resnet50_l2_eps1_cifar' --total_size=1000 --batch_size=250 --max_iters=3072 --rho=3.0 --eps=0.5 --targeted

## CIFAR-10 (standard model)
#python run_BABIES.py --model='vgg13_cifar' --total_size=1000 --batch_size=1000 --max_iters=3072 --rho=2.4 --eps=2.0
#python run_BABIES.py --model='vgg13_cifar' --total_size=1000 --batch_size=1000 --max_iters=3072 --rho=4.0 --eps=2.0 --targeted
#python run_BABIES.py --model='inception_v3_cifar' --total_size=1000 --batch_size=125 --max_iters=3072 --rho=2.4 --eps=2.0
#python run_BABIES.py --model='inception_v3_cifar' --total_size=1000 --batch_size=125 --max_iters=3072 --rho=4.0 --eps=2.0 --targeted

## ImageNet (standard model)
#python run_BABIES.py --model='inception_v3' --total_size=200 --batch_size=50 --max_iters=5000 --rho=5.0 --eps=2.0
#python run_BABIES.py --model='inception_v3' --total_size=200 --batch_size=50 --max_iters=25000 --rho=12.0 --eps=3.0 --targeted
#python run_BABIES.py --model='resnet50' --total_size=200 --batch_size=50 --max_iters=5000 --rho=5.0 --eps=2.0
#python run_BABIES.py --model='resnet50' --total_size=200 --batch_size=50 --max_iters=25000 --rho=12.0 --eps=3.0  --targeted

## ImageNet (robust model)
#python run_BABIES.py --model='resnet18_l2_eps3' --total_size=100 --batch_size=50 --max_iters=5000 --rho=12.0 --eps=8.0
#python run_BABIES.py --model='resnet18_l2_eps3' --total_size=100 --batch_size=50 --max_iters=25000 --rho=32.0 --eps=8.0 --targeted
#python run_BABIES.py --model='resnet50_l2_eps3' --total_size=100 --batch_size=50 --max_iters=5000 --rho=12.0 --eps=8.0
#python run_BABIES.py --model='resnet50_l2_eps3' --total_size=100 --batch_size=50 --max_iters=25000 --rho=32.0 --eps=8.0 --targeted
```

