import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from CIFAR10.models.inception       import inception_v3
from CIFAR10.models.vgg             import vgg13_bn
from CIFAR10.models.robust_resnet   import resnet50 as robust_resnet50
from MNIST.models.small_cnn         import SmallCNN
from PIL import Image

#------------------------------------------------ 
# Load data and models
#------------------------------------------------ 
                
def load_model(model_name,dataset,sampled_image_dir,model_dir,device):

    # ---- CIFAR-10 tests ----
    if 'cifar' in model_name:
        
        # load data
        image_size = 32
        batchfile = '%s/%s/cifar_testset.pth' % (dataset,sampled_image_dir)
        checkpoint = torch.load(batchfile)
        images = checkpoint['images']
        labels = torch.from_numpy(checkpoint['labels'])
        labels_targeted = checkpoint['labels_targeted'].long()
          
        stride       = 4   
        freq_dims    = 20
        interval     = 10    
        
        # load model
        # standard models
        if model_name == 'inception_v3_cifar':
            model = inception_v3(pretrained=True).to(device)
        elif model_name == 'vgg13_cifar':
            model = vgg13_bn(pretrained=True).to(device)
        # l2 robust models
        elif model_name == 'resnet50_l2_eps1_cifar':
            model = robust_resnet50().to(device)
            model.load_state_dict(torch.load('%s/%s/%s.pt'%(dataset,model_dir,model_name)))
        model.eval()
        
        # ---- MNIST tests ----
    elif 'mnist' in model_name:
        
        # load data
        image_size = 28
        batchfile = '%s/%s/mnist_testset.pth' % (dataset,sampled_image_dir)
        checkpoint = torch.load(batchfile)
        images = checkpoint['images']
        labels = checkpoint['labels']
        labels_targeted = checkpoint['labels_targeted'].long()
        
        stride       = 4
        freq_dims    = 16
        interval     = 50    
        
        # load model
        model = SmallCNN().to(device)
        model.load_state_dict(torch.load('%s/%s/%s.pt'%(dataset,model_dir,model_name), map_location=torch.device('cpu')))
        model.eval()
     
    # ---- ImageNet tests ----
    else:
        
        # load data
        image_size = 299
        trans1 = transforms.ToTensor()
        images = torch.zeros(1000, 3, image_size, image_size)
        for i in range(1000):
            im = Image.open('%s/%s/imgs/image_%04d.jpeg'%(dataset,sampled_image_dir,i))
            images[i,:,:,:] = trans1(im).unsqueeze(0)
        labels = np.loadtxt('%s/%s/class2image.txt'%(dataset,sampled_image_dir))
        labels = torch.tensor(labels,dtype=torch.long)
        labels_targeted = np.loadtxt('%s/%s/target_label.txt'%(dataset,sampled_image_dir))
        labels_targeted = torch.tensor(labels_targeted,dtype=torch.long)
            
        stride       = 9
        freq_dims    = 38
        interval     = 50    
        
        # load model
        # standard models
        if (model_name == 'inception_v3') or (model_name == 'resnet50'):
            model = getattr(models, model_name)(pretrained=True).to(device)
        else: 
            if model_name == 'resnet18_l2_eps3':
                model = getattr(models, 'resnet18')().to(device)
            elif model_name == 'resnet50_l2_eps3':
                model = getattr(models, 'resnet50')().to(device)
            model.load_state_dict(torch.load('%s/%s/%s.pt'%(dataset,model_dir,model_name)))
        model.eval()
        
    return model, images, labels, labels_targeted, image_size, stride, freq_dims, interval
