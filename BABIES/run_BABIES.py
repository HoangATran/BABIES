import sys
sys.path.append("../")

import torch
from loader import load_model
import utils
import os
import numpy as np
import math      
import argparse

## MNIST (robust model)
#python run_BABIES.py --model='smallCNN_l2_eps0.005_mnist' --total_size=1000 --batch_size=1000 --max_iters=5000 --rho=1.0 --eps=0.5
#python run_BABIES.py --model='smallCNN_l2_eps0.005_mnist' --total_size=1000 --batch_size=1000 --max_iters=5000 --rho=2.0 --eps=0.5 --targeted
#
## CIFAR-10 (robust model)
#python run_BABIES.py --model='resnet50_l2_eps1_cifar' --total_size=1000 --batch_size=250 --max_iters=3072 --rho=2.0 --eps=0.5
#python run_BABIES.py --model='resnet50_l2_eps1_cifar' --total_size=1000 --batch_size=250 --max_iters=3072 --rho=3.0 --eps=0.5 --targeted
#
## CIFAR-10 (standard model)
#python run_BABIES.py --model='vgg13_cifar' --total_size=1000 --batch_size=1000 --max_iters=3072 --rho=2.4 --eps=2.0
#python run_BABIES.py --model='vgg13_cifar' --total_size=1000 --batch_size=1000 --max_iters=3072 --rho=4.0 --eps=2.0 --targeted
#python run_BABIES.py --model='inception_v3_cifar' --total_size=1000 --batch_size=125 --max_iters=3072 --rho=2.4 --eps=2.0
#python run_BABIES.py --model='inception_v3_cifar' --total_size=1000 --batch_size=125 --max_iters=3072 --rho=4.0 --eps=2.0 --targeted
#
## ImageNet (standard model)
#python run_BABIES.py --model='inception_v3' --total_size=1000 --batch_size=125 --max_iters=5000 --rho=5.0 --eps=2.0
#python run_BABIES.py --model='inception_v3' --total_size=1000 --batch_size=125 --max_iters=25000 --rho=12.0 --eps=3.0 --targeted
#python run_BABIES.py --model='resnet50' --total_size=1000 --batch_size=250 --max_iters=5000 --rho=5.0 --eps=2.0
#python run_BABIES.py --model='resnet50' --total_size=1000 --batch_size=250 --max_iters=25000 --rho=12.0 --eps=3.0  --targeted
#
## ImageNet (robust model)
#python run_BABIES.py --model='resnet18_l2_eps3' --total_size=1000 --batch_size=250 --max_iters=5000 --rho=12.0 --eps=8.0
#python run_BABIES.py --model='resnet18_l2_eps3' --total_size=1000 --batch_size=250 --max_iters=25000 --rho=32.0 --eps=8.0 --targeted
#python run_BABIES.py --model='resnet50_l2_eps3' --total_size=1000 --batch_size=250 --max_iters=5000 --rho=12.0 --eps=8.0
#python run_BABIES.py --model='resnet50_l2_eps3' --total_size=1000 --batch_size=250 --max_iters=25000 --rho=32.0 --eps=8.0 --targeted

#------------------------------------------------     
# Supporting functions
#------------------------------------------------ 
def expand_vector(x, size):
    batch_size   = x.size(0)
    num_channels = 1 if dataset == '../MNIST' else 3
    x = x.view(-1, num_channels, size, size)
    z = torch.zeros(batch_size, num_channels, image_size, image_size)
    z[:, :, :size, :size] = x
    return z

def normalize(x):
    return utils.apply_normalization(x, dataset)

def get_probs(model, x, y):
    output = model(normalize(torch.autograd.Variable(x.to(device)))).cpu()
    probs = torch.index_select(torch.nn.Softmax(dim=1)(output).data, 1, y)
    return torch.diag(probs)

def get_preds(model, x):
    output = model(normalize(torch.autograd.Variable(x.to(device)))).cpu()
    _, preds = output.data.max(1)
    return preds
        
def l2_projection(x,dir,radius): # project a new perturbation at x to l2 ball if ||x|| = radius. Set geodesic length = magnitude of dir
    # x: current pertubation (4D tensor).     Size: batch_size x num_channels x image_size x image_size
    # dir: pertubation direction (4D tensor). Size: either bs or 1 x num_channels x image_size x image_size
    # radius: radius of l2 ball (positive number)
    bsx = x.size(0)
    bsd = dir.size(0)
    normx   = (x**2).sum((1,2,3)).sqrt()+1.e-10     # norm of x
    normd   = (dir**2).sum((1,2,3)).sqrt()+1.e-10   # norm of dir
    # divide the image batch into two sets, based on whether the perturbed images are inside l2 ball or on the boundary
    inside_mask  = (radius-normx >= normd).float()
    onBound_mask = (radius-normx <  normd).float()
    # form tangent direction
    x_norm   = x/normx[:,None,None,None]
    dir_norm = (dir/normd[:,None,None,None]).repeat(bsx//bsd,1,1,1)
    tang_dir = (dir_norm - x_norm * torch.einsum('ij, ij->i', torch.reshape(dir_norm,(bsx,-1)), torch.reshape(x_norm,(bsx,-1)))[:,None,None,None])
    normtg   = (tang_dir**2).sum((1,2,3)).sqrt()+1.e-10
    # auxiliary perturbation on tangent direction
    perturb0 = tang_dir * (normx * torch.tan(normd/normx)/normtg)[:,None,None,None]
    normp0   = ((x+perturb0)**2).sum((1,2,3)).sqrt()+1.e-10
    # project to l2 ball for perturbed images on boundary
    x_new1 = ((x+perturb0) * (radius/normp0)[:,None,None,None]) * onBound_mask[:,None,None,None] 
    # just add perturbation directly otherwise
    x_new2 = (x + dir) * inside_mask[:,None,None,None]
    return (x_new1 + x_new2) - x


#------------------------------------------------     
# BABIES algorithm
#------------------------------------------------ 
def run_BABIES(model, images_batch, labels_batch, max_iters, rho, eps, freq_dims, stride, order, targeted, attack, log_every):
    torch.manual_seed(seed)
    
    batch_size   = images_batch.size(0)
    num_channels = images_batch.size(1)
    image_size   = images_batch.size(2)
    if order == 'rand':         # random attack
        indices = torch.randperm(num_channels * image_size * image_size)
    elif order == 'strided':    # low frequency attack
        indices = utils.block_order(image_size, num_channels, initial_size=freq_dims, stride=stride)
    n_dims = num_channels * image_size * image_size
    # logging tensors
    probs = torch.zeros(batch_size,max_iters)
    succs = torch.zeros(batch_size,max_iters)
    queries = torch.zeros(batch_size,max_iters)
    l2_norms = torch.zeros(batch_size,max_iters)
    linf_norms = torch.zeros(batch_size,max_iters)
    iters = torch.zeros(batch_size,max_iters)

    if attack == 'dct':
        trans = lambda z: utils.block_idct(z, block_size=image_size, linf_bound=0.0)
    else:
        trans = lambda z: z
        
    images_k = images_batch
    probs_k  = get_probs(model, images_batch, labels_batch)
    preds = get_preds(model, images_batch)
      
    if targeted:
        remaining = preds.ne(labels_batch)
    else:
        remaining = preds.eq(labels_batch)        
    remaining_indices = torch.arange(0, batch_size)[remaining].long()
    
    epsilon = eps * torch.ones(batch_size)
    
    for k in range(max_iters):
        
        dim = indices[k%n_dims]

        # check if the image are successfully attacked and stop early
        if remaining.sum() == 0:
            probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
            succs[:, k:] = torch.ones(batch_size, max_iters - k)
            queries[:, k:] = torch.zeros(batch_size, max_iters - k)
            l2_norms[:, k:] = l2_norms[:, k-1].unsqueeze(1).repeat(1, max_iters - k)
            linf_norms[:, k:] = linf_norms[:, k-1].unsqueeze(1).repeat(1, max_iters - k)
            iters[:, k:] = torch.zeros(batch_size, max_iters - k)
            break
        
        distortion  = (images_k - images_batch)     # current distortion
        tot_perturb = torch.zeros_like(images_k)    # final perturbation for this iteration: initialize
        cur_perturb = torch.zeros_like(images_k)    # trial perturbation for this iteration: initialize
        queries_k   =  torch.zeros(batch_size)

        remaining_indices = torch.arange(0, batch_size)[remaining].long()
            
        probs_left = torch.zeros(batch_size)
        probs_right = torch.zeros(batch_size)
        probs_interp = torch.zeros(batch_size)
            
        # initialize arrays saving indices of images that takes left, right and interpolation vector
        left_indices = torch.tensor([], dtype=torch.int64)
        right_indices = torch.tensor([], dtype=torch.int64)
        interp_indices = torch.tensor([], dtype=torch.int64)
                        
        # pertubation
        diffL = torch.zeros(batch_size,n_dims)
        diffR = torch.zeros(batch_size,n_dims)
        diffL[:,dim] = -epsilon   # left perturbation
        diffR[:,dim] = epsilon    # right perturbation
            
        # trying left direction
        cur_perturb[remaining_indices] = l2_projection(distortion[remaining_indices], trans(expand_vector(diffL[remaining_indices], image_size)), rho)
        images_left = (images_k[remaining_indices] + cur_perturb[remaining_indices]).clamp(0, 1)
        probs_left[remaining_indices]  = get_probs(model, images_left, labels_batch[remaining_indices])
        queries_k[remaining_indices]  += 1
                        
        if targeted:
            left_improved = probs_left[remaining_indices].gt(probs_k[remaining_indices])
        else:
            left_improved = probs_left[remaining_indices].lt(probs_k[remaining_indices])

        left_indices = remaining_indices[left_improved]
        tot_perturb[left_indices] = cur_perturb[left_indices]

        # trying right direction
        if (~left_improved).sum() > 0:
            non_left_indices = remaining_indices[~left_improved]
            cur_perturb[non_left_indices] = l2_projection(distortion[non_left_indices], trans(expand_vector(diffR[non_left_indices], image_size)), rho)
            images_right = (images_k[non_left_indices] + cur_perturb[non_left_indices]).clamp(0, 1)
            probs_right[non_left_indices]  = get_probs(model, images_right, labels_batch[non_left_indices])
            queries_k[non_left_indices]  += 1

            if targeted:
                right_improved = probs_right[non_left_indices].gt(probs_k[non_left_indices])
            else:
                right_improved = probs_right[non_left_indices].lt(probs_k[non_left_indices])

            right_indices = non_left_indices[right_improved]
            tot_perturb[right_indices] = cur_perturb[right_indices]

            # trying interpolation
            if args.interp and (~right_improved).sum() > 0:
                left_right_equal = probs_left[non_left_indices].eq(probs_right[non_left_indices])
                interp_indices = non_left_indices[~right_improved & ~left_right_equal]
                if interp_indices.nelement():
                    diffn = torch.zeros((~right_improved & ~left_right_equal).sum(),n_dims)
                    diffn[:,dim] = (probs_left[interp_indices] - probs_right[interp_indices])*epsilon[interp_indices] \
                                        / ((probs_left[interp_indices] + probs_right[interp_indices] - 2*probs_k[interp_indices])+1e-10) /2
                    probs_interp[interp_indices] = (probs_left[interp_indices] + probs_right[interp_indices] - 2*probs_k[interp_indices]) / 2 / epsilon[interp_indices]**2 * diffn[:,dim]**2 \
                                        + (probs_right[interp_indices] - probs_left[interp_indices]) / 2 / epsilon[interp_indices] * diffn[:,dim]\
                                        + probs_k[interp_indices]

                    cur_perturb[interp_indices] = l2_projection(distortion[interp_indices], trans(expand_vector(diffn,image_size)), rho)
                    tot_perturb[interp_indices] = cur_perturb[interp_indices]

        images_k = (images_k + tot_perturb).clamp(0,1)
        

        if args.interp and (k + 1) % interval != 0:
            probs_k[left_indices] = probs_left[left_indices]        # true probs_left
            probs_k[right_indices] = probs_right[right_indices]     # true probs_right
            probs_k[interp_indices] = probs_interp[interp_indices]  # interpolating middle
        else:    # to avoid accumulation of error, query instead of interpolation after some iterations (10 for CIFAR and MNIST, 50 for ImageNet)
            probs_k = get_probs(model, images_k, labels_batch) 
            queries_k[remaining_indices] += 1 
           
        # record 
        probs[:,k] = probs_k
        succs[:,k] = ~remaining
        queries[:,k] = queries_k
        l2_norms[:,k] = (images_k - images_batch).view(batch_size, -1).norm(2, 1)
        linf_norms[:,k] = (images_k - images_batch).view(batch_size, -1).abs().max(1)[0]
        iters[remaining_indices,k] = 1

        preds = get_preds(model, images_k)
        remaining = preds.ne(labels_batch) if targeted else preds.eq(labels_batch)

        if args.interp and (k + 1) % interval != 0:
            # label predictions are not available at interpolated pertubation
            # add interpolated indices to the set of remaining indices
            remaining_indices = np.union1d(torch.arange(0, batch_size)[remaining].long(), interp_indices)
        else:
            remaining_indices = torch.arange(0, batch_size)[remaining].long()

        if (k + 1) % log_every == 0 or k == max_iters - 1:
            print('Iter %d: Avg.QY = %.2f, Med.QY = %.2f, SR = %.2f%%, Avg.l2 = %.2f' % (
                    k + 1, queries[~remaining,:k].sum(1).mean(), 0 if (~remaining).sum() == 0 else queries[~remaining,:k].sum(1).median(), (1-remaining.float().mean())*100, \
                    l2_norms[~remaining, k].mean() ))
    return probs, succs, queries, l2_norms, linf_norms, iters
    
#------------------------------------------------     
# Main program
#------------------------------------------------     

parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--model', type=str, help='Model name.')                                        # choose target model 
parser.add_argument('--attack', type=str, default='dct', help='Attack')                             # DCT or pixel attack. Default: DCT
parser.add_argument('--order', type=str, default='strided', help='Order: strided or random.')       # strided or random. Default: strided
parser.add_argument('--targeted', action='store_true', help='Targeted or untargeted attack.')       # TARGETED or UNTARGETED
parser.add_argument('--interp', action='store_false', help='Applying interpolation or not.')        # apply interpolation or not. Default: True
parser.add_argument('--total_size', type=int, help='Number of images.')                             # 1000 for CIFAR10 and MNIST, 200 for ImageNet (standard), 100 for ImageNet (robust)
parser.add_argument('--batch_size', type=int, help='Batch size.')                                   # 5000 for ImageNet and CIFAR, 
parser.add_argument('--max_iters', type=int, help='Maximum number of iterations.')                  # maximum number of iterations.                                                          # 
parser.add_argument('--rho', type=float, help='Radius of the L2 ball.')  
parser.add_argument('--eps', type=float, help='Querying step.')                                            
parser.add_argument('--log_every', type=int, default=1, help='log every n iterations.')

args = parser.parse_args()

seed         = 16     # random seed
device = 'cuda' if torch.cuda.is_available() else 'cpu'

result_dir        = 'results'
sampled_image_dir = 'data'
model_dir         = 'models/state_dicts' if 'cifar' in args.model else 'models'
dataset           = '../CIFAR10' if 'cifar'  in args.model else '../MNIST' if 'mnist' in args.model else  '../ImageNet'

model, images, labels, labels_targeted, image_size, stride, freq_dims, interval = load_model(args.model,dataset,sampled_image_dir,model_dir,device)

print('device: ',device)
print('model: %s' %args.model)
print('attack: %s, order: %s' %(args.attack,args.order))
print('interpolation: %s' %('True' if args.interp else 'False'))

# number of runs
N = int(math.floor(float(args.total_size) / float(args.batch_size)))
  
corr = np.zeros(N)
for i in range(N):
    corr[i] = (get_preds(model, images[i * args.batch_size:(i+1) * args.batch_size])==labels[i * args.batch_size:(i+1) * args.batch_size]).sum()/args.batch_size
print('Accuracy: %.2f'%np.mean(corr))

for i in range(N):
    print('--------Run batch %d out of %d batches-----------'%(i+1,N))  
    upper = min((i + 1) * args.batch_size, args.total_size)
    images_batch = images[(i * args.batch_size):upper]
    if args.targeted:
        labels_batch = labels_targeted[(i * args.batch_size):upper]
    else: 
        labels_batch = labels[(i * args.batch_size):upper]
        
    probs, succs, queries, l2_norms, linf_norms, iters = run_BABIES(
        model, images_batch, labels_batch, args.max_iters, args.rho, args.eps, freq_dims, stride, args.order,
        args.targeted, args.attack, args.log_every)
    
    if i == 0:
        all_probs = probs
        all_succs = succs
        all_queries = queries
        all_l2_norms = l2_norms
        all_linf_norms = linf_norms
        all_iters = iters

    else:
        all_probs = torch.cat([all_probs, probs], dim=0)
        all_succs = torch.cat([all_succs, succs], dim=0)
        all_queries = torch.cat([all_queries, queries], dim=0)
        all_l2_norms = torch.cat([all_l2_norms, l2_norms], dim=0)
        all_linf_norms = torch.cat([all_linf_norms, linf_norms], dim=0)
        all_iters = torch.cat([all_iters, iters], dim=0)

        
# create folder for saving results
if not os.path.isdir('%s' % (result_dir)):
    os.mkdir('%s' % (result_dir))

prefix = 'BABIES-' + args.attack 
if args.targeted:
    prefix += '_targeted'
if args.interp:
    prefix += '_interp'
savefile = '%s/%s_%s_step%.2f_radius%.2f.pth' % (
    result_dir, prefix, args.model, args.eps, args.rho)
torch.save({'probs': all_probs, 'succs': all_succs, 'queries': all_queries,
            'l2_norms': all_l2_norms, 'linf_norms': all_linf_norms, 'iters': all_iters}, savefile)

# print final result to screen
succs_indices = torch.arange(0, len(all_succs[:,-1]))[all_succs[:,-1] > 0].long()
print('-- FINAL RESULT: Avg.QY = %.2f, Med.QY = %.2f, SR = %.2f%%, Avg.l2 = %.2f' % (
all_queries[succs_indices,:].sum(1).mean(), all_queries[succs_indices,:].sum(1).median(), all_succs[:,-1].mean()*100, all_l2_norms[succs_indices,-1].mean() ))





