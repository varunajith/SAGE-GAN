
# Import the necessary libraries and modules
import os
import argparse
from unet_models import AttentionUnetSegmentation,StyleUnetGenerator,NLayerDiscriminator#,DeepSea#,UnetSegmentation,DeepSea
#from cellpose_model import Cellpose_CPnet
from utils import set_requires_grad,mixed_list,noise_list,image_noise,initialize_weights
from data import BasicDataset,get_image_mask_pairs
import itertools
import torch.nn as nn
from tqdm import tqdm
from evaluate import evaluate_segmentation
from loss import CombinedLoss,VGGLoss
import torch.utils.data as data
import torch.nn.functional as F
import transforms as transforms
import torch
import numpy as np
import random
import logging
from diffaug import DiffAugment
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torchvision

import sys

# Clear GPU cache
torch.cuda.empty_cache()

# Simulate command-line arguments
sys.argv = ['train.py', '--seg_model','UNET', '--train_set_dir',
            '',
            '--lr','0.0001',
            '--p_vanilla_pretrain','0.5', '--p_vanilla_maintrain', '0.2','--p_diff','0.2', '--patience_pretrain','200', '--patience_maintrain', '200',
            '--output_dir','']

#previous_bestseg_checkpoint = ''

# Set a constant seed for reproducibility
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def reset_logging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

def train(args, image_size=[512,768], image_means=[0.5], image_stds=[0.5], save_checkpoint=True):
    reset_logging()
    logging.basicConfig(filename=os.path.join(args.output_dir, 'train.log'), filemode='w',
                       format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('>>>> image size=(%d,%d) , learning rate=%f , batch size=%d' % (
        image_size[0], image_size[1], args.lr, args.batch_size))
    
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'runs'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on:", device)

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([transforms.RandomOrder([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.33, contrast=0.33, saturation=0.33, hue=0)],p=0.5),
            transforms.RandomApply([transforms.GaussianBlur((5, 5), sigma=(0.1, 1.0))],p=0.5),
            transforms.RandomApply([transforms.RandomHorizontalFlip(0.5)],p=0.5),
            transforms.RandomApply([transforms.RandomVerticalFlip(0.5)],p=0.5),
            transforms.RandomApply([transforms.AddGaussianNoise(0., 0.01)], p=0.5),
            transforms.RandomApply([transforms.CLAHE()], p=0.5),
            transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.5),
            transforms.RandomApply([transforms.RandomCrop()], p=0.5),
        ])],p=args.p_vanilla_pretrain),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means, std=image_stds)
    ])

    dev_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means, std=image_stds)
    ])
    
    sample_pairs = get_image_mask_pairs(args.train_set_dir)
    #print(len(sample_pairs))# debugging
    random.shuffle(sample_pairs)
    train_pretrain_sample_pairs = sample_pairs[:int(args.pretrain_ratio*len(sample_pairs))]
    valid_sample_pairs = sample_pairs[int(args.pretrain_ratio*len(sample_pairs)):]

    train_maintrain_sample_pairs = train_pretrain_sample_pairs

    train_pretrain_data = BasicDataset(train_pretrain_sample_pairs, transforms=train_transforms,
                             vanilla_aug=False if args.p_vanilla_pretrain == 0 else True, gen_nc=args.gen_nc)
    
    #print(len(train_data)) #debugging
    valid_data = BasicDataset(valid_sample_pairs, transforms=dev_transforms, gen_nc=args.gen_nc)
    #print(len(valid_data)) #debugging
    
    # Set num_workers to # of CPU cores
    num_workers = min(os.cpu_count(), 32) 
    print("Num_worker:",num_workers)
    #def seed_worker(worker_id):
        #worker_seed = torch.initial_seed() % 2**32
        #np.random.seed(worker_seed)
        #random.seed(worker_seed)

    train_pretrain_iterator = data.DataLoader(train_pretrain_data, shuffle=True, batch_size=args.batch_size, num_workers=1,worker_init_fn=seed_worker, pin_memory=True,persistent_workers=True)
    valid_iterator = data.DataLoader(valid_data, shuffle=False, batch_size=args.batch_size, num_workers=0, pin_memory=True,worker_init_fn=seed_worker,persistent_workers=False)

    Gen =  StyleUnetGenerator(style_latent_dim=128, output_nc=args.gen_nc)
    if args.seg_model == 'UNET':
        Seg = AttentionUnetSegmentation(n_channels=args.gen_nc, n_classes=2)
    elif args.seg_model == 'CellPose':
        #Seg = Cellpose_CPnet(n_channels=args.gen_nc, n_classes=2)
        2==2
    elif args.seg_model == 'DeepSea':
        #Seg = DeepSea(n_channels=args.gen_nc, n_classes=2)
        2==2

    initialize_weights(Seg)
    
    grad_scaler = torch.amp.GradScaler('cuda',enabled=True)
    Seg_criterion = CombinedLoss()
    Seg = Seg.to(device)
    Seg_criterion = Seg_criterion.to(device)

    # Segmentation Pre-Training Phase
    pretrain_lr = 0.0001
    nstop_pretrain=0
    print('>>>> Starting  Pre-Training')
    logging.info(">>>> Starting Segmentation Pre-Training-' lr={}".format(pretrain_lr))
    pretrain_optimizer = torch.optim.Adam(Seg.parameters(), lr=pretrain_lr,betas=(0.9, 0.999))
    scheduler_Seg = ReduceLROnPlateau(pretrain_optimizer, 'min', patience=80, factor=0.9)
    
    
    #Seg.load_state_dict(torch.load(previous_bestseg_checkpoint,weights_only=True))
    
    best_pretrain_fscore = 0
    for pretrain_epoch in range(args.max_pretrain_epoch):
        Seg.train()
        epoch_loss = 0
        for step, batch in enumerate(tqdm(train_pretrain_iterator)):
            real_img = batch['image']
            real_mask = batch['mask']
            real_img = real_img.to(device=device, dtype=torch.float32,non_blocking=True)
            real_mask = real_mask.to(device=device, dtype=torch.float32,non_blocking=True)
            
            pretrain_optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=True):
                pred_mask,psi1 = Seg(real_img)
                psi1 = psi1.detach()  # Detach from the computation graph if you're not using it in the loss
                loss = Seg_criterion(pred_mask, torch.squeeze(real_mask.to(torch.long), dim=1))

            grad_scaler.scale(loss).backward()
            grad_scaler.step(pretrain_optimizer)
            grad_scaler.update()
            epoch_loss += loss.item()
        print("")
        print(
    f"[Epoch {pretrain_epoch}/{args.max_pretrain_epoch}], [Seg loss: {epoch_loss/len(train_pretrain_iterator):.3f}]"
)
          

        val_scores = evaluate_segmentation(Seg, valid_iterator, device,Seg_criterion,len(valid_data),is_avg_prec=True,prec_thresholds=[0.5],output_dir='train_val_pretrain')
        #val_scores = evaluate_segmentation(Seg, train_pretrain_iterator, device,Seg_criterion,len(train_pretrain_data),is_avg_prec=True,prec_thresholds=[0.5],output_dir=None)
            
            
        logging.info('>>>> Epoch:%d  , Dice score=%f , avg fscore=%f'  % (pretrain_epoch,val_scores['dice_score'], val_scores['avg_fscore']))
        if val_scores['avg_fscore'] is not None and val_scores['avg_fscore'] > best_pretrain_fscore:
            best_pretrain_fscore = val_scores['avg_fscore']
            logging.info('>>>> Save the petrained model checkpoints to %s'%(os.path.join(args.output_dir)))
            torch.save(Seg.state_dict(), os.path.join(args.output_dir, 'Seg_pretrained.pth'))
            nstop_pretrain=0
        elif val_scores['avg_fscore'] is not None and val_scores['avg_fscore']<=best_pretrain_fscore:
                  nstop_pretrain +=1
        if nstop_pretrain == args.patience_pretrain:
               print('INFO: Early Stopping met ...')
               print('INFO: Finish training process')
               break
        scheduler_Seg.step(val_scores['avg_fscore'] if val_scores['avg_fscore'] is not None else 0)
    print("Learning Rate of Seg:", scheduler_Seg.get_last_lr())
    Seg.load_state_dict(torch.load(os.path.join(args.output_dir, 'Seg_pretrained.pth'),weights_only=True))
    
    D1 = NLayerDiscriminator(input_nc=args.gen_nc)
    D2 = NLayerDiscriminator(input_nc=1)

    initialize_weights(Gen)
    initialize_weights(D1)
    initialize_weights(D2)

    optimizer_G = torch.optim.Adam(itertools.chain(Gen.parameters(), Seg.parameters()), lr=args.lr, betas=(0.9, 0.999))
    optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler_G = ReduceLROnPlateau(optimizer_G, 'min', patience=80, factor=0.85)
    scheduler_D1 = ReduceLROnPlateau(optimizer_D1, 'min', patience=80, factor=0.85)
    scheduler_D2 = ReduceLROnPlateau(optimizer_D2, 'min', patience=80, factor=0.85)
    d_criterion = nn.MSELoss()
    Gen_criterion_1 = nn.L1Loss() 
    Gen_criterion_2 = VGGLoss()
    Gen = Gen.to(device) 
    D1 = D1.to(device)
    D2 = D2.to(device)
    
    d_criterion = d_criterion.to(device)
    Gen_criterion_1 = Gen_criterion_1.to(device)
    Gen_criterion_2 = Gen_criterion_2.to(device)


    # Main Training with Gradual Mixing
    mix_ratio = 0.0
    mix_step = 0.005
    freeze_gen_epochs = 30
    real_masks = None
    pred_masks = None
    avg_fscore_best = 0
    nstop_maintrain = 0
    
    logging.info('>>>> Start Main Training')
    print('>>>> Starting Main Training')
    '''
    train_transforms_2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([transforms.RandomOrder([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.33, contrast=0.33, saturation=0.33, hue=0)],p=0.5),
            transforms.RandomApply([transforms.GaussianBlur((5, 5), sigma=(0.1, 1.0))],p=0.0),
            transforms.RandomApply([transforms.RandomHorizontalFlip(0.5)],p=0.5),
            transforms.RandomApply([transforms.RandomVerticalFlip(0.5)],p=0.5),
            transforms.RandomApply([transforms.AddGaussianNoise(0., 0.01)], p=0.5),
            transforms.RandomApply([transforms.CLAHE()], p=0.5),
            transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.5),
            transforms.RandomApply([transforms.RandomCrop()], p=0.5),
        ])],p=args.p_vanilla_maintrain),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means, std=image_stds)
    ])

    dev_transforms_2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means, std=image_stds)
    ])
    
    sample_pairs = get_image_mask_pairs(args.train_set_dir)
    #print(len(sample_pairs))# debugging
    random.shuffle(sample_pairs)
    train_sample_pairs = sample_pairs[:int(args.maintrain_ratio*len(sample_pairs))]
    valid_sample_pairs = sample_pairs[int(args.maintrain_ratio*len(sample_pairs)):]

    train_data = BasicDataset(train_sample_pairs, transforms=train_transforms_2,
                             vanilla_aug=False if args.p_vanilla_maintrain == 0 else True, gen_nc=args.gen_nc)
    #print(len(train_data)) #debugging
    valid_data = BasicDataset(valid_sample_pairs, transforms=dev_transforms, gen_nc=args.gen_nc)
    #print(len(valid_data)) #debugging
    train_iterator = data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size, num_workers=1,worker_init_fn=seed_worker, pin_memory=True,persistent_workers=True)
    valid_iterator = data.DataLoader(valid_data, shuffle=False, batch_size=args.batch_size, num_workers=0, pin_memory=True,worker_init_fn=seed_worker,persistent_workers=False)
    '''
    
    train_maintrain_data = BasicDataset(train_maintrain_sample_pairs, transforms=train_transforms,
                             vanilla_aug=False if args.p_vanilla_maintrain == 0 else True, gen_nc=args.gen_nc)
    train_maintrain_iterator = data.DataLoader(train_maintrain_data, shuffle=True, batch_size=args.batch_size, num_workers=1,worker_init_fn=seed_worker, pin_memory=False)#,persistent_workers=True)
    
    for epoch in range(args.max_epoch):
        mix_ratio = min(mix_ratio + mix_step, 0.6)
        print("")
        print(f"mix_ratio = {mix_ratio}")  
        
        Gen.train()
        Seg.train()
        D1.train()
        D2.train()

        for step, batch in enumerate(tqdm(train_maintrain_iterator)):
            real_img = batch['image']
            real_mask = batch['mask']

            valid = torch.full((real_mask.shape[0], 1, 62, 94), 1.0, dtype=real_mask.dtype, device=device)
            fake = torch.full((real_mask.shape[0], 1, 62, 94), 0.0, dtype=real_mask.dtype, device=device)

            real_img = real_img.to(device=device, dtype=torch.float32)#,non_blocking=True)
            real_mask = real_mask.to(device=device, dtype=torch.float32)#,non_blocking=True)

            set_requires_grad(D1, True)
            set_requires_grad(D2, True)
            set_requires_grad(Gen, epoch >= freeze_gen_epochs)
            set_requires_grad(Seg, True)

            optimizer_G.zero_grad(set_to_none=True)
            optimizer_D1.zero_grad(set_to_none=True)
            optimizer_D2.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda',enabled=True):
                # Generate synthetic data
                if random.random() < 0.9:
                    style = mixed_list(real_img.shape[0], 5, Gen.latent_dim, device=device)
                else:
                    style = noise_list(real_img.shape[0], 5, Gen.latent_dim, device=device)
                
                im_noise = image_noise(real_mask.shape[0], image_size, device=device)
                fake_img = Gen(real_mask, style, im_noise)

                # Mixed training logic
                use_generated = random.random() < mix_ratio
                #seg_weight = 1.0 - mix_ratio

                if use_generated:
                 seg_input = fake_img.detach() if epoch < freeze_gen_epochs else fake_img
                 #seg_weight = mix_ratio
                else:
                  seg_input = real_img

             

 # Calculate rec_mask_loss and id_mask_loss based on the condition
                if use_generated:
                  rec_mask, psi1 = Seg(seg_input)
                  psi1 = psi1.detach()  # Detach from the computation graph if you're not using it in the loss
    # If it's a generated (fake) image, calculate both rec_mask_loss and id_mask_loss
                  rec_mask_loss = 200 * Seg_criterion(rec_mask, torch.squeeze(real_mask.to(dtype=torch.long), dim=1)) 

    # GAN components
                  fake_mask, psi2 = Seg(real_img)
                  psi2 = psi2.detach()  # Detach from the computation graph if you're not using it in the loss
                  fake_mask_p = F.softmax(fake_mask, dim=1).float()
                  fake_mask_p = torch.unsqueeze(fake_mask_p.argmax(dim=1), dim=1)
                  fake_mask_p = fake_mask_p.to(dtype=torch.float32)

                  if random.random() < 0.9:
                   style = mixed_list(real_mask.shape[0], 4, Gen.latent_dim, device=device)
                  else:
                   style = noise_list(real_mask.shape[0], 4, Gen.latent_dim, device=device)

                  im_noise = image_noise(real_mask.shape[0], image_size, device=device)
                  rec_img = Gen(fake_mask_p, style, im_noise)

                  set_requires_grad(D1, False)
                  set_requires_grad(D2, False)

                  d_img_loss = d_criterion(D1(DiffAugment(fake_img, p=args.p_diff)), valid)
                  d_mask_loss = d_criterion(D2(fake_mask_p), valid)
                  id_mask_loss = 100 * Seg_criterion(fake_mask, torch.squeeze(real_mask.to(dtype=torch.long), dim=1)) 
                  rec_img_loss = (50 * Gen_criterion_1(rec_img, real_img) + 100 * Gen_criterion_2(rec_img, real_img)) 
                  id_img_loss = (25 * Gen_criterion_1(fake_img, real_img) + 50 * Gen_criterion_2(fake_img, real_img)) 
                  g_loss = d_mask_loss + d_img_loss + rec_mask_loss + rec_img_loss + id_mask_loss + id_img_loss
                else:
            # If it's a real image, only calculate id_mask_loss
                 fake_mask, psi2 = Seg(seg_input)  # Segment the real image
                 fake_mask_p = F.softmax(fake_mask, dim=1).float()
                 fake_mask_p = torch.unsqueeze(fake_mask_p.argmax(dim=1), dim=1)
                 fake_mask_p = fake_mask_p.to(dtype=torch.float32)
                 if random.random() < 0.9:
                   style = mixed_list(real_mask.shape[0], 4, Gen.latent_dim, device=device)
                 else:
                   style = noise_list(real_mask.shape[0], 4, Gen.latent_dim, device=device)

                 im_noise = image_noise(real_mask.shape[0], image_size, device=device)
                 rec_img = Gen(fake_mask_p, style, im_noise)

                 set_requires_grad(D1, False)
                 set_requires_grad(D2, False)

                 id_mask_loss = 100 * Seg_criterion(fake_mask, torch.squeeze(real_mask.to(dtype=torch.long), dim=1)) 
                 d_img_loss = d_criterion(D1(DiffAugment(fake_img, p=args.p_diff)), valid)
                 d_mask_loss = d_criterion(D2(fake_mask_p), valid)
                 rec_img_loss = (50 * Gen_criterion_1(rec_img, real_img) + 100 * Gen_criterion_2(rec_img, real_img)) 
                 id_img_loss = (25 * Gen_criterion_1(fake_img, real_img) + 50 * Gen_criterion_2(fake_img, real_img))
                 g_loss = d_mask_loss + d_img_loss + rec_img_loss + id_mask_loss + id_img_loss  # Only the mask loss in this case


                grad_scaler.scale(g_loss).backward()
                grad_scaler.step(optimizer_G)
                grad_scaler.update()

                set_requires_grad(D1, True)
                set_requires_grad(D2, True)
                
                real_img_loss = d_criterion(D1(DiffAugment(real_img,p=args.p_diff)), valid)
                fake_img_loss = d_criterion(D1(DiffAugment(fake_img.detach(),p=args.p_diff)), fake)
                d_img_loss = (real_img_loss + fake_img_loss) / 2

                grad_scaler.scale(d_img_loss).backward()
                grad_scaler.step(optimizer_D1)
                grad_scaler.update()

                real_mask_loss = d_criterion(D2(real_mask), valid)
                fake_mask_loss = d_criterion(D2(fake_mask_p.detach()), fake)
                d_mask_loss = (real_mask_loss + fake_mask_loss) / 2

                if epoch % 10 == 0:
                    writer.add_scalar('Loss/Generator', g_loss.item(), epoch)
                    writer.add_scalar('Loss/Discriminator', d_mask_loss.item() + d_img_loss.item(), epoch)
                    if use_generated:
                       writer.add_scalar('Loss/Segmentation', rec_mask_loss.item(), epoch)
                    grid_img = torchvision.utils.make_grid(fake_img[:4])
                    writer.add_image('Generated Images', grid_img, epoch * len(train_maintrain_iterator) + step)
                    grid_mask = torchvision.utils.make_grid(fake_mask_p[:4])
                    writer.add_image('Generated Masks', grid_mask, epoch * len(train_maintrain_iterator) + step)

                grad_scaler.scale(d_mask_loss).backward()
                grad_scaler.step(optimizer_D2)
                grad_scaler.update()

        print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, args.max_epoch, d_mask_loss.item()+d_img_loss.item(), g_loss.item()))
        
        scores = evaluate_segmentation(Seg, valid_iterator, device, Seg_criterion, len(valid_data), is_avg_prec=True, prec_thresholds=[0.5], output_dir='train_val_maintrain')
        #scores = evaluate_segmentation(Seg, train_maintrain_iterator, device,Seg_criterion,len(train_maintrain_data),is_avg_prec=True,prec_thresholds=[0.5],output_dir=None)
        if epoch % 10 == 0:
            writer.add_scalar('Loss/Validation', scores['avg_val_loss'], epoch)
        
        if scores['avg_fscore'] is not None:
            logging.info('>>>> Epoch:%d  , Dice score=%f , avg fscore=%f' % (epoch,scores['dice_score'], scores['avg_fscore']))
            writer.add_scalar('Metrics/Avg_FScore', scores['avg_fscore'].item(), epoch)
        else:
            logging.info('>>>> Epoch:%d  , Dice score=%f' % (epoch,scores['dice_score']))
            '''      
        if scores['avg_fscore'] is not None and scores['avg_fscore']>avg_fscore_best:
            avg_fscore_best = scores['avg_fscore']
            real_masks = scores['real_masks']
            pred_masks = scores['predicted_masks']
            if real_masks is not None and pred_masks is not None:
                real_masks_tensor = torch.stack([torch.from_numpy(mask * 255) for mask in real_masks]).byte()
                pred_masks_tensor = torch.stack([torch.from_numpy(mask * 255) for mask in pred_masks]).byte()
                real_masks_tensor = real_masks_tensor.cpu()
                pred_masks_tensor = pred_masks_tensor.cpu()
                grid_real_masks = torchvision.utils.make_grid(real_masks_tensor.unsqueeze(1), normalize=False, scale_each=True)
                grid_pred_masks = torchvision.utils.make_grid(pred_masks_tensor.unsqueeze(1), normalize=False, scale_each=True)
                writer.add_image('Real mask list at Epoch %d' % epoch, grid_real_masks, epoch)
                writer.add_image('Predicted mask list at Epoch %d' % epoch, grid_pred_masks, epoch)
            
            attention_overlay_images = scores['attention_overlay_images']  # use this block if attention overlay is required by the user in validation results
            if attention_overlay_images is not None:
                attention_overlay_images_tensor = torch.stack([torch.from_numpy(overlay_image).permute(2, 0, 1) for overlay_image in attention_overlay_images]) 
                attention_overlay_images_tensor = attention_overlay_images_tensor.cpu()
                grid_attention_overlay_images = torchvision.utils.make_grid(attention_overlay_images_tensor, normalize=False, scale_each=True)
                writer.add_image('Real image with attention overlay at Epoch %d' % epoch, grid_attention_overlay_images, epoch)
       '''   
        if epoch>39:
            #print("Starting to save checkpoints!!!")
                
            if scores['avg_fscore'] is not None and scores['avg_fscore']>avg_fscore_best:
                avg_fscore_best = scores['avg_fscore']
                real_masks = scores['real_masks']
                pred_masks = scores['predicted_masks']
                if real_masks is not None and pred_masks is not None:
                   real_masks_tensor = torch.stack([torch.from_numpy(mask * 255) for mask in real_masks]).byte()
                   pred_masks_tensor = torch.stack([torch.from_numpy(mask * 255) for mask in pred_masks]).byte()
                   real_masks_tensor = real_masks_tensor.cpu()
                   pred_masks_tensor = pred_masks_tensor.cpu()
                   grid_real_masks = torchvision.utils.make_grid(real_masks_tensor.unsqueeze(1), normalize=False, scale_each=True)
                   grid_pred_masks = torchvision.utils.make_grid(pred_masks_tensor.unsqueeze(1), normalize=False, scale_each=True)
                   writer.add_image('Real mask list at Epoch %d' % epoch, grid_real_masks, epoch)
                   writer.add_image('Predicted mask list at Epoch %d' % epoch, grid_pred_masks, epoch)
            
                   attention_overlay_images = scores['attention_overlay_images']  # use this block if attention overlay is required by the user in validation results
                   if attention_overlay_images is not None:
                      attention_overlay_images_tensor = torch.stack([torch.from_numpy(overlay_image).permute(2, 0, 1) for overlay_image in attention_overlay_images]) 
                      attention_overlay_images_tensor = attention_overlay_images_tensor.cpu()
                      grid_attention_overlay_images = torchvision.utils.make_grid(attention_overlay_images_tensor, normalize=False, scale_each=True)
                      writer.add_image('Real image with attention overlay at Epoch %d' % epoch, grid_attention_overlay_images, epoch)
            
                      if save_checkpoint:
                         torch.save(Gen.state_dict(), os.path.join(args.output_dir, 'Gen.pth'))
                         torch.save(Seg.state_dict(), os.path.join(args.output_dir, 'Seg.pth'))
                         torch.save(D1.state_dict(), os.path.join(args.output_dir, 'D1.pth'))
                         torch.save(D2.state_dict(), os.path.join(args.output_dir, 'D2.pth'))
                         logging.info('>>>> Save the model checkpoints to %s'%(os.path.join(args.output_dir)))
                         nstop_maintrain=0
                    
            elif scores['avg_fscore'] is not None and scores['avg_fscore']<=avg_fscore_best:
                  nstop_maintrain +=1
        
            if nstop_maintrain == args.patience_maintrain:
               print('INFO: Early Stopping met ...')
               print('INFO: Finish training process')
               break
      
        scheduler_G.step(scores['avg_fscore'] if scores['avg_fscore'] is not None else 0)
        scheduler_D1.step(scores['avg_fscore'] if scores['avg_fscore'] is not None else 0)
        scheduler_D2.step(scores['avg_fscore'] if scores['avg_fscore'] is not None else 0)
        
        writer.add_scalar('Training/Mix Ratio', mix_ratio, epoch)
        writer.add_scalar('Training/Gen Frozen', float(epoch < freeze_gen_epochs), epoch)
    print("")
    print("Learning Rate of G:", scheduler_G.get_last_lr())
    print("Learning Rate of D1:", scheduler_D1.get_last_lr())
    print("Learning Rate of D2:", scheduler_D2.get_last_lr())

    writer.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_set_dir", required=True, type=str, help="path for the train dataset")
    ap.add_argument("--lr", default=1e-4, type=float, help="learning rate") 
    ap.add_argument("--pretrain_ratio", default=0.9, type=float, help="pretraining split")
    ap.add_argument("--maintrain_ratio", default=0.9, type=float, help="maintraining split")
    ap.add_argument("--max_epoch", default=300, type=int, help="maximum epoch to train model")
    ap.add_argument("--max_pretrain_epoch", default=500, type=int, help="maximum epoch to pretrain model")
    ap.add_argument("--batch_size", default=2, type=int, help="train batch size")
    ap.add_argument("--output_dir", required=True, type=str, help="path for saving the train log and best checkpoint")
    ap.add_argument("--p_vanilla_pretrain", default=0.2, type=float, help="probability value of vanilla augmentation")
    ap.add_argument("--p_vanilla_maintrain", default=0.2, type=float, help="probability value of vanilla augmentation")
    ap.add_argument("--p_diff", default=0.2, type=float, help="probability value of diff augmentation, a value between 0 and 1")
    ap.add_argument("--seg_model", required=True, type=str, help="segmentation model type (DeepSea or CellPose or UNET)")
    ap.add_argument("--patience_pretrain", default=30, type=int, help="Number of patience epochs for early stopping")
    ap.add_argument("--patience_maintrain", default=30, type=int, help="Number of patience epochs for early stopping")
    ap.add_argument("--gen_nc", default=1, type=int, help="1 for 2D or 3 for 3D, the number of generator output channels")

    args = ap.parse_args()
    assert os.path.isdir(args.train_set_dir), 'No such file or directory: ' + args.train_set_dir
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
