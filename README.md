# SAGE-GAN
SAGE-GAN: Our proposed model which consists of a self-attention U-Net which is pretrained from scratch on a small set of real EM images to accurately segment nanoparticle features while suppressing background noise. This pretrained U-Net is then embedded into a CycleGAN (cGAN-Seg) architecture, enabling the generation of synthetic EM images that are structurally aligned with ground-truth masks via cycle consistency. This integration enhances nanoparticle segmentation accuracy and robustness, especially in data-limited scenarios. This repository contains the scripts for training and evaluating our model, as well as the publicly available S1 nanoparticle dataset upon which this model was trained. 

![SAGE-GAN_model2](https://github.com/user-attachments/assets/150f2e85-41d5-42ce-b72b-5c8b802fedf1)
# Dataset
The publicly available S1 nanoparticle dataset's link is being provided here: https://doi.org/10.6084/m9.figshare.11783661.v1

This dataset has been cited in the following paper: Boiko, Daniil; Pentsak, Evgeniy; Cherepanova, Vera; Ananikov, Valentine (2020). Electron microscopy dataset for the recognition of nanoscale ordering effects and location of nanoparticles
# Requirements
- Install the requirements using the following code:
  <pre>pip install -r requirements.txt</pre>
# Utilization
## Training SAGE-GAN
The model can be trained using either a user-supplied dataset or the S1 dataset whose link is provided in this repository. Prior to training, users are advised to configure key hyperparameters, including the choice of segmentation architecture (Attention Unet or UNet) and the early stopping patience parameters. Throughout the training process, the script records detailed logs and periodically saves model checkpoints for the segmentation network (Seg.pth), generator (Gen.pth), and discriminators (D1.pth and D2.pth) within the designated output_dir.
<pre>Example:
python pretrain+finetune.py --seg_model UNET --train_set_dir  .../S1 dataset/train  --lr 0.0001 --p_vanilla 0.2 --p_diff 0.2 --patience 500 --output_dir tmp/</pre>
## Testing the segmentation model
Evaluate the segmentation model using the S1 dataset or yours, specifying the segmentation model type (seg_model) and its checkpoint directory (seg_ckpt_dir).
<pre>Example:
python test_segmentation_model.py --seg_model UNET --test_set_dir .../S1 dataset/test --seg_ckpt_dir .../SAGE-GAN_checkpoints/UNET_model/S1 dataset/Seg.pth --output_dir tmp/</pre>
## Testing the generator model
Evaluate the StyleUNET generator's performance, using the synthetic or real mask images using the following example code:
<pre>Example:
python test_generation_model.py --test_set_dir .../S1 dataset/test/ --gen_ckpt_dir .../SAGE-GAN_checkpoints/UNET_model/S1 dataset/Gen.pth --output_dir tmp/</pre>
# Comparitive Models
A few standard segmentation architectures like UNET,UNET++,DCUNET etc. and generation architectures like GAN and CycleGAN have been implemented, incorporating them within our designed pipeline, including the data augmentations used in our proposed model. These scripts can be found in the folder, named "comparitive models".
# Useful Information
For any queries, contact varunajith29@gmail.com / anindyapal264@gmail.com .
