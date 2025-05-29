from torch import Tensor
import torch
import torch.nn.functional as F
#from torch.nn import Module
import torch.nn as nn

from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # If input image has 1 channel (grayscale), duplicate it to have 3 channels
        if X.shape[1] == 1:
            X = X.repeat(1, 3, 1, 1)
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss  


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, num_classes=2, focal_weight=1, ce_weight=0, 
                 threshold=0.3, step_size=0.02,useadaptiveloss = False, tversky_alpha=0.3, tversky_beta=0.7,
                 tv_weight=1, focal_tv_weight=0, dice_weight=0, delta=2/3,fce=0,
                 dice=0):
        super(CombinedLoss, self).__init__()
        
        # Initialize parameters
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        self.threshold = threshold
        self.step_size = step_size
        self.dice_weight = dice_weight
        self.delta = delta
        self.tv_weight = tv_weight
        self.focal_tv_weight = focal_tv_weight
        self.useadaptiveloss = useadaptiveloss
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.fce=fce; self.dice=dice

        # Register buffers for dynamic parameters and coverage tracking
        #init_alpha = max(0.3, min(tversky_alpha, 0.7))
        #init_beta = 1.0 - init_alpha
        #self.register_buffer('tversky_alpha', torch.tensor(tversky_alpha))
        #self.register_buffer('tversky_beta', torch.tensor(tversky_beta))
        #self.register_buffer('input_coverage_sum', torch.tensor(0.0))
        #self.register_buffer('input_coverage_count', torch.tensor(0))
        #self.register_buffer('target_coverage_sum', torch.tensor(0.0))  # For target coverage tracking
        #self.register_buffer('target_coverage_count', torch.tensor(0))  # For target coverage tracking

    def forward(self, inputs, targets, smooth=1e-6):
        # Calculate coverage for dynamic alpha/beta updates
        #device = inputs.device
        #self.input_coverage_sum = self.input_coverage_sum.to(device)
        #self.input_coverage_count = self.input_coverage_count.to(device)
        #self.target_coverage_sum = self.target_coverage_sum.to(device)
       # self.target_coverage_count = self.target_coverage_count.to(device)

       # with torch.no_grad():
            # Predicted mask coverage
          #  pred_probs = F.softmax(inputs, dim=1)[:, 1]  # Get foreground probabilities
           # pred_masks = (pred_probs > 0.5).float()
           # H, W = pred_masks.shape[-2], pred_masks.shape[-1]
           # pred_coverages = torch.sum(pred_masks, dim=(-2, -1)) / (H * W)
           # self.input_coverage_sum += torch.sum(pred_coverages)
           # self.input_coverage_count += inputs.shape[0]

            # Target mask coverage
            #targets_float = (targets == 1).float()  # Convert targets to binary mask
           # target_coverages = torch.sum(targets_float, dim=(-2, -1)) / (H * W)
           # self.target_coverage_sum += torch.sum(target_coverages)
           # self.target_coverage_count += inputs.shape[0]

        # Calculate Cross-Entropy loss
        ce_loss = F.cross_entropy(inputs, targets)

        # Calculate Dice loss
        inputs_softmax = F.softmax(inputs, dim=1)[:, 1]
        targets_float = (targets == 1).float()
        intersection = (inputs_softmax * targets_float).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / 
                         (inputs_softmax.sum() + targets_float.sum() + smooth))

        # Calculate Focal Loss
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Calculate Tversky Loss with current parameters
        true_pos = (inputs_softmax * targets_float).sum()
        false_neg = ((1 - inputs_softmax) * targets_float).sum()
        false_pos = (inputs_softmax * (1 - targets_float)).sum()
        tversky_loss = 1 - (true_pos / (true_pos + 
                            self.tversky_beta * false_neg + 
                            self.tversky_alpha * false_pos + smooth))
        
        # Calculate Focal Tversky Loss
        focal_tversky_loss = tversky_loss ** (1 / self.delta)
        self.fce =  self.focal_weight * focal_loss.mean()
        self.dice= self.dice_weight * dice_loss
        
        # Combine losses
        combined_loss = (self.dice_weight * dice_loss +
                         self.focal_weight * focal_loss +
                         self.ce_weight * ce_loss +
                         self.tv_weight * tversky_loss +
                         self.focal_tv_weight * focal_tversky_loss)

        return combined_loss

    def update_alpha_beta(self):
        """Update Tversky parameters after epoch based on comparison of predicted and target coverage."""
       
        if self.useadaptiveloss and self.input_coverage_count > 0 and self.target_coverage_count > 0:
            # Calculate mean coverage of predicted and target masks
            mean_pred_coverage = (self.input_coverage_sum / self.input_coverage_count).item()
            mean_target_coverage = (self.target_coverage_sum / self.target_coverage_count).item()

            # Compare predicted coverage with target coverage
            if mean_pred_coverage > mean_target_coverage:
                # If predicted coverage is higher than target coverage, adjust alpha and beta
                new_alpha = self.tversky_alpha.item() + self.step_size
                new_beta = self.tversky_beta.item() - self.step_size
                self.focal_tv_weight = self.focal_tv_weight#0.7
                self.focal_weight = self.focal_weight#0.3
            else:
                # If predicted coverage is lower than target coverage, adjust alpha and beta
                new_alpha = self.tversky_alpha.item() - self.step_size
                new_beta = self.tversky_beta.item() + self.step_size
                self.focal_tv_weight = self.focal_tv_weight#0.3
                self.focal_weight = self.focal_weight#0.7

            # Enforce constraints on alpha and beta
            new_alpha = max(min(new_alpha, 0.7), 0.3)
            new_beta = 1.0 - new_alpha  # Enforce sum constraint

            # Validate constraints
            assert 0.3 <= new_alpha <= 0.7 and 0.3 <= new_beta <= 0.7, "Parameter constraints violated"
            assert abs(new_alpha + new_beta - 1.0) < 1e-6, "Sum constraint violated"

            # Update Tversky parameters
            self.tversky_alpha.fill_(new_alpha)
            self.tversky_beta.fill_(new_beta)

        # Reset coverage tracking
        self.input_coverage_sum.fill_(0.0)
        self.input_coverage_count.fill_(0)
        self.target_coverage_sum.fill_(0.0)
        self.target_coverage_count.fill_(0)

        return mean_pred_coverage,mean_target_coverage, self.focal_tv_weight, self.focal_weight
        

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]
