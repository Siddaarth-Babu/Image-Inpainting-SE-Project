# Importing the Libraries
import torch
import torch.nn as nn
from torchvision import models

# Importing Modules
from utils import xtoySize

# Loss Functions are defined as classes. Details of Loss functions can be found in Report

class ReconstructionLoss(nn.L1Loss):
    """
    This loss accounts for the structure of the image. It is defined as mean
    absolute error of result and target image
    """

    def __init__(self) :
        """
        Initialises a L1 loss
        """

        # Calling the constructor ofparent classs
        super().__init__()

        # Initialising the L1 loss
        self.l1_loss = nn.L1Loss()

    def forward(self,results,targets) :
        """
        Forward pass of the results and targets to calculate the loss

        **Inputs**:
        - results: List of result images (Tensors) obtained from different layers
        - targets: List of target images (Tensors) obtained from different layers

        **Returns**:
        - Net loss: Tensor of size 1
        """

        # Initialising the loss to 0
        loss = 0.

        # Grouping the result,target of that layer and finding the l1 loss
        for idx,(res,tar) in enumerate(zip(results,targets)) :
            loss += self.l1(res,tar)

        return loss/len(targets)
    

class VGGFeatures(nn.Module) :
    """
    Used to obtain the features used for calculating the Perceptual loss and Style loss.
    """

    def __init__(self) :
        """
        Initializes a pretrained VGG16 model and the features of the selected layers
        """

        # Calling the constructor of the parent class
        super().__init__()
    
        # Initializing vgg16 model from torchvision.models
        vgg16 = models.vgg16(pretrained=True)

        # Since we are not going to train the parameters, we will set require_grad to False which stops making computation graph for backpropagation
        for par in vgg16.parameters() :
            par.requires_grad = False
        
        # Selecting the Features from 4th, 9th, 16th layer
        self.vgg16_j1 = nn.Sequential(*vgg16.features[0:5])
        self.vgg16_j2 = nn.Sequential(*vgg16.features[5:10])
        self.vgg16_j3 = nn.Sequential(*vgg16.features[10:17])

    def forward(self,x) :
        """
        Forward propagation of x in the vgg selected layers

        **Input**:
        - x: Tensor of any size

        **Returns**: List of
        - feat_1: features obtained till layer 4
        - feat_2: features obtained till layer 9
        - feat_3: features obtained till layer 16
        """

        feat_1 = self.vgg16_j1(x)
        feat_2 = self.vgg16_j2(feat_1)
        feat_3 = self.vgg16_j3(feat_2)

        return [feat_1,feat_2,feat_3]
    

class PerceptualLoss(nn.Module) :
    """
    Computes the error between the high level features obtained from a pretrained vgg network
    """

    def __init__(self) :
        """
        Initialising L1 loss
        """

        # Calling the constructor of parent class
        super().__init__()

        self.l1 = nn.L1Loss()

    def forward(self,vgg_results,vgg_targets) :
        """
        Forward pass of the features obtained from vgg network

        **Inputs**:
        - vgg_results: Features from selected layers of all result images
        - vgg_targets: Features from selected layers of all target images

        **Returns**:
        - Net loss: Average of loss all results images
        """

        # Initializing loss equal to 0
        loss = 0.

        # Since vgg_results,vgg_targets is two dimensional ie contains j features of total images
        for res,tar in zip(vgg_results,vgg_targets) :

            # Considering the j features of each image and computing L1 loss
            for feat_res,feat_tar in zip(res,tar) :
                loss += self.l1(feat_res,feat_tar)

        return loss/len(vgg_results)


class StyleLoss(nn.Module) :
    """
    It compares the correlation between the feature maps using the gram matrix
    """

    def __init__(self) :
        """
        Initialising L1 loss
        """

        # Calling the constructor of parent class
        super().__init__()
        self.l1loss = nn.L1Loss()

    def gram_matrix(self,feature) :
        """
        It represents the correlation between features channels

        **Input**:
        - feature: Represents the vgg feature

        **Returns**:
        - Gram Matrix of the feature
        """

        # Unpacking the feature's shape
        N,C,H,W = feature.shape

        # Reshaping the feature tensor to (N,C,H*W) 
        feature = feature.view(N,C,H*W)

        # Since feature contains batches we need to perform batch matrix multiplication with its transpose 
        # (N,C,H*W) @ (N,H*W,C) = (N,C,C)
        gram_mat = torch.bmm(feature,torch.transpose(feature,1,2))

        return gram_mat / (H*W*C)
    
    def forward(self,vgg_results,vgg_targets) :
        """
        Forward pass for finding the L1 loss of gram matrices of 
        vgg features of results and targets image 
        
        **Inputs**:
        - vgg_results: Features from selected layers of all result images
        - vgg_targets: Features from selected layers of all target images

        **Returns**:
        - Net loss: Average of loss all results images
        """

        # Initialising the loss to 0
        loss = 0.

        # Grouping each vgg feature of  result and target image
        for res,tar in zip(vgg_results,vgg_targets) :

            # Considering the feature from the selected layer individually and computing the L1 loss
            for feat_res,feat_tar in zip(res,tar) :
                loss += self.l1(self.gram_matrix(feat_res),self.gram_matrix(feat_tar))
        
        return loss/len(vgg_results)

class TotalVariationLoss(nn.Module) :
    """
    Computes the error of the pixel with respect to its top and left pixel
    (I<sub>x,y +1</sub> - I<sub>x,y</sub>) + (I<sub>x -1,y</sub> - I<sub>x,y</sub>)
    """

    def __init__(self,channel_img = 3) :
        """
        Initializing the 3 x 3 kernel for computing the variation loss
        """

        # Calling the constructor of the parent class
        super().__init__()

        # Initializing the channels of image
        self.channel_img = channel_img

        # Creating a kernel to calculate the variation in each pixel of the image
        kernel = torch.FloatTensor([
            [0,1,0],[1,-2,0],[0,0,0]
        ]).view(1,1,3,3)

        # Creating 3 different kernels for handling the 3 different channels individually
        # Kernel shape is (3,1,3,3) ie (Cout,Cin,Hk,Wk). As noticed the Cin here is 1 but input image has 3 channels. We will use grouped convolution
        self.kernel = torch.cat([kernel]*3,dim=0)

    def variation(self,x) :
        """
        Calculates the variation by passing the image to conv layer with our kernel and groups = 3
        The height and width remains same since we are using stride=1 and padding=1 for kernel size = 3

        **Input**:
        - x: Tensor of size (N,Cin=3,H,W)

        **Ouput**:
        - out: Tensor of size (N,Cout=3,H,W)
        """
        return nn.functional.conv2d(x,stride=1,padding=1,groups=self.channel_img)
    
    def forward(self,results,mask) :
        """
        Forward pass of our results obtained with the mask to find the variation loss for the missing part

        **Inputs**:
        - results: Tensor images obtained from selected layers
        - mask: Grayscale mask image

        **Result**:
        - Net loss: Average of loss all results images
        """

        # Initializing the loss to 0
        loss=0.

        # Adding a extra dimension to mask to avoid broadcasting error
        mask = torch.from_numpy(mask).unsqueeze(0)

        # For each result image finding the variation of the cut out portion
        for idx,res in enumerate(results) :

            var = self.variation(res) * (xtoySize(mask,res) <= 127)

            # Finding the mean of the variation in the image
            loss += torch.mean(torch.abs(var))

        return loss/len(results)
    
class InpaintLoss(nn.Module) :
    """
    This is the total loss function incorporating the structure loss and texture loss
    """

    def __init__(self,channel_img=3,s_layers = [0,1,2,3,4,5],t_layers=[0,1,2],w_l1=6.,w_per=0.1,w_style=240.,w_tv=0.1) :
        """
        Initializing all the loss function and their respective weights

        **Inputs**:
        - channel_img: Number of channels in the image
        - s_layers: List of index of the structure layers (ie in the final U Network)
        - t_layers: List of index of the texture layers (ie in the final U Network)
        - w_l1: Weight for the L1 loss
        - w_per: Weight for the Perceptual loss
        - w_style: Weight for the style loss
        - w_tv: Weight for the tv loss
        """

        # Calling the constructor of the parent class
        super().__init__()

        # Saving the inputs as class parameters
        self.struct_layers = s_layers
        self.text_layers = t_layers

        self.w_l1 = w_l1
        self.w_per = w_per
        self.w_style = w_style
        self.w_tv = w_tv

        self.vgg = VGGFeatures()
        self.rloss = ReconstructionLoss()
        self.ploss = PerceptualLoss()
        self.stloss = StyleLoss()
        self.tvloss = TotalVariationLoss(channel_img=channel_img)

    def forward(self,results,target,mask) :
        """
        Forward pass of results, target, mask finding the total loss including structure loss and texture loss

        **Input**:
        - results: List of results images (tensor). Need not be of the same size as target image
        - target: Target image (tensor).
        - mask: Grayscale mask image
        """

        # Resizing the target image according to the images in the results
        targets = [xtoySize(target,res) for res in results]

        # Initializing structure loss and texture loss to 0
        loss_structure = 0.
        loss_texture = 0.

        # Finding Structure loss
        if len(self.struct_layers) > 0 :

            # Storing the results and targets image corresponding to the structure layers selected
            res_struct = [results[i] for i in self.struct_layers]
            tar_struct = [targets[i] for i in self.struct_layers]

            # Calling the forward function of Reconstruction loss
            loss_structure = self.rloss(res_struct,tar_struct) * self.w_l1

        # Finding the texturee loss
        if len(self.text_layers) > 0:
            
            # Storing the results and targets image corresponding to the texture layers selected 
            res_text = [results[i] for i in self.text_layers]
            tar_text = [targets[i] for i in self.text_layers]

            # Storing the vgg features of the results and target images of the texture layers
            vgg_res = [self.vgg(res) for res in res_text]
            vgg_tar = [self.vgg(tar) for tar in tar_text]

            # Calling the forward function of Perceptual loss, Style loss, TotalVariation loss
            loss_per = self.ploss(vgg_res,vgg_tar)*self.w_per
            loss_style =self.stloss(vgg_res,vgg_tar)*self.w_style
            loss_tv = self.tvloss(res_text,mask)*self.w_tv

            # Computing the overall texture loss
            loss_texture = loss_style + loss_per + loss_tv

        # Computing the overall loss (structure + texture) for the image
        loss_total = loss_structure + loss_texture

        return loss_total
        







