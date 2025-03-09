# Importing the libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Importing the modules
from utils import xtoySize

# Normalization layer

def normalization(name,c_out) :
    """
    Provides a normalization layer either Batch wise or instance wise

    **Inputs**:
    - name: Either 'batch' or 'instance'
    - c_out: No of output channels for Normalization layer

    **Returns**:
    - norm: Normalization layer based on the name
    """

    # For Batch Normalisation
    if name == 'batch' :
        norm = nn.BatchNorm2d(c_out)

    # For Instance Normalisation
    elif name == 'instance' :
        norm = nn.InstanceNorm2d(c_out)

    else :
        norm = None

    return norm

# Activation layer

def activation(name) :
    """
    Provides a activation layer for the network based on the name provided
    
    **Inputs**:
    - name: Either 'relu' , 'leaky_relu', 'sigmoid', 'tanh', 'elu'

    **Returns**:
    - act: Activation layer based on the name provided
    """

    # For ReLU Activation
    if name == 'relu' :
        act = nn.ReLU()

    # For LeakyReLU Activation
    elif name == 'leaky_relu' :
        act = nn.LeakyReLU()

    # For Tanh Activation
    elif name == 'tanh' :
        act = nn.Tanh()
    
    # For Sigmoid Activation
    elif name == 'sigmoid' :
        act = nn.Sigmoid()

    # For ELU Activation
    elif name == 'elu' :
        act = nn.ELU()

    else :
        act = None

    return act


class SameDimConv2d(nn.Module) :
    """
    Provides a Convolution layer such that output image will have same spatial volume as input image ie (H,W)
    """

    def __init__(self,c_in,c_out,kernel_size,stride) :
        """
        Initialises the padding based on stride and kernel size such that output image has same spatial
        dimesions as input (only for stride = 1)

        **Inputs**:
        - c_in: Channels of Input image
        - c_out: Channels of output image
        - kernel_size: Int k where kernel spatial size would be k x k
        - stride: Represents the gap to be leaved during convolution  
        """

        # Calling the constructor of parent class
        super().__init__()

        # Calling member function to get the appropriate padding
        padding = self.get_pad(kernel_size,stride)

        # When returned padding is tuple, then we can't use this padding as a parameter 
        # for nn.Conv2d because it expects it to be (P1,P2) where P1 for left,right and P2 for top,bottom.
        # But in our case P1 for left,top and P2 for right,bottom. So we use Constant Padding
        if type(padding) is tuple :
            self.conv2d = nn.Sequential(
                
                # Constant Padding expects padding to be (left,right,top,bottom)
                nn.ConstantPad2d(padding=padding*2,value=0),

                # Since we have already set the padding. So no need of padding in nn.Conv2d
                nn.Conv2d(c_in,c_out,kernel_size,stride,padding=0)
            )

        # When padding is single integer, we can directly use the nn.Conv2d padding parameter
        else :
            self.conv2d = nn.Conv2d(c_in,c_out,kernel_size,stride,padding=padding)


    def get_pad(self,kernel_size,stride) :
        """
        Provides the padding needed for same spatial dimension based on kernel size and stride (fro stride=1)

        **Inputs**:
        - kernel_size: Int k where kernel spatial size would be k x k
        - stride: Represents the gap to be leaved during convolution

        **Returns**:
        - padding: can be int or tuple
        """

        # Hout = (Hin + 2*padding - kernel_size)/stride + 1
        # Wout = (Win + 2*padding - kernel_size)/stride + 1
        if (kernel_size - stride) % 2 == 0 :

            pad = (kernel_size - stride) // 2
            return pad
        
        else :

            left = (kernel_size - stride) // 2
            right = left + 1
            return left,right
        
    def forward(self,x) :
        """
        Forward pass to the convolution layer

        **Input**:
        - x: Image tensor

        **Returns**:
        - out: Output image from the conv layer
        """
        return self.conv2d(x)


class Upsample(nn.Module) :
    """
    Upsamples the image according to given scale factor. Needed because in Encoder 
    we used stride of two that halves the image size so in teh decoder side we need
    to upsample the image with the scale of 2
    """

    def __init__(self,mode='nearest',scale=2) :
        """
        Initialising the mode and scale. 

        **Inputs**:
        - mode: Parameter in Interpolate functions which tells the way to fill the extra pixels.
        Can be 'nearest', 'linear', 'bilinear' etc
        - scale: Scale factor to increase the spatial dimension of input image
        """

        # Calling the constructor of the parent class
        super().__init__()

        # Storing the inputs as class parameters
        self.mode = mode
        self.scale_factor = scale

    def forward(self,x) :
        """
        Forward pass of input image for upsampling

        **Input**:
        - x: Image tensor

        **Returns**:
        - out: Output upsampled image
        """
        return F.interpolate(x,scale_factor=self.scale_factor,mode=self.mode)
    

class Encoder(nn.Module) :
    """
    This corresponds to the first part of the U network. This leads to extraction of features from 
    from the masked portion
    """

    def __init__(self,c_in,c_out,kernel_size,stride,norm=None,act=None) :
        """
        Initialises the layers needed for the encoder

        **Inputs**:
        - c_in: Channels of Input image
        - c_out: Channels of output image
        - kernel_size: Int k where kernel spatial size would be k x k
        - stride: Represents the gap to be leaved during convolution
        - norm: A string specifying the name of the normalisation required
        ('batch' or 'instance')
        - act: A string specifying the name of the activation required
        ('relu' , 'leaky_relu', 'sigmoid', 'tanh', 'elu')
        """

        # Calling the constructor of the parent class
        super().__init__()

        # Storing the channels as class parameters
        self.c_in = c_in
        self.c_out = c_out

        # Initialising a empty list to append all the layers necessary for encoder
        layers = []

        # Appending the convolution layer with the kernel size and stride
        layers.append(
            SameDimConv2d(c_in,c_out,kernel_size,stride)
        )

        # If normalization is required then the layer is appended
        if norm :
            layers.append(normalization(norm,c_out))

        # If activation is required then the layer is appended
        if act :
            layers.append(activation(act))

        # Unpacking the layers and forming a sequence for the input to pass in
        self.encode = nn.Sequential(*layers)

    def forward(self,x) :
        """
        Forward pass of input image to encoder

        **Input**:
        - x: Image tensor

        **Returns**:
        - out: Output encoded image
        """
        return self.encode(x)
    

class Decoder(nn.Module) :
    """
    This corresponds to teh second part ofthe U-Net. This generates the features corresponding
    to missing region from the image 
    """

    def __init__(self,c_back,c_encode,c_out,kernel_size,scale=2,mode='nearest',norm='batch',act='relu') :
        """
        Initialising the layers of the decoder block

        **Inputs**:
        - c_back: Number of channels of the previous decoded feature
        - c_encode: Number of channels of the parallel encoder feature (similar to transfer learning)
        - kernel_size: Int k where kernel spatial size would be k x k
        - mode: Parameter in Interpolate functions which tells the way to fill the extra pixels.
        Can be 'nearest', 'linear', 'bilinear' etc
        - scale: Scale factor to increase the spatial dimension of input image
        - norm: A string specifying the name of the normalisation required
        ('batch' or 'instance')
        - act: A string specifying the name of the activation required
        ('relu' , 'leaky_relu', 'sigmoid', 'tanh', 'elu')
        """

        # Calling the constructor of the parent class nn.Module
        super().__init__()

        # Storing the channel inputs as class parameters
        # Since we are concatenating image obtained from previous decoder and parallel encoder, thus channels will be added up
        self.c_in = c_back + c_encode
        self.c_out = c_out
        self.c_encode = c_encode

        # Since at every step we need to double the size of image we upsample it
        self.up = Upsample(mode,scale)

        # Initialising a empty list to append all the layers necessary for decoder
        layers = []

        # Appending the convolution layer with the kernel size and stride
        layers.append(
            SameDimConv2d(self.c_in,self.c_out,kernel_size,stride=1)
        )

        # If normalization is required then the layer is appended
        if norm :
            layers.append(normalization(norm,self.c_out))

        # If activation is required then the layer is appended
        if act :
            layers.append(activation(act))

        # Unpacking the layers and forming a sequence for the input to pass in
        self.decode = nn.Sequential(*layers)

    def forward(self,x,encode) :
        """
        Forward pass of input image to encoder

        **Input**:
        - x: Image tensor
        - encode: Represents the parallel encoded feature that has to be concatenated

        **Returns**:
        - out: Output encoded image
        """

        # Upsampling the previous decoded image
        out = self.up(x)

        # Concatenating the encoded feature with it along the channel dimension
        if self.c_encode > 0:
            out = torch.cat([out,encode],dim=1)
        
        # Passing the image to decoder network
        out = self.decode(out)

        return out
    

class Blending(nn.Module) :
    """
    It causes the smoothening effect in the edges seperating the missing region with the
    known pixels
    """

    def __init__(self,c_in,c_out,kernel_size,norm='batch',act='leaky_relu') :
        """
        Initialises the blending layers for edge smoothening

        **Inputs**:
        - c_in: Channels of Input image
        - c_out: Channels of output image
        - kernel_size: Int k where kernel spatial size would be k x k
        - norm: A string specifying the name of the normalisation required
        ('batch' or 'instance')
        - act: A string specifying the name of the activation required
        ('relu' , 'leaky_relu', 'sigmoid', 'tanh', 'elu')
        """

        # Calling the parent constructor
        super().__init__()

        # Setting the hidden size for blending
        c_hidden = max(c_in//2,32)

        # Initialising the blend network with layers
        self.blend = nn.Sequential(
            SameDimConv2d(c_in,c_hidden,1,1),
            normalization(norm,c_hidden),
            activation(act),
            SameDimConv2d(c_hidden,c_out,kernel_size,stride=1),
            normalization(norm,c_out),
            activation(act),
            SameDimConv2d(c_out,c_out,1,1),
            activation('sigmoid')
        )

    def forward(self,x) :
        """
        Forward pass of input image to blending

        **Input**:
        - x: Image tensor

        **Returns**:
        - out: Output blended image
        """
        return self.blend(x)
    

class Fuse(nn.Module) :
    """
    Combines the features obtained from the decoder and the masked image
    """

    def __init__(self,c_features,c_alpha=1):
        """
        Initialises the blending network and additional convolutional network

        **Inputs**:
        - c_features: Number of channels in the decoded feature
        - c_alpha: Number of channels in the alpha (blended image) 
        """

        # Calling the parent constructor
        super().__init__()

        # Initialising the image channel
        c_img = 3

        # Convolution network to extract features from the decoder image
        self.feat2raw = nn.Sequential(
            SameDimConv2d(c_features,c_img,kernel_size=1,stride=1),
            activation('sigmoid')
        )

        # Initialising a blender
        self.blend = Blending(c_img*2,c_alpha)

    def forward(self,masked_img,feat_dec) :
        """
        Forward pass for fusion of features and the masked image

        **Inputs**:
        - masked_img: Image with cut out portion and channels = 3
        - feat_dec: Features of the decoder and channels = 3

        **Returns**:
        - result: Final reconstructed image using the raw form and alpha composition 
        - alpha: Blended image of masked iamge and the raw form 
        - raw: Extracting the features from the decoder image
        """

        # Resizing the masked image according to decoder feature
        masked_img = xtoySize(masked_img,feat_dec)

        # Obtaining the raw form of image reconstruction
        raw = self.feat2raw(feat_dec)

        # Smoothening the edges using Blend of concatenated masked and raw image
        alpha = self.blend(torch.cat([masked_img,raw],dim=1))

        # Getting the final image (alpha tweaks the raw form)
        result = alpha * raw + (1 - alpha) * masked_img

        return result,alpha,raw
    
    
class InpaintNetwork(nn.Module) :
    """
    Complete Network for Image Inpainting
    """

    def __init__(self,c_img=3,c_alpha=3,c_mask=1,mode='nearest',norm='batch',act_enc='relu',
                 act_dec='leaky_relu',kernel_enc = [7,5,3,3,3,3,3,3], kernel_dec=[3]*8,
                 layers_blend = [0,1,2,3,4,5] ) :
        """
        Initialising all the encoder,decoder,fusion blocks

        **Inputs**:
        - c_img: No of channels in image
        - c_alpha: No of channels in alpha composition
        - c_mask: No of channels in mask
        - mode: Parameter in Interpolate functions which tells the way to fill the extra pixels.
        Can be 'nearest', 'linear', 'bilinear' etc
        - norm: A string specifying the name of the normalisation required
        ('batch' or 'instance')
        - act_enc: Activation for encoder
        - act_dec: Activation for decoder
        - kernel_enc: Kernel sizes for the 8 encoder layers
        - kernel_dec: Kernel sizes for the 8 decoder layers
        - layers_blend: Layers indexed from top that has to be fused
        """
        
        # Calling the parent constructor
        super().__init__()

        # Initial image will have 4 channels (3 from orginal image and 1 from mask)
        c_in = c_img + c_mask

        # Storing the length of the encoder and decoder
        self.no_enc = len(kernel_enc)
        self.no_dec = len(kernel_dec)

        # Making sure that the encoder and decoder are equal in number
        assert self.no_enc == self.no_dec, "Encoder and decoder must be equal in number"

        # making sure that top layer is blended
        assert 0 in layers_blend, "Layer 0 must always be blended"

        # Initialising empty list for encoders
        self.enc = []

        # Appending the first encoder
        self.enc.append(Encoder(c_in,64,kernel_enc[0],stride=2,norm=None,act=None))

        for enc_ksize in kernel_enc[1:] :

            # Previous c_out will be the present c_in
            c_in = self.enc[-1].c_out

            # Limiting max c_out to be 512 and appending the encoder at next index
            c_out = max(512,c_in*2)
            self.enc.append(Encoder(c_in,c_out,enc_ksize,stride=2,norm=norm,act=act_enc))
        
        # Initialising empty list for decoders and fuse
        self.dec = []
        self.fuse = []
    
        for i,dec_ksize in enumerate(kernel_dec) :

            # When i equals 0 then we need the out channel of the last encoder
            # Otherwise c_back will be the c_out of previous decoder
            c_back = self.dec[-1].c_out if i!=0 else self.enc[-1].c_out
            c_out = c_encode = self.enc[-i-1].c_in

            # Layer index notation 0 at top
            layer_index = self.no_dec - i -1

            # Appending the decoder layer
            self.dec.append(Decoder(c_back,c_encode,c_out,dec_ksize,scale=2,norm=norm,act=act_dec))

            # If blending needs to be done then we append the fuse layer
            if layer_index in layers_blend :
                self.fuse.append(Fuse(c_out,c_alpha))

            # Otherwise None is appended
            else :
                self.fuse.append(None)

    def forward(self,masked_img,mask) :
        """
        Forward pass of mask and the masked image

        **Inputs**:
        - masked_img: Image with cut out portion and channels = 3
        - mask: Grayscale image representing the cut portion (0 for cut, 1 for known region)

        **Returns**:
        - results: List of Final reconstructed image using the raw form and alpha composition from the top 
        - alphas: List of Blended image of masked iamge and the raw form from the top 
        - raws: List of Extracting the features from the decoder image from the top 
        """

        # Concatenating maksed_img and mask to povide importance for the cut out region
        start = torch.cat([masked_img,mask],dim=1)

        # Adding the input in the list of encoded images
        out_enc = [start]

        # Looping through the encoders and adding the encoded images
        for encoder in self.enc :
            start = encoder(start)
            out_enc.append(start)

        # Initialising the empty list for storing result,alpha,raw images at each layer which needs blending
        results,alphas,raws = [],[],[]

        # Looping through the decoders and fusing the required image with the masked image and storing the results
        for i,(decoder,fuse) in enumerate(zip(self.dec,self.fuse)) :
            start = decoder(start,out_enc[-i-2])

            # Checking whether the layer index needs blending or not
            if fuse :
                result, alpha, raw = fuse(masked_img,start)
                results.append(result)
                alphas.append(alpha)
                raws.append(raw)

        return results[::-1], alphas[::-1], raws[::-1]