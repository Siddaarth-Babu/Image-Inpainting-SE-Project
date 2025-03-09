# Importing the libraries

import cv2
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Defining the utility functions

def get_img_paths(path) :
    """
    It lists out the images path for a given directory otherwise just return the 
    path of the image file

    **Input**:
    - path: A string representing the location of the image directory or file

    **Returns**:
    - .png, .jpg, .jpeg: List of all the paths with this extension in sorted order
    """

    # Check if the path is a file and return a single path
    if Path(path).is_file() :
        return [Path(path)]
    
    # Returns the path of set of images in the directory
    else :
        return sorted(list(Path(path).glob('*.png')) + list(Path(path).glob('*.jpg')) + list(Path(path).glob('*.jpeg')))


def xtoySize(x,y,mode='billinear') :
    """
    It converts the image x to fit the shape of image y by interpolation. Either
    upsamples or downsamples based on input y

    **Inputs**:
    - x: Tensor of size (N,Cx,Hx,Wx)
    - y: Tensor of size (N,Cy,Hy,Wy)
    - mode: This is used for interpolation (can be "billinear","nearest","linear" etc) 

    **Returns**:
    - int_x: A interploated version of x according to shape of y. Tensor of size (N,Cy,Hy,Wy)
    """

    # For even spacing we used align_corners = False
    int_x = F.interpolate(x,y.shape[-2:],mode=mode,align_corners=False)

    return int_x

def list_to_numpy(lst,dtype=None) :
    """
    Converts n dimensional list to numpy ndim array

    **Inputs**:
    - lst: List of any dimension
    - dtype: dtype used for conversion to numpy array

    **Returns**:
    - res_np: Numpy array of given dtype and dimension
    """

    # Check if list is 1 dimensionsal
    if isinstance(lst[0],(int,float,np.number)) :
        return np.array(lst,dtype=dtype)
    
    # For list of dimension > 1
    else :

        # Checking the first element it may be >=1 dimension. So we get Numpy array
        flist = list_to_numpy(lst[0])

        # if dtype is None then use the dtype of flist
        if dtype is None :
            dtype = flist.dtype

        # Getting the shape of final numpy array ie (# length of lst, shape of each entry in it)
        final_shape = [len(lst)] + list(flist.shape)

        # Initialize empty numpy array for storing the list values
        res_np = np.empty(final_shape,dtype=dtype)

        # Copying the values to numpy array
        for idx,i in enumerate(lst) :
            res_np[idx] = i

        return res_np
    
def generate_miss(org_img_path,mask_path,out_path) :
    """
    Generates the masked out image from the original image and the mask

    **Inputs**:
    - org_img_path: string representing the location of original image (can be a dir or file)
    - mask_path: string representing the location of mask (can be a dir or file)
    - out_path: string representing the location of output drectory
    """

    # Obtaining the list of Images paths
    imgs_paths = get_img_paths(org_img_path)

    # Obtaining the list of Mask paths
    mask_paths = get_img_paths(mask_path)

    # Grouping the respective image and mask for finding the masked image
    for idx,(img,mask) in tqdm(enumerate(zip(imgs_paths,mask_paths))) :

        # Creating a path to store the masked image in the output directory
        out = Path(out_path).joinpath(f"miss_{idx+1}.png")

        # Reading the image and mask in COLOR and GRAYSCALE respectively
        img = cv2.imread(str(img),cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask),cv2.IMREAD_GRAYSCALE)

        # Resizing the mask to match the image size
        # Cv2.resize expects shape of the form (Width,Height) 
        mask = cv2.resize(img,img.shape[:2][::-1])

        # Since mask has only 2 dimension, we expand the channel dimension
        mask = np.expand_dims(mask,axis=2)

        # The cut portion is near to black so it has value < 127. For masked image we want that region to be white
        miss = img * (mask > 127) + 255 * (mask <= 127)

        # Saving the masked image
        cv2.imwrite(str(out),miss)

def merge_result_org(dirs,out_path) :
    """
    Merges the masked image, alpha composition, raw, result and original image

    **Inputs**:
    - dirs: List of directories containing the above mentioned images
    - out_path: A path to the output directory where the merged image is stored
    """

    # Obtaining the list of images path in each directory
    img_paths_list = [get_img_paths(dir_path) for dir_path in dirs]

    # Obtaining the list of length of images in each directory
    count_imgs = [len(dir_images) for dir_images in img_paths_list]

    # Printing the total images in each directory
    print("Total images present are: ",count_imgs)

    # making sure that no direcotry is empty
    assert min(count_imgs) > 0, "Please check the path of empty folder."

    # Creating the direcotry if it doesn't exist
    out_dir = Path(out_path)
    out_dir.mkdir(exist_ok=True,parents=True)

    # Grouping the images in each directory
    for idx, set in tqdm(enumerate(zip(*img_paths_list))) :

        # Creating a path to store the merged image
        filename = out_dir.joinpath(f"merge_{idx+1}.png")

        # Creating a figure and later on subplots of length of the dirs
        plt.figure(figsize=(15,7))
        total_img_set = len(set)

        # Iterating on image in each group
        for j,img in enumerate(set) :

            # Reading the image in RGB format
            img = cv2.imread(str(img),cv2.COLOR_BGR2RGB)

            # Creating a subplot with proper index
            plt.subplot(f"1{total_img_set}{j+1}")
            plt.imshow(img)
            plt.axis('off')

        # Creating a padding between subplots
        plt.tight_layout()

        # Saving the fig in the required path
        plt.savefig(str(filename))