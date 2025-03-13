# Importing Libraries
import os
import tqdm
from pathlib import Path
import argparse
import cv2
import numpy as np
import torch
from itertools import islice
from collections import defaultdict
from multiprocessing.pool import ThreadPool

# Importing the Modules
from utils import xtoySize,generate_miss,merge_result_org
from model import InpaintNetwork

class Test:
    def __init__(self,path_model,input_size,batch_size) :

        self.path_model_ = path_model
        self.input_size_ = input_size
        self.batch_size_ = batch_size

        self.set_device(path_model)

    def root_dir(self) :
        return Path(self.output)
    
    def create_sub_dir(self,name) :
        return self.root_dir.joinpath(name)
    
    def make_folders(self,folders) :
        for dir in folders :
            Path(dir).mkdir(parents=True,exist_ok=True)

    def set_device(self,path) :
        if torch.cuda.is_available() :
            self.device = torch.device("cuda")
            print("Using cpu")
        else :
            self.device = torch.device("cpu")
            print("Using gpu")

        self.model = InpaintNetwork().to(self.device)
        state_load = torch.load(path,map_location=self.device)
        self.model.load_state_dict(state_load)
        self.model.eval()

        print(f"Model is Ready and Loaded from path {path}")

    def name_file(self,path) :
        return os.path.basename(path).split('.')[0]
    
    def result_paths_file(self,img_path,mask_path) :
        img_name = self.name_file(img_path)
        mask_name = self.name_file(mask_path)

        item = {
            "img_path" : img_path,
            "mak_path" : mask_path,
            "result_path" : self.create_sub_dir("result").joinpath(f"result_{img_name}_{mask_name}.png"),
            "raw_path" : self.create_sub_dir("raw").joinpath(f"raw_{img_name}_{mask_name}"),
            "alpha_path" : self.create_sub_dir("alpha").joinpath(f"alpha_{img_name}_{mask_name}")
        }

        self.path_set.append(item)

    def result_paths_dir(self,img_dir,mask_dir) :
        img_dir = Path(img_dir)
        mask_dir = Path(mask_dir)
        imgs_path = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        masks_path = sorted(list(mask_dir.glob("*.jpg")) + list(mask_dir.glob("*.png")))

        n_img = len(imgs_path)
        n_mask = len(masks_path)
        n_min = min(n_img,n_mask)

        self.path_set = []
        for i in range(n_min) :
            self.result_paths_file(imgs_path[i],masks_path[i])

    def to_numpy(self,tensor) :
        tensor = tensor.mul(255).byte().detach().cpu().numpy()
        tensor = np.transpose(tensor,[0,2,3,1])
        return tensor
    
    def process_batch(self,batch):

        imgs = torch.from_numpy(batch["img"]).to(self.device)
        masks = torch.from_numpy(batch["mask"]).to(self.device)

        imgs.float().div(255)
        masks.float().div(255)
        masked_img = imgs*masks
        result, alpha,raw = self.model(masked_img,masks)
        # It will be k x N x C x H x W. Taking top layer
        result,alpha,raw = result[0],alpha[0],raw[0]
        result = imgs * masks + (1 - masks) * result

        result = self.to_numpy(result)
        alpha = self.to_numpy(alpha)
        raw = self.to_numpy(raw)
        
        for i in range(raw.shape[0]):
            cv2.imwrite(str(batch["result_path"][i]),result[i])
            cv2.imwrite(str(batch["alpha_path"][i]),alpha[i])
            cv2.imwrite(str(batch["raw_path"][i]),raw[i])

    def process(self,pair) :
        img = cv2.imread(str(pair["img_path"]),cv2.IMREAD_COLOR)
        mask = cv2.imread(str(pair["mask_path"]),cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img,self.input_size_)
        mask = cv2.resize(mask,self.input_size_)

        img =np.ascontiguousarray(img.transpose(2,0,1)).astype(np.uint8)
        mask = np.ascontiguousarray(np.expand_dims(mask,0)).astype(np.unit8)

        pair["img"], pair["mask"] = img, mask

        return pair


    def create_batch(self) :
        
        n_items = len(self.path_set)
        n_batch = (n_items - 1)//self.batch_size_ + 1

        for i in tqdm.trange(n_batch,leave=False) :
            buffer = defaultdict(list)
            start_i = i*self.batch_size_
            stop_i = start_i + self.batch_size_

            batch = ThreadPool().imap_unordered(self.process,islice(self.path_set,start_i,stop_i))

            for set in batch :
                for k,v in set.items() :
                    buffer[k].append(v)
            yield buffer

    def batch_generator(self) :
        generator = self.create_batch

        for buf in generator() :
            yield buf

    # def inpaint