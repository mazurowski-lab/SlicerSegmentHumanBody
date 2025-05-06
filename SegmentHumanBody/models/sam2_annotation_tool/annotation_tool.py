import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import random
import sys
import traceback

import pandas as pd
import numpy as np

from models.sam2_annotation_tool.sam2.build_sam import build_sam2_video_predictor
import nibabel as nib
from PIL import Image
import nrrd
import yaml

from models.sam2_annotation_tool.training.train import *
import submitit
import torch

from hydra import compose, initialize_config_module, initialize
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra

from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf

from models.sam2_annotation_tool.training.utils.train_utils import makedir, register_omegaconf_resolvers

import argparse
import sys

from ruamel.yaml import YAML

import torch
import json

from torchvision import transforms
from torch import nn
from tqdm.autonotebook import tqdm
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
os.environ["HYDRA_FULL_ERROR"] = "1"

import time

# use bfloat16 for the entire notebook
torch.autocast(device_type="cpu", dtype=torch.bfloat16).__enter__()

# if torch.cuda.get_device_properties(0).major >= 8:
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

class SAM2AnnotationTool:
    def __init__(
        self,
        ckpt_folder="checkpoints",
        cfg_folder="sam2/configs/sam2.1_training",
        ckpt_file="checkpoint.pt",
        cfg_file="lstm_sam2.1_hiera_t_mobile.yaml",
        search_path="configs/sam2.1_training",
        verbose=True
    ):
        self.search_path = search_path
        self.current_directory = os.getcwd()
        self.sam2_checkpoint = os.path.join(self.current_directory, ckpt_folder, ckpt_file)
        self.model_cfg = os.path.join(self.current_directory, cfg_folder, cfg_file)
        self.model_cfg_search_path = os.path.join(search_path, cfg_file)
        
        self.verbose = verbose
        if self.verbose:
            print("Welcome to SAM 2 Annotation Tool")

    def load_data(
        self,
        img_path,
        mask_path,
        data_type="npy",
        return_vol=True
    ):
        assert data_type in ["npy", "nib"]
        
        if data_type == "nib":
            self.img_vol = nib.load(img_path)
            self.img_vol_data = self.img_vol.get_fdata()
    
            self.mask_vol_data, self.mask_header = nrrd.read(mask_path)
            self.name_value_dict = self.get_label_name_value()
        elif data_type == "npy":
            self.img_vol_data = np.load(img_path)
            self.mask_vol_data = np.load(mask_path)

        self.length = self.img_vol_data.shape[2]

        if self.verbose:
            print("-" * 30)
            print(f"Image path: {img_path}")
            print(f"Mask path: {mask_path}")
            if data_type == "nib":
                print(f"Labels: {self.name_value_dict}")
            print("Load data successfully!")
            print("-" * 30)

        if return_vol:
            return self.img_vol_data, self.mask_vol_data

    def get_label_name_value(self):
        i = 0
        name_value_dict = {}
        while True:
            if f"Segment{i}_LabelValue" in self.mask_header:
                name_value_dict[self.mask_header[f"Segment{i}_Name"]] = int(self.mask_header[f"Segment{i}_LabelValue"])
                i = i + 1
            else:
                break
        return name_value_dict

    def get_label_value_by_name(self, label_name):
        return self.name_value_dict[label_name]

    def get_label_name_by_value(self, label_value):
        return [key for key, value in self.name_value_dict.items() if value == label_value][0]
        
    def preprocess_data_to_sam2_format(self, data_save_directory, img_save_directory, mask_save_directory, volume_name, img_vol_data=None, mask_vol_data=None, verbose=True, phase="train"):
        os.makedirs(os.path.join(img_save_directory, volume_name), exist_ok=True)
        os.makedirs(os.path.join(mask_save_directory, volume_name), exist_ok=True)

        if img_vol_data is not None:
            self.img_vol_data = img_vol_data
            self.mask_vol_data = mask_vol_data

        self.data_save_directory = data_save_directory
        self.img_save_directory = img_save_directory
        self.mask_save_directory = mask_save_directory
        self.txt_path = os.path.join(data_save_directory, f"{phase}.txt")
        
        for i in range(self.length):
            img_ = Image.fromarray(self.img_vol_data[:,:,i].astype("uint8"))
            mask_ = Image.fromarray(self.mask_vol_data[:,:,i].astype("uint8"))

            img_.save(os.path.join(img_save_directory, volume_name, f"{i:05d}.jpg"))
            mask_.save(os.path.join(mask_save_directory, volume_name, f"{i:05d}.png"))

        with open(self.txt_path, "w") as f:
            f.write(volume_name)
            
        if verbose:
            print("-" * 30)
            print(f"Preprocess {i} slices into SAM 2 format.")
            print(f"Img save directory: {img_save_directory}")
            print(f"Mask save directory: {mask_save_directory}")
            print("-" * 30)

    def update_yaml(self, file_path, updates):
        """
        Load a YAML file, update specific parameters, and keep everything else unchanged.
        
        :param file_path: Path to the YAML file
        :param updates: Dictionary of keys to update (e.g., {"dataset.img_folder": "new_path"})
        """
        yaml = YAML()  # ruamel.yaml preserves formatting
        yaml.preserve_quotes = True  # Keep any existing quotes
        yaml.width = 4096  # Prevents unwanted line wrapping
    
        # Load YAML while preserving format
        with open(file_path, "r") as file:
            data = yaml.load(file)
    
        # Apply updates (handle nested keys like "dataset.img_folder")
        for key, new_value in updates.items():
            keys = key.split(".")  # Convert "dataset.img_folder" -> ["dataset", "img_folder"]
            d = data
            for k in keys[:-1]:  # Traverse until last key
                d = d[k]
            d[keys[-1]] = new_value  # Update value
    
        # Save back to the same file (keeping formatting intact)
        with open(file_path, "w") as file:
            yaml.dump(data, file)

    def train(
        self, 
        data_save_directory, 
        img_save_directory, 
        mask_save_directory, 
        volume_name, 
        img_vol_data=None, 
        mask_vol_data=None,
        config=None,
        use_cluster=0,
        partition=None,
        account=None,
        qos=None,
        num_gpus=1,
        num_nodes=None
    ):
        self.preprocess_data_to_sam2_format(
            data_save_directory=data_save_directory,
            img_save_directory=img_save_directory,
            mask_save_directory=mask_save_directory,
            volume_name=volume_name,
            img_vol_data=img_vol_data,
            mask_vol_data=mask_vol_data
        )

        self.update_yaml(
            self.model_cfg,
            {
                "dataset.img_folder": self.img_save_directory,
                "dataset.gt_folder": self.mask_save_directory,
                "dataset.file_list_txt": self.txt_path
                
            }
        )

        if config is None:
            config = self.model_cfg_search_path

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        initialize_config_module(config_module="models/sam2_annotation_tool/training", version_base="1.2")
        sys.argv = ["notebook"]

        # Ensure config is provided
        if config is not None:
            sys.argv.extend(["--config", str(config)])
        
        if use_cluster is not None:
            sys.argv.extend(["--use-cluster", str(use_cluster)])
        
        if partition is not None:
            sys.argv.extend(["--partition", str(partition)])
        
        if account is not None:
            sys.argv.extend(["--account", str(account)])
        
        if qos is not None:
            sys.argv.extend(["--qos", str(qos)])
        
        if num_gpus is not None:
            sys.argv.extend(["--num-gpus", str(num_gpus)])
        
        if num_nodes is not None:
            sys.argv.extend(["--num-nodes", str(num_nodes)])

        parser = ArgumentParser()
        parser.add_argument(
            "-c",
            "--config",
            required=True,
            type=str,
            help="path to config file (e.g. configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml)",
        )
        parser.add_argument(
            "--use-cluster",
            type=int,
            default=None,
            help="whether to launch on a cluster, 0: run locally, 1: run on a cluster",
        )
        parser.add_argument("--partition", type=str, default=None, help="SLURM partition")
        parser.add_argument("--account", type=str, default=None, help="SLURM account")
        parser.add_argument("--qos", type=str, default=None, help="SLURM qos")
        parser.add_argument(
            "--num-gpus", type=int, default=None, help="number of GPUS per node"
        )
        parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes")

        try:
            self.args, self.unknown = parser.parse_known_args()
        except SystemExit as e:
            print(f"Argument parsing failed: {e}")
        self.args.use_cluster = bool(self.args.use_cluster) if self.args.use_cluster is not None else None
        register_omegaconf_resolvers()
        print("self.args", self.args)
        main(self.args)

    def inference(
        self,
        data_save_directory, 
        img_save_directory, 
        mask_save_directory, 
        volume_name,
        ann_frame_idx,
        img_vol_data=None,
        mask_vol_data=None,
        checkpoint_folder="./sam2_logs/configs/sam2.1_training/lstm_sam2.1_hiera_t.yaml/checkpoints",
        checkpoint_name="checkpoint.pt",
        cfg_folder="./configs/sam2.1/",
        cfg_name="lstm_sam2_hiera_t.yaml",
        register_operator=False,
        phase="test"
    ):
        if register_operator:
            register_omegaconf_resolvers()
        print(os.path.join(cfg_folder, cfg_name))
        print(os.path.join(checkpoint_folder, checkpoint_name))
        
        predictor = build_sam2_video_predictor(os.path.join(cfg_folder, cfg_name), os.path.join(checkpoint_folder, checkpoint_name), device="cpu")
        predictor.eval()

        self.preprocess_data_to_sam2_format(
            data_save_directory=data_save_directory,
            img_save_directory=img_save_directory,
            mask_save_directory=mask_save_directory,
            volume_name=volume_name,
            img_vol_data=img_vol_data,
            mask_vol_data=mask_vol_data,
            phase=phase
        )

        with open(os.path.join(data_save_directory, f"{phase}.txt"), "r") as file:
            for n, vol_id in enumerate(file):
                volume_dir = os.path.join(img_save_directory, vol_id)
                mask_dir = os.path.join(mask_save_directory, vol_id)

                # scan all the PNG frame names in this directory
                frame_names = [
                    p for p in os.listdir(volume_dir)
                    if os.path.splitext(p)[-1] in [".jpg"]
                ]
                frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
                
                mask_frame_names = [
                    p for p in os.listdir(mask_dir)
                    if os.path.splitext(p)[-1] in [".png"]
                ]
                mask_frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
                
                num_frame = len(frame_names)

                inference_state = predictor.init_state(video_path=volume_dir)
                
                ann_obj_id = 1
                target_index = 1
                out_mask_logits_gt = np.array(Image.open(os.path.join(mask_dir, mask_frame_names[ann_frame_idx])))
                out_mask_logits_gt = np.array(out_mask_logits_gt == target_index, dtype=np.uint8)
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    mask=out_mask_logits_gt,
                )

                video_segments = {}  # video_segments contains the per-frame segmentation results

                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, recent_n=1):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                    
                if ann_frame_idx > 0:
                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, reverse=True, recent_n=1):
                        video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
                
                pred = []
                for k in sorted(video_segments.keys()):
                    pred.append(np.squeeze(video_segments[k][1]))
                pred = np.stack(pred, axis=-1)

                pred_directory = os.path.join(data_save_directory, "pred")
                os.makedirs(pred_directory, exist_ok=True)

                np.save(os.path.join(pred_directory, volume_name), pred)
                return pred

