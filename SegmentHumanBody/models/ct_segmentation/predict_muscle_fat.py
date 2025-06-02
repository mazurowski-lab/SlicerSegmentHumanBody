import torch
import os
import warnings
import argparse
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import join, subfiles
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import numpy as np
import pandas as pd
import shutil
import json

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def segment_vertebrae(input_path, output_path):
    """
    Perform vertebrae segmentation on the given input NIfTI file and save the results.
    """
    # Create output directory path
    output_path = os.path.join(output_path, f"{os.path.basename(input_path).replace('.nii.gz', '')}_vertebrae_segmentation")
    os.makedirs(output_path, exist_ok=True)
    
    # Use specific options to avoid environment issues
    command = (f"TotalSegmentator -i {input_path} -o {output_path} "
              f"--roi_subset vertebrae_T12 vertebrae_L3 vertebrae_L4")
    print(f"Running command: {command}")
    os.system(command)
        
def calculate_2D_metrics(input_nii, segmentation_nii, vertebrae_path):
    """
    Calculate 2D body composition metrics at L3 level
    """
    # Load images
    img = nib.load(input_nii)
    seg = nib.load(segmentation_nii)
    
    # Get spacing information
    spacing = img.header.get_zooms()
    
    # Load L3 vertebra mask
    l3_mask = nib.load(os.path.join(vertebrae_path, "vertebrae_L3.nii.gz"))
    
    # Find the L3 slice
    l3_data = l3_mask.get_fdata()
    l3_slice = np.argmax(np.sum(l3_data, axis=(0,1)))
    
    # Get the image data at L3 level
    img_data = img.get_fdata()[:,:,l3_slice]
    seg_data = seg.get_fdata()[:,:,l3_slice]
    
    # Calculate metrics
    metrics = {
        'filename': os.path.basename(input_nii),
        'muscle_area_mm2': np.sum(seg_data == 1) * spacing[0] * spacing[1],
        'muscle_density_hu': np.mean(img_data[seg_data == 1]),
        'sfat_area_mm2': np.sum(seg_data == 2) * spacing[0] * spacing[1],
        'vfat_area_mm2': np.sum(seg_data == 3) * spacing[0] * spacing[1],
        'mfat_area_mm2': np.sum(seg_data == 4) * spacing[0] * spacing[1],
        'total_fat_area_mm2': np.sum(np.isin(seg_data, [2,3,4])) * spacing[0] * spacing[1],
        'body_area_mm2': np.sum(img_data > -500) * spacing[0] * spacing[1]
    }
    
    return metrics

def calculate_3D_metrics(input_nii, segmentation_nii, vertebrae_path):
    """
    Calculate 3D body composition metrics between T12 and L4
    """
    # Load images
    img = nib.load(input_nii)
    seg = nib.load(segmentation_nii)
    spacing = img.header.get_zooms()
    
    # Load vertebrae masks
    t12_mask = nib.load(os.path.join(vertebrae_path, "vertebrae_T12.nii.gz"))
    l4_mask = nib.load(os.path.join(vertebrae_path, "vertebrae_L4.nii.gz"))
    
    # Find the range between T12 and L4
    t12_data = t12_mask.get_fdata()
    l4_data = l4_mask.get_fdata()
    
    t12_slice = np.argmax(np.sum(t12_data, axis=(0,1)))
    l4_slice = np.argmax(np.sum(l4_data, axis=(0,1)))
    
    slice_range = range(min(t12_slice, l4_slice), max(t12_slice, l4_slice) + 1)
    
    # Get image data
    img_data = img.get_fdata()
    seg_data = seg.get_fdata()
    
    # Calculate volumes and densities
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    
    metrics = {
        'filename': os.path.basename(input_nii),
        'muscle_volume_mm3': np.sum(seg_data[:,:,slice_range] == 1) * voxel_volume,
        'muscle_density_hu': np.mean(img_data[:,:,slice_range][seg_data[:,:,slice_range] == 1]),
        'sfat_volume_mm3': np.sum(seg_data[:,:,slice_range] == 2) * voxel_volume,
        'vfat_volume_mm3': np.sum(seg_data[:,:,slice_range] == 3) * voxel_volume,
        'mfat_volume_mm3': np.sum(seg_data[:,:,slice_range] == 4) * voxel_volume,
        'total_fat_volume_mm3': np.sum(np.isin(seg_data[:,:,slice_range], [2,3,4])) * voxel_volume,
        'body_volume_mm3': np.sum(img_data[:,:,slice_range] > -500) * voxel_volume,
        'body_height_mm': len(slice_range) * spacing[2]
    }
    
    return metrics

def calculate_custom_metrics(input_nii, img_data, segmentation_nii, start_slice, end_slice):
    """
    Calculate body composition metrics for a custom slice range
    
    Args:
        input_nii (str): Path to input NIfTI file
        img_data (numpy.ndarray): Image data array
        segmentation_nii (str): Path to segmentation NIfTI file
        start_slice (int): Starting slice index
        end_slice (int): Ending slice index
    """
    # Load images
    img = nib.load(input_nii)
    seg = nib.load(segmentation_nii)
    spacing = img.header.get_zooms()  # Get spacing from the original NIfTI file
    
    slice_range = range(start_slice, end_slice)
    
    # Get segmentation data
    seg_data = seg.get_fdata()
    
    # Calculate volumes and densities
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    
    metrics = {
        'filename': os.path.basename(input_nii),
        'muscle_volume_mm3': np.sum(seg_data[:,:,slice_range] == 1) * voxel_volume,
        'muscle_density_hu': np.mean(img_data[:,:,slice_range][seg_data[:,:,slice_range] == 1]),
        'sfat_volume_mm3': np.sum(seg_data[:,:,slice_range] == 2) * voxel_volume,
        'vfat_volume_mm3': np.sum(seg_data[:,:,slice_range] == 3) * voxel_volume,
        'mfat_volume_mm3': np.sum(seg_data[:,:,slice_range] == 4) * voxel_volume,
        'total_fat_volume_mm3': np.sum(np.isin(seg_data[:,:,slice_range], [2,3,4])) * voxel_volume,
        'body_volume_mm3': np.sum(img_data[:,:,slice_range] > -500) * voxel_volume,
        'body_height_mm': len(slice_range) * spacing[2]
    }
    
    return metrics

def segmentVolume(input_path, body_composition_type="both", custom_slice_range=None, output_path="results", 
         save_probabilities=False, overwrite=False, checkpoint_type="final", 
         output_body_composition=None):
    """
    Main function to run muscle and fat segmentation and calculate body composition metrics.
    
    Args:
        input_path (str): Path to input file or folder containing .nii.gz files
        body_composition_type (str, optional): Type of body composition analysis.
            Options: '2D', '3D', 'both', or 'None'. Defaults to 'both'
        custom_slice_range (list, optional): List of tuples containing start and end slice indices.
            Defaults to None
        output_path (str, optional): Path for output results. Defaults to 'results'
        save_probabilities (bool, optional): Whether to save probability maps. Defaults to False
        overwrite (bool, optional): Whether to overwrite existing predictions. Defaults to False
        checkpoint_type (str, optional): Type of checkpoint to use ('best' or 'final'). Defaults to 'final'
        output_body_composition (str, optional): Path for body composition metrics. 
            Defaults to output_path if it's a directory, otherwise 'results'
    
    Returns:
        tuple: (segmentation_array, metrics_dict) where:
            - segmentation_array: numpy array of the segmentation
            - metrics_dict: Contains metrics based on body_composition_type
    """
    # Check if all inputs are valid
    print("girdi-0")
    if body_composition_type not in ["2D", "3D", "both", "custom", "None"]:
        raise ValueError("Invalid body_composition_type. Must be one of: '2D', '3D', 'both', or 'None'")
    if custom_slice_range is not None:
        if not isinstance(custom_slice_range, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in custom_slice_range):
            raise ValueError("Invalid custom_slice_range. Must be a list of tuples containing start and end slice indices.")
    if save_probabilities not in [True, False]:
        raise ValueError("Invalid save_probabilities. Must be True or False")
    if overwrite not in [True, False]:
        raise ValueError("Invalid overwrite. Must be True or False")
    if checkpoint_type not in ["best", "final"]:
        raise ValueError("Invalid checkpoint_type. Must be 'best' or 'final'")
    
    # Instantiate the nnUNetPredictor
    print("girdi-1")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    print("girdi-2")
    # Define nnUNet model path correctly
    model_path = input_path.replace("demo/example.nii.gz", "nnUNetTrainer__nnUNetResEncUNetXLPlans__2d")

    # Choose checkpoint file based on user input
    checkpoint_name = 'checkpoint_best.pth' if "best" in checkpoint_type else 'checkpoint_final.pth'
    
    # Initialize the network architecture and load the checkpoint
    predictor.initialize_from_trained_model_folder(
        model_path,
        use_folds=(5,),  # Ensure you are using the correct fold(s)
        checkpoint_name=checkpoint_name,
    )
    print("girdi-3")
    # makedir output path if it doesn't ends with .nii.gz
    if not output_path.endswith(".nii.gz"):
        os.makedirs(output_path, exist_ok=True)
    
    # Check input type
    if os.path.isfile(input_path):
        input_files = [[input_path]]  # Single file in nested list
        if os.path.isdir(output_path):
            print("Output path is a directory")
            output_files = [join(output_path, os.path.basename(f).replace(".nii.gz", "_segmentation.nii.gz")) for f in input_files[0]]
        else:
            output_files = [output_path]  # Single output file
    elif os.path.isdir(input_path):
        input_files = [[f] for f in subfiles(input_path, suffix='.nii.gz', join=True)]
        if not input_files:
            raise FileNotFoundError(f"No NIfTI files found in {input_path}")
        if os.path.isfile(output_path):
            raise ValueError("Output cannot be a file when input is a folder.")
        output_files = [join(output_path, os.path.basename(f).replace(".nii.gz", "_segmentation.nii.gz")) for f in input_files[0]]
    else:
        raise ValueError("Input path must be either a valid file or directory.")
    
    # Perform prediction
    predictor.predict_from_files(
        input_files,
        output_files,
        save_probabilities=save_probabilities,
        overwrite=overwrite,
        num_processes_preprocessing=len(input_files),  # Reduce processes to avoid RAM overuse
        num_processes_segmentation_export=len(output_files),
        folder_with_segs_from_prev_stage=None,
        num_parts=4,
        part_id=0
    )
    print("girdi-4")
    print("Segmentation completed! Results saved in:", output_path)

    # Load segmentation result
    segmentation_array = None
    if len(output_files) > 0:
        segmentation_array = nib.load(output_files[0]).get_fdata()

    print(segmentation_array)
    # Calculate body composition metrics only if requested
    if body_composition_type != "None" and body_composition_type != "custom":
        output_body_composition = output_body_composition if output_body_composition else (
            output_path if os.path.isdir(output_path) else "results"
        )
        os.makedirs(output_body_composition, exist_ok=True)
        
        for input_file in input_files[0]:
                segment_vertebrae(input_file, output_body_composition)

    # Initialize return values
    metrics_2d = None
    metrics_3d = None

    if body_composition_type == "2D" or body_composition_type == "both":
        # Calculate 2D body composition metrics
        for input_file in input_files[0]:
            # Get correct paths for segmentation and vertebrae files
            base_name = os.path.basename(input_file).replace('.nii.gz', '')
            seg_file = os.path.join(output_path, f"{base_name}_segmentation.nii.gz")
            vertebrae_path = os.path.join(output_body_composition, 
                                        f"{base_name}_vertebrae_segmentation")
            
            try:
                if os.path.exists(seg_file) and os.path.exists(vertebrae_path):
                    metrics_2d = calculate_2D_metrics(input_file, seg_file, vertebrae_path)
                else:
                    print(f"Missing files for {input_file}:")
                    print(f"Segmentation file exists: {os.path.exists(seg_file)}")
                    print(f"Vertebrae path exists: {os.path.exists(vertebrae_path)}")
            except Exception as e:
                print(f"Error processing {input_file} for 2D metrics: {str(e)}")
        
    if body_composition_type == "3D" or body_composition_type == "both":
        # Calculate 3D body composition metrics
        for input_file in input_files[0]:
            # Get correct paths for segmentation and vertebrae files
            base_name = os.path.basename(input_file).replace('.nii.gz', '')
            seg_file = os.path.join(output_path, f"{base_name}_segmentation.nii.gz")
            vertebrae_path = os.path.join(output_body_composition,
                                        f"{base_name}_vertebrae_segmentation")
            
            try:
                if os.path.exists(seg_file) and os.path.exists(vertebrae_path):
                    metrics_3d = calculate_3D_metrics(input_file, seg_file, vertebrae_path)
                else:
                    print(f"Missing files for {input_file}:")
                    print(f"Segmentation file exists: {os.path.exists(seg_file)}")
                    print(f"Vertebrae path exists: {os.path.exists(vertebrae_path)}")
            except Exception as e:
                print(f"Error processing {input_file} for 3D metrics: {str(e)}")

    if body_composition_type == "custom":
        for input_file in input_files[0]:
            # Load input image
            img = nib.load(input_file)
            img_data = img.get_fdata()
            
            if custom_slice_range[0] is not None:
                start_slice = custom_slice_range[0][0]
                end_slice = custom_slice_range[0][1]
                # check if start_slice and end_slice are within the image dimensions
                if start_slice < 0 or end_slice > img_data.shape[2] or start_slice >= end_slice:
                    raise ValueError("Custom slice range is out of image dimensions")
            else:
                start_slice = 0
                end_slice = img_data.shape[2]
                
            base_name = os.path.basename(input_file).replace('.nii.gz', '')
            seg_file = os.path.join(output_path, f"{base_name}_segmentation.nii.gz")
            try:
                if os.path.exists(seg_file):
                    metrics_custom = calculate_custom_metrics(input_file, img_data, seg_file, start_slice, end_slice)
                else:
                    print(f"Missing files for {input_file}:")
                    print(f"Segmentation file exists: {os.path.exists(seg_file)}")
            except Exception as e:
                print(f"Error processing {input_file} for custom metrics: {str(e)}")

    # Format metrics result based on type
    metrics_result = None
    if body_composition_type != "None":
        if body_composition_type == "2D":
            metrics_result = {"2D": metrics_2d}
        elif body_composition_type == "3D":
            metrics_result = {"3D": metrics_3d}
        elif body_composition_type == "both":
            metrics_result = {
                "2D": metrics_2d,
                "3D": metrics_3d
            }
        elif body_composition_type == "custom":
            metrics_result = {"custom": metrics_custom}

    # Clean up temporary files
    if os.path.exists(output_path):
        pass
        #shutil.rmtree(output_path)
    if output_body_composition and os.path.exists(output_body_composition):
        pass
        #shutil.rmtree(output_body_composition)

    with open(output_path + "/metrics.txt", "w") as file:
        json.dump(metrics_result, file, indent=4)


    return segmentation_array, metrics_result

if __name__ == "__main__":
    input_path = "C:/Users/zafry/SlicerSegmentHumanBody/SegmentHumanBody/Resources/UI/../../models/ct_segmentation/demo/example.nii.gz"
    output_path = "C:/Users/zafry/SlicerSegmentHumanBody/SegmentHumanBody/Resources/UI/../../models/ct_segmentation/results"
    body_composition_type = "custom"
    custom_slice_range = [(0,10)]
    
    # Run main function and print results
    segmentation, metrics = segmentVolume(input_path, body_composition_type, custom_slice_range, output_path=output_path)
    
    print("Segmentation array shape:", segmentation.shape)
    print("\nBody Composition Metrics:")
    print(metrics)