import torch
import os
import warnings
import argparse
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import join, subfiles
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from totalsegmentator.python_api import totalsegmentator 

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def segment_vertebrae(input_path, output_path):
    """
    Perform vertebrae segmentation on the given input NIfTI file and save the results.
    """
    img = nib.load(input_path)

    # Define vertebrae segmentation output folder
    vertebrae_output_path = join(output_path, f"{input_path.replace(".nii.gz", "")}vertebrae_segmentation")
    os.makedirs(vertebrae_output_path, exist_ok=True)

    # Run TotalSegmentator for vertebrae segmentation
    totalsegmentator(img, vertebrae_output_path)
    print(f"Vertebrae segmentation completed! Results saved in: {vertebrae_output_path}")

def main(args):
    # Instantiate the nnUNetPredictor
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
    
    # Define nnUNet model path correctly
    model_path = 'nnUNetTrainer__nnUNetResEncUNetXLPlans__2d'

    # Choose checkpoint file based on user input
    checkpoint_name = 'checkpoint_best.pth' if "best" in args.checkpoint_type else 'checkpoint_final.pth'
    
    # Initialize the network architecture and load the checkpoint
    predictor.initialize_from_trained_model_folder(
        model_path,
        use_folds=(5,),  # Ensure you are using the correct fold(s)
        checkpoint_name=checkpoint_name,
    )
    
    # Define input and output paths from arguments
    input_path = args.input
    output_path = args.output

    # Check input type
    if os.path.isfile(input_path):
        input_files = [[input_path]]  # Single file in nested list
        if os.path.isdir(output_path):
            output_files = [join(output_path, os.path.basename(f).replace(".nii.gz", "_segmentation.nii.gz")) for f in input_files[0]]
        else:
            output_files = [output_path]  # Single output file
    elif os.path.isdir(input_path):
        input_files = [[f] for f in subfiles(input_path, suffix='.nii.gz', join=True)]
        if not input_files:
            raise FileNotFoundError(f"No NIfTI files found in {input_path}")
        if os.path.isfile(output_path):
            raise ValueError("Output cannot be a file when input is a folder.")
        os.makedirs(output_path, exist_ok=True)
        output_files = [join(output_path, os.path.basename(f).replace(".nii.gz", "_segmentation.nii.gz")) for f in input_files[0]]
    else:
        raise ValueError("Input path must be either a valid file or directory.")
    
    # Perform prediction
    # predictor.predict_from_files(
    #     input_files,
    #     output_files,
    #     save_probabilities=args.save_probabilities,
    #     overwrite=args.overwrite,
    #     num_processes_preprocessing=len(input_files),  # Reduce processes to avoid RAM overuse
    #     num_processes_segmentation_export=len(output_files),
    #     folder_with_segs_from_prev_stage=None,
    #     num_parts=4,
    #     part_id=0
    # )

    # print("Segmentation completed! Results saved in:", output_path)

    # Calculate body composition metrics
    output_body_composition = args.output_body_composition if args.output_body_composition else (output_path if os.path.isdir(output_path) else "results")

    if args.body_composition_type != "None":
        for input_file in input_files[0]:
            segment_vertebrae(input_file, output_body_composition)

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run nnUNet segmentation on muscle and fat MRI/CT images.")

    # Define command-line arguments
    parser.add_argument("--input", type=str, default="demo", help="Path to input file or folder containing .nii.gz files (default: demo)")
    parser.add_argument("--output", type=str, default="results", help="Path to output file or folder for segmentation results (default: results)")
    parser.add_argument("--save_probabilities", type=bool, default=False, help="Whether to save probability maps (default: False)")
    parser.add_argument("--overwrite", type=bool, default=False, help="Whether to overwrite existing predictions (default: False)")
    parser.add_argument("--checkpoint_type", type=str, default="final", help="Specify 'best' to use checkpoint_best.pth or 'final' to use checkpoint_final.pth (default: final)")
    parser.add_argument("--body_composition_type", type=str, default="both", help="Specify '2D' for 2D body composition, '3D' for 3D body composition, 'both' for both, 'None' for no body composition (default: both)")
    parser.add_argument("--output_body_composition", type=str, help="Path to output body composition metric file. If --output is a folder, this will be set to the same folder. Otherwise, it defaults to 'results'.""Path to output body composition metric file")
    # Parse arguments
    args = parser.parse_args()

    # Run main function
    main(args)