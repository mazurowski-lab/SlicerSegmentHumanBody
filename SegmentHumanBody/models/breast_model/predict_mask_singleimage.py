import torch
import numpy as np
from models.breast_model.utils import VNet

def zscore_image(image_array):
    image_array = (image_array - np.mean(image_array)) / np.std(image_array)
    return image_array

def normalize_image(image_array, min_cutoff = 0.001, max_cutoff = 0.001):
    sorted_array = np.sort(image_array.flatten())

    # Find %ile index and get values
    min_index = int(len(sorted_array) * min_cutoff)
    min_intensity = sorted_array[min_index]

    max_index = int(len(sorted_array) * min_cutoff) * -1
    max_intensity = sorted_array[max_index]

    # Normalize image and cutoff values
    image_array = (image_array - min_intensity) / (max_intensity - min_intensity)
    image_array[image_array < 0.0] = 0.0
    image_array[image_array > 1.0] = 1.0 
    return image_array


def breast_model_predict_volume(image_numpy, weight_path, device='cpu'):
    '''
    Input:
        image_numpy: input image with format numpy.array. Expect input shape: H*W*D
        weight_path: the path to pre-trained weights
        device: cpu or cuda
    Output:
        out: predictions with shape: 3*H*W*D
        Meaning of 3 channel:
            1 - Background
            2 - Vessel prediction
            3 - Tissue prediction
    '''
    image_numpy = zscore_image(normalize_image(image_numpy))
    # Hyperparameter you can adjust
    x_iter = y_iter = 8
    z_iter = 3
    # Hyperparamter you shouldn't adjust
    size = 96

    # Define model and load pre-trained weights
    model = VNet(outChans=3).to(device)
    weights = torch.load(weight_path)
    new_weights = {}
    for name in weights:
        new_name = name.replace('module.', '')
        new_weights[new_name] = weights[name]
    model.load_state_dict(new_weights)

    # Load image
    image = torch.tensor(image_numpy).float().to(device)
    image = image.unsqueeze(0).unsqueeze(0)
    
    # Define placeholder for mask
    image_shape = list(image.shape)
    image_shape[1] = 3
    pred_mask  = torch.zeros(image_shape).float().to(device)
    pred_count = torch.zeros(image_shape).float().to(device)
    
    # Determine step size of sliding window
    x_step = (image.shape[2]-size) // (x_iter-1)
    y_step = (image.shape[3]-size) // (y_iter-1)
    z_step = (image.shape[4]-size) // (z_iter-1)
    
    # Real inference
    count = 0
    with torch.no_grad():
        for x in range(x_iter):
            for y in range(y_iter):
                for z in range(z_iter):

                    curr_patch = image[:,:, x*x_step:x*x_step+size,
                                            y*y_step:y*y_step+size,
                                            z*z_step:z*z_step+size]
                    # Need to call module.forward since the other process is waiting
                    curr_pred = model(curr_patch)
                    curr_pred = torch.softmax(curr_pred, dim=1)

                    pred_mask[:,:, x*x_step:x*x_step+size,
                                   y*y_step:y*y_step+size,
                                   z*z_step:z*z_step+size] += curr_pred
                    pred_count[:,:,x*x_step:x*x_step+size,
                                   y*y_step:y*y_step+size,
                                   z*z_step:z*z_step+size] += 1
                    count += 1
                    

    # Average prediction over volumes
    pred_mask = torch.div(pred_mask, pred_count)
    pred_binary = (pred_mask > 0.5).float()
    
    pred_numpy = pred_binary.cpu().numpy().squeeze()
    return pred_numpy


if __name__ == '__main__':
    '''
    NOTE: if you read a raw 3D image, say from nii.gz, it is VERY important to do the following normalization
    
    # Assume image is also numpy.array
    image = zscore_image(normalize_image(image))

    def zscore_image(image_array):
       image_array = (image_array - np.mean(image_array)) / np.std(image_array)
       return image_array
    
    def normalize_image(image_array, min_cutoff = 0.001, max_cutoff = 0.001):
       sorted_array = np.sort(image_array.flatten())
    
       # Find %ile index and get values
       min_index = int(len(sorted_array) * min_cutoff)
       min_intensity = sorted_array[min_index]
    
       max_index = int(len(sorted_array) * min_cutoff) * -1
       max_intensity = sorted_array[max_index]

       # Normalize image and cutoff values
       image_array = (image_array - min_intensity) / (max_intensity - min_intensity)
       image_array[image_array < 0.0] = 0.0
       image_array[image_array > 1.0] = 1.0 
       return image_array
    '''

    input_path = '../../breast_mri_data/prior_work_1k/mri_data_pre_numpy/Breast_MRI_214.npy'
    input_data = np.load(input_path)

    weight_path = 'vnet_dv_6958_8879.pth'

    #out = breast_model_predict_volume(input_data, weight_path, 'cpu')
