import torch
from torchvision import transforms
from torchvision.utils import save_image
from models.segment_any_muscle import cfg
from PIL import Image
from models.segment_any_muscle.sam import sam_model_registry
import os

def predict_one_image(img, checkpointLocation, device='cpu'):
    '''
    Input:
        img: image to be predicted. Has to be size [B,3,1024,1024]
    Output:
        pred: prediction. Size is [256,256]
    '''

    # Hyperparameter. Should be fixed
    args = cfg.parse_args()
    args.if_mask_decoder_adapter=True
    args.decoder_adapt_depth=2
    args.if_encoder_adapter = True
    args.encoder_adapter_depths=[0,1,10,11]
    
    # Load model and weights
    sam = sam_model_registry["vit_t"](args,checkpoint=None,num_classes=3, moe=10, k=10)
    states = torch.load(checkpointLocation, map_location=torch.device('cpu'))
    sam.load_state_dict(states)
    model = sam
    model.to(device)
    # Inference
    img_emb, _, _ = model.image_encoder(img, training=False)
    sparse_emb, dense_emb = model.prompt_encoder(points=None,boxes=None,masks=None)
    pred, _, _, _ = model.mask_decoder(
                        image_embeddings=img_emb,
                        image_pe=model.prompt_encoder.get_dense_pe(), 
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb, 
                        multimask_output=True,
                        training=False,)
    pred = pred[0,1,:,:]
    pred = (pred > 0).float()
    return pred


if __name__ == "__main__":
    # Load and pre-process image
    image_path = 'image.png' # Change this to the image path you want to predict
    img = Image.open(image_path).convert('RGB')
    img = transforms.Resize((1024,1024))(img) 
    img = transforms.ToTensor()(img)
    img = (img-img.min())/(img.max()-img.min()+1e-8)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    img = img.unsqueeze(0)

    # Make prediction
    print(torch.cuda.is_available())
    pred = predict_one_image(img)
    print(pred)
    print(pred.shape)
    print(type(pred))
    save_image(pred, 'pred.png')
