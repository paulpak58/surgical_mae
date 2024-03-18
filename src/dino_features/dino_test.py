from transformers import ViTFeatureExtractor, ViTModel, ViTImageProcessor
import torch
import torchvision
import os
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from PIL import Image

# Module imports
from extractor import ViTExtractor
from pca import pca, plot_pca

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='./surgical_images', help='Path to input images')
parser.add_argument('--output_dir', type=str, default='./outputs', help='Path to output images')
parser.add_argument('--threshold', type=float, default=None, help='Threshold for attention maps')
parser.add_argument('--stride', type=int, default=8, help='Stride for attention maps')
parser.add_argument('--model_type', type=str, default='dino_vitb8', help='Model type')
parser.add_argument('--save', action='store_true', help='Save attention maps')
parser.add_argument('--pca', action='store_true', help='Use PCA for dimensionality reduction')
parser.add_argument('--extractor', action='store_true', help='Use PCA for dimensionality reduction')
args = parser.parse_args()

input_dir = args.input_dir
root_output_dir = args.output_dir
save = args.save
threshold = args.threshold
stride = args.stride
model_type = args.model_type
run_pca = args.pca
use_fb_model = args.extractor
patch_size = int(model_type.split('_')[-1][-1])
model_huggingface = f'facebook/{model_type.replace("_", "-")}'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

image_processor = ViTImageProcessor.from_pretrained(model_huggingface)
model = ViTModel.from_pretrained(model_huggingface)
if use_fb_model:
    extractor = ViTExtractor(model_type=model_type, stride=stride, device=device)
else:
    extractor = None

surgical_image_batch = []
img_names = []
for img in os.listdir(input_dir):
    img_names.append(img)
    image = Image.open(input_dir + '/' + img)
    surgical_image_batch.append(image)
s = image_processor(images=surgical_image_batch, return_tensors="pt")
img_batch = s.pixel_values
print(f'Input shape {img_batch.shape}')

'''
for img in os.listdir(input_dir):
    img_names.append(img)
    surgical_image, surgical_image_pil = extractor.preprocess('./surgical_images/' + img, load_size=224)
    surgical_image_batch.append(surgical_image)
surgical_image_batch = torch.cat(surgical_image_batch, dim=0)
print(surgical_image_batch.shape)
print(img_names)
'''

for i,img in enumerate(img_batch):
    # Transform image
    # img = transform(img.unsqueeze(0))
    img = img.unsqueeze(0)
    inputs = {'pixel_values': img}
    # inputs = image_processor(images=img, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    last_hidden_states = outputs.last_hidden_state
    attentions = outputs.attentions[-1]

    if model_type == 'dino_vits8' and extractor:
        s_maps = extractor.extract_saliency_maps(img.to(device))
    else:
        s_maps = None
    if extractor:
        descriptors = extractor.extract_descriptors(img.to(device))
        print(f'Descriptors shape {descriptors.shape}')
    print(f'Last Layer shape {last_hidden_states.shape}')

    img = img.clone().detach().cpu()
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    nh = attentions.shape[1] # number of head
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    if threshold is not None:
        val,idx = torch.sort(attentions)
        val/= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval>(1-threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        th_attn = torch.nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode='nearest')[0].cpu().numpy()
        print(f'Thresholded Attention maps shape {th_attn.shape}\n')


    if s_maps is not None:
        print(f'Initial saliency maps shape {s_maps.shape}')
        s_maps = s_maps.reshape(-1, w_featmap, h_featmap)
        s_maps = torch.nn.functional.interpolate(s_maps.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().detach().numpy()
        print(f'Final saliency maps shape {s_maps.shape}')  
    print(f'Initial Attention maps shape {attentions.shape}')
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = torch.nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().detach().numpy()
    print(f'Final Attention maps shape {attentions.shape}\n')

    # save attentions heatmaps
    if save:
        output_dir = root_output_dir + '/' + img_names[i].split('.')[0]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(output_dir, img_names[i]))
        for j in range(nh):
            fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[j], format='png')
            print(f"{fname} saved.")
        if s_maps is not None:
            plt.imsave(fname=os.path.join(output_dir, "saliency-map.png"), arr=s_maps[0], format='png')

    if run_pca:
        last_components_rgb = True
        save_resized = True
        output_dir = root_output_dir + '/' + img_names[i].split('.')[0]
        images_paths = [os.path.join(input_dir, img) for img in os.listdir(input_dir)]
        pca_per_image = pca(
            images_paths,
            load_size=224,
            layer=11,
            facet='key',
            bin=False,
            stride=stride,
            model_type=model_type,
            n_components=4,         # number of components for PCA
            all_together=True       # Whether to apply PCA on all images
        )
        if save:
            save_dir = output_dir
            for image_path, (pil_image, pca_image) in tqdm(zip(images_paths, pca_per_image)):
                save_prefix = 'pca'
                plot_pca(
                    pil_image,
                    pca_image,
                    str(save_dir),
                    last_components_rgb,
                    save_resized,
                    save_prefix
                )
    
    # Create bitwise map from attention maps
    if threshold is not None:
        if save:
            output_dir = root_output_dir + '/' + img_names[i].split('.')[0]
            for j in range(nh):
                fname = os.path.join(output_dir, f"attn-head{str(j)}-threshold{threshold}.png")
                plt.imsave(fname=fname, arr=th_attn[j], format='png')
                print(f"{fname} saved.")

            print(img_batch[-1].shape)
            mask = th_attn[-1:]==0
            masked_image = (img_batch[-1] * mask)
            print(masked_image.shape)
            test = torchvision.utils.make_grid(img, normalize=True, scale_each=False)*mask
            torchvision.utils.save_image(test, "masked-image.png")
