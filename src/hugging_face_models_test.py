import transformers
from transformers import AutoImageProcessor, ViTMAEModel, VideoMAEForPreTraining
import torch
import numpy as np
from PIL import Image
import requests

# Model imports
from get_data import retrieve_dataloaders, get_data


def vitmae():
    ###################
    label_path = '/saiil2/paulpak/surgical_ncp/train_val_paths_labels1.pkl'
    train_dataset_80, train_num_each_80, \
    val_dataset, val_num_each, test_dataset, test_num_each = get_data(label_path)
    # val_dataset, val_num_each, test_dataset, test_num_each = get_data('./train_val_paths_labels1.pkl')


    (train_feature_loader, val_feature_loader, test_feature_loader,
     train_dataset, val_dataset, test_dataset) = retrieve_dataloaders(
        (train_dataset_80),
        (train_num_each_80),
        (val_dataset, test_dataset),
        (val_num_each, test_num_each)
    )

    ###################
    sequence_length = 1
    model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    print('Start training')
    for data in train_feature_loader:
        if torch.cuda.is_available():
            inputs, labels_phase = data[0].to('cuda:0'), data[1].to('cuda:0')
            model = model.to('cuda:0')
        else:
            inputs, labels_phase = data[0], data[1]
        print(f'Inputs shape: {inputs.shape}')
        # inputs = inputs.view(-1, sequence_length, 3, 224, 224)
        outputs = model(inputs)
        print(f'Outputs shape {outputs.last_hidden_state.shape}')

    # image = torch.rand(1, 3, 224, 224)
    # last_hidden_states = outputs.last_hidden_state
    # print('ViTMAE Model', model)

def videomae_pretrain():
    num_frames = 16
    video = list(np.random.randint(0, 256, (num_frames, 3, 224, 224)))

    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")

    pixel_values = image_processor(video, return_tensors="pt").pixel_values
    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
    seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
    bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

    print(f' Input shape, {pixel_values.shape}')
    print(f' Masked shape, {bool_masked_pos.shape}')

    outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
    loss = outputs.loss

def videomae():
    from transformers import VideoMAEModel, VideoMAEConfig

    configuration = VideoMAEConfig()

    model = VideoMAEModel(configuration)
    configuration = model.config
    # print('VideoMAE Model', model)
    
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    # image = torch.rand(1, 3, 224, 224)
    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = inputs['pixel_values'].unsqueeze(1).repeat(1, 16, 1, 1, 1)
    # outputs = model(**inputs)
    outputs = model(inputs)
    print(f'Outputs shape {outputs["last_hidden_state"].shape}')

    print(f'Configurations {configuration}')

    class_head = torch.nn.Linear(configuration.hidden_size, configuration.num_labels)
    outputs = class_head(outputs.last_hidden_state)
    print(f'Final class outputs shape {outputs.shape}')


if __name__ == '__main__':
    # videomae_pretrain()
    # videomae()
    vitmae()