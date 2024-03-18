from transformers import ViTMAEForPreTraining

def download_pretrained(model_name):
    model = ViTMAEForPreTraining.from_pretrained(model_name)
    model.save_pretrained('./offline_saved_weights')

if __name__=='__main__':
    download_pretrained('facebook/vit-mae-base')