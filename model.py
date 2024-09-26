from transformers import AutoFeatureExtractor, ViTMSNModel
import torch
from PIL import Image
import requests
import configs

class VIT_MSN():
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.image_processing = AutoFeatureExtractor.from_pretrained(configs.MODEL_PATH)
        self.model = ViTMSNModel.from_pretrained(configs.MODEL_PATH)
        if self.device == 'cuda':
            self.model.cuda()
    def get_features(self, images):
        inputs = []
        for image in images:
            pixel_value = self.image_processing(images=image, return_tensors="pt")['pixel_values']
            inputs.append(pixel_value)
        
        inputs = torch.vstack(inputs).to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs).last_hidden_state[:, 0, :]
        return outputs.cpu().numpy()
    

