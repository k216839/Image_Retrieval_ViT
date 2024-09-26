import torch
import os
import numpy as np
from PIL import Image
import argparse
import configs
from model import VIT_MSN
from tqdm import tqdm

def run(list_file, output_folder, batch_size=64):
    features = []
    list_names = []
    path_batch = []
    img_batch = []
    feature_batch = []

    num = 0
    for idx, img_path in tqdm(enumerate(list_file), desc='Gen features: '):
        num += 1 
        img = Image.open(img_path).convert('RGB')
        img_name = os.path.basename(img_path)
        path_batch.append(img_name)
        img_batch.append(img)

        if num % batch_size == 0 or idx == len(list_file)-1:
            feature_batch = model.get_features(img_batch).tolist()
            features.extend(feature_batch)
            list_names.extend(path_batch)

            path_batch = []
            img_batch = []

    features = np.array(features, dtype=np.float32)
    with open(f'{output_folder}/candidate_ids.txt', 'w') as f:
        for _name in list_names:
            f.write(_name + '\n')
    np.save(f'{output_folder}/features.npy', features)

            


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='data/oxbuild/images')
    parser.add_argument('-o', '--output', type=str, default='features')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    model = VIT_MSN()
    list_files = os.listdir(args.input)
    list_files = [os.path.join(args.input, _name) for _name in list_files]
    output_folder = args.output

    os.makedirs(output_folder, exist_ok=True)
    run(list_file=list_files, output_folder=output_folder, batch_size=16)
