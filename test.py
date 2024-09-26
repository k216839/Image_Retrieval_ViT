import numpy as np
import torch
from PIL import Image
import argparse
import configs
from sklearn.metrics.pairwise import cosine_similarity
from model import VIT_MSN


arr = np.load('mmt_features/features.npy')
model = VIT_MSN()

# load candidate_ids
with open('mmt_features/candidate_ids.txt', 'r') as f:
    candidate_ids = f.readlines()
candidate_ids = [line.strip() for line in candidate_ids]

# image --> feature
feature = model.get_features([Image.open('query_image/img1.jpg')]).reshape(1, -1)
score = cosine_similarity(feature, arr)
distances, indices = torch.topk(torch.from_numpy(score), k=5)

for (dis, ind) in zip(distances[0], indices[0]):
    print(dis, ind)
    print('Distance: {}, file name: {}'.format(dis, candidate_ids[ind]))