import os
import sys
import numpy as np
import pickle

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

from PIL import Image
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.color import gray2rgb, rgba2rgb
import skimage.io

import json

import matplotlib.pyplot as plt
import cv2

def recommend_similarity_item(item):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  ### hyper parameter
  k     = 3    ### 출력할 유사한 item의 갯수
  rows  = 1    ### 출력할 item image의 행
  cols  = k    ### 출력할 item image의 열
  space = 1    ### 출력할 item이 놓일 위치

  ### 경로 설정
  root_path    = "../../experiment/"    ### file 경로의 공통된 root_path      

  feature_path = "data/polyvore/dataset/"    ### feature vector를 저장한 파일의 위치

  train_json = 'data/polyvore/jsons/train_no_dup.json'    ### train data의 위치
  valid_json = 'data/polyvore/jsons/valid_no_dup.json'    ### validation data의 위치
  test_json  = 'data/polyvore/jsons/test_no_dup.json'     ### test data의 위치

  image_folder_path = 'data/polyvore/images/'    ### dataset의 image들이 저장된 위치

  ### item 간의 유사도를 계산하기 위한 cosine similarity 함수
  def cos_sim(A, B):
    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

  ### 미리 학습한 training set에 있는 item들의 feature vector 불러오기
  with open(root_path+feature_path+'imgs_featdict_train.pkl', 'rb') as train_feat:
    train_data = pickle.load(train_feat)

  ### 미리 학습한 validation set에 있는 item들의 feature vector 불러오기
  with open(root_path+feature_path+'imgs_featdict_valid.pkl', 'rb') as valid_feat:
    valid_data = pickle.load(valid_feat)
    
  ### 미리 학습한 test set에 있는 item들의 feature vector 불러오기
  with open(root_path+feature_path+'imgs_featdict_test.pkl', 'rb') as test_feat:
    test_data = pickle.load(test_feat)

  save_feat = []     ### save_id: 각 item을 구별할 수 있는 id만 저장하기 위한 list
  save_id   = []     ### save_feat: 각 item의 feature vector만 저장하기 위한 list

  ### train data에 대하여...
  for key, value in train_data.items():
    save_feat.append(value)
    save_id.append(key)

  ### validation data에 대하여...
  for key, value in valid_data.items():
    save_feat.append(value)
    save_id.append(key)

  ### test data에 대하여...
  for key, value in test_data.items():
    save_feat.append(value)
    save_id.append(key)

  print('nums of total item: {}\n'.format(len(save_id)))

  ###############################################
  ### 모든 item들의 id와 feature vector를 저장 ###
  ###############################################
  data_dict = {}    ### data_dict은 전체 data에 있는 item에 대하여
                    ### key: item_id, value: feature vector를 갖는다.
  
  for i in range(len(save_id)):
    data_dict[save_id[i]] = save_feat[i]
  ###############################################

  ###############################################################################
  ###                         user가 등록한 이미지 출력                        ###
  ###############################################################################
  ### 임의의 user item 등록
  item_path = item

  ### 등록 이미지 출력
  user_img = cv2.imread(item_path)
  user_img = cv2.resize(user_img, dsize=(256,256), interpolation=cv2.INTER_AREA)
  user_img = plt.imshow(cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB))
  plt.axis("off")
  ###############################################################################

  print("Start loading pretrained resnet50 model...")
  model = models.resnet50(pretrained=True)
  model = nn.Sequential(*list(model.children())[:-1])   
  model = model.to(device)
  model.eval()
  print("Finish loading pretrained restnet50 model!!!\n")

  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

  transform = transforms.Compose([
              transforms.ToPILImage(),
              transforms.Resize(256), transforms.CenterCrop(224),
              transforms.ToTensor(), normalize
              ])

  ### resnet50을 통해 user image의 feature vector를 추출하는 함수
  def process_image(im):
    im  = transform(im)
    im  = im.unsqueeze_(0)
    im  = im.to(device)
    out = model(im)
    return out.squeeze()

  print("Start extract user items feature vector")
  with torch.no_grad():
    im = skimage.io.imread(item_path)
    if len(im.shape) == 2:
      im = gray2rgb(im)
    if im.shape[2] == 4:
      im = rgba2rgb(im)
    
    im = resize(im, (256,256))
    im = img_as_ubyte(im)
    
    feats = process_image(im).cpu().numpy()
  print("Finish extract user items feature vector\n")

  res_list = []    ### res_list에는 user의 item과 data의 item의 유사도를 저장하는 list  
  res_dict = {}    ### key: polyvore item, value: similarity score

  print("Start calculating similarity scores...")
  for i in range(len(save_feat)):
    res = cos_sim(feats,save_feat[i])
    res_list.append(res)
    res_dict[save_id[i]] = res_list[i]
  print("Finish calculating simiarity scores!!!\n")
  
  print("Start sorting similarity score...")
  sort_list = sorted(res_dict.items(), key=lambda item:item[1], reverse=True)
  print("Finish sorting similarity score!!!\n")
  
  best_sim_list = []    ### k개의 가장 유사성이 높은 item을 저장하는 list
  
  for i in range (k):
    best_sim_list.append(sort_list[i][0])

  ######################################################
  ###                json data load                  ###
  ######################################################
  ### train.json에 대하여...
  with open(root_path+train_json, "r") as train_file:
    train_in_outfit_id = json.load(train_file)

  ### valid.json에 대하여...
  with open(root_path+valid_json, "r") as valid_file:
    valid_in_outfit_id = json.load(valid_file)

  ### test.json에 대하여...
  with open(root_path+test_json, "r") as test_file:
    test_in_outfit_id = json.load(test_file)
  #######################################################

  outfit2item  = {}    ### outfit2item은 outfit을 조합하는 item을 저장한다.
                       ### key: outfit_id, value: outfit을 구성하는 item id들
  outfit2index = {}    ### outf2index는 outfit을 조합하는 index를 저장한다.
                       ### key: outfit_id, value: outfit을 구성하는 index들

  ################################################################################
  ###         각 outfit 마다 item_id와 index를 딕셔너리 형태로 각각 저장         ###
  ################################################################################
  ### 17,316개의 outfit을 갖고 있는 train set에ㅔ 대하여...
  for i in range(len(train_in_outfit_id)):    ### len(train_in_outfit_id: 17316)
    outfit_id = train_in_outfit_id[i]["set_id"]
    
    item_index = []    ### train data에서 outfit을 구성하는 index를 저장하는 list
    item_id    = []    ### train data에서 outfit을 구성하는 item id를 저장하는 list
    
    for j in range(len(train_in_outfit_id[i]["items"])):
      index = train_in_outfit_id[i]["items"][j]["index"]
      item_index.append(index)

      _, each_item_id = train_in_outfit_id[i]["items"][j]["image"].split('id=')
      item_id.append(each_item_id)
        
    outfit2index[outfit_id] = item_index
    outfit2item[outfit_id]  = item_id

  ### 1,497개의 outfit을 갖고 있는 validation set에 대하여...
  for i in range(len(valid_in_outfit_id)):    ### len(train_in_outfit_id: 17316)
    outfit_id = valid_in_outfit_id[i]["set_id"]
    
    item_index = []    ### validation data에서 outfit을 구성하는 index를 저장하는 list
    item_id    = []    ### validation data에서 outfit을 구성하는 item id를 저장하는 list  

    for j in range(len(valid_in_outfit_id[i]["items"])):
      index = valid_in_outfit_id[i]["items"][j]["index"]
      item_index.append(index)

      _, each_item_id = valid_in_outfit_id[i]["items"][j]["image"].split('id=')
      item_id.append(each_item_id)
      
    outfit2index[outfit_id] = item_index
    outfit2item[outfit_id]  = item_id
      
  ### 3,076개의 outfit을 갖고 있는 test set에 대하여....
  for i in range(len(test_in_outfit_id)):    ### len(train_in_outfit_id: 17316)
    outfit_id = test_in_outfit_id[i]["set_id"]
    
    item_index = []    ### test data에서 outfit을 구성하는 index를 저장하는 list
    item_id    = []    ### test data에서 outfit을 구성하는 item id를 저장하는 list
  
    for j in range(len(test_in_outfit_id[i]["items"])):
      index = test_in_outfit_id[i]["items"][j]["index"]
      item_index.append(index)

      _, each_item_id = test_in_outfit_id[i]["items"][j]["image"].split('id=')
      item_id.append(each_item_id)
        
    outfit2index[outfit_id] = item_index
    outfit2item[outfit_id] = item_id
  ################################################################################

  outfit_id = []    ### outfit의 id를 저장하는 list
  index_id  = []    ### outfit의 index 갯수를 저장하는 list

  for i in range(k):
    for key in outfit2item.keys():
      for j in range(len(outfit2item[key])):
        if best_sim_list[i] == outfit2item[key][j]:
          outfit_id.append(str(key))
          index_id.append(str(j+1))
    
  sim_save_file = []    ### 유사한 item의 image 파일 경로를 저장하는 list
  for i in range(k):
      sim_save_file.append(image_folder_path+outfit_id[i]+'/'+index_id[i]+'.jpg')

  ### http://blog.daum.net/geoscience/1263 참고
  fig = plt.figure()

  for i in range(k):
    img = cv2.imread(root_path+sim_save_file[i])
    img = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_AREA)
    ax = fig.add_subplot(rows,cols,space)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    space += 1
    ax.axis("off")
  plt.show()

if __name__ == '__main__':
  img_path = './sample_img/7.jpg'

  recommend_similarity_item(img_path)