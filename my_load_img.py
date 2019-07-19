import torch.optim as optim 
import torch.nn as nn 
import torch
import model
import os
import cv2 
import numpy as np 
from dataset import UCF101DataSet
train_list = 'list/train_ucf101.list'
batch_size = 1

trainset = UCF101DataSet(datalist_file=train_list, clip_len=16, crop_size=112,split="training")
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=2)


for i, data in enumerate(trainloader, 0):
	clip, label = data['clip'], data['label']
	frames = np.transpose(clip[0], (1,0,2,3))
	for frame in frames:
		npimg = frame.numpy()
		cv2.imshow('img',np.transpose(npimg, (1,2,0)))
		k = cv2.waitKey(0)
		if k == 27:
			cv2.destroyAllWindows()

	if i == 0:
		break

