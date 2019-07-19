from __future__ import print_function, division 
import os
import cv2 
import numpy as np 
import torch 
import torchvision
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader 


class UCF101DataSet(Dataset):
	def __init__(self, datalist_file, clip_len, crop_size,split,device,transform=None):
		'''
		datalist_file contains the list of frame information e.g. 
		/Users/carriex/git/supervised_training/data/v_ApplyEyeMakeup_g01_c01/ 1 0
		The shape of the return clip is 3 x clip_len x crop_size x crop_size
		'''
		self.datalist = self.get_datalist(datalist_file)  
		self.transform = transform
		self.clip_len = clip_len
		self.crop_size = crop_size
		self.split = split
		self.device = device

	def __len__(self):
		return len(self.datalist)

	def __getitem__(self, idx):
		data = self.datalist[idx]
		frame_dir, start_frame, label = data[0], int(data[1]), data[2]
		np_mean = np.load("ucf101_volume_mean_official.npy") 
		clip = self.load_frames(frame_dir,start_frame)
		clip = self.normalize(clip,np_mean)
		clip = self.crop(clip)
		clip = self.random_flip(clip)
		clip,label = self.to_tensor(clip,label)
		sample = {'clip':clip, 'label':label}

		if self.transform:
			sample = self.transform(sample)

		return sample 

	def get_datalist(self,datalist_file):
		datalist = list(open(datalist_file, 'r'))
		for i in range(len(datalist)):
			datalist[i] = datalist[i].strip('\n').split()

		return datalist

	def load_frames(self,frame_dir,start_frame):
		clip = []
		for i in range(self.clip_len):
			frame_path = os.path.join(frame_dir, "frame" + "{:06}.jpg".format(start_frame+i))
			frame_origin = cv2.imread(frame_path)
			frame_resize = cv2.resize(frame_origin, (171, 128))
			frame = np.array(frame_resize).astype(np.uint8)
			clip.append(frame_resize)
		clip = np.array(clip).astype(np.uint8)
		return clip

	def crop(self,clip):
		# clip - 16 x 128 x 171 x 3 
		# frame - 128 x 171 x 3 
		x = clip[0].shape[0]
		y = clip[0].shape[1]
		crop_size = self.crop_size
		crop_clip = []
		if self.split == "training":
			crop_x = np.random.randint(0, x - crop_size)
			crop_y = np.random.randint(0, y - crop_size)
		else:
			crop_x = (x - crop_size) // 2 
			crop_y = (y - crop_size) // 2
		for frame in clip:
			frame = frame[crop_x:(crop_x + crop_size), crop_y:(crop_y+crop_size),:]
			crop_clip.append(frame)
		return np.array(crop_clip).astype(np.uint8)


	def normalize(self,clip,np_mean):
		norm_clip = []
		for i in range(len(clip)):
			norm_clip.append(clip[i] - np_mean[i])
		return np.array(norm_clip).astype(np.uint8)

	def random_flip(self,clip):
		flip_clip = []
		if self.split == "training":
			for i in range(len(clip)):
				mirror = np.random.randint(0,1)
				if mirror == 0:
					flip_clip.append(cv2.flip(clip[i], 1))
				else:
					flip_clip.append(clip[i])
		else:
			flip_clip = clip 

		return np.array(flip_clip).astype(np.uint8)


	def to_tensor(self,clip,label):
		return torch.from_numpy(clip.transpose((3,0,1,2))).to(self.device),torch.from_numpy(np.array(label).astype(np.int64)).to(self.device)

	def show_image(self,image):
		print(image.dtype)
		cv2.imshow('img',image)
		k = cv2.waitKey(0)
		if k == 27:
			cv2.destroyAllWindows()















