from __future__ import print_function, division
import os
import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import time


class UCF101DataSet(Dataset):
    def __init__(self, datalist_file, clip_len, crop_size, split, test_sample_number=10):
        '''
        datalist_file contains the list of frame information e.g. 
        /Users/carriex/git/supervised_training/data/v_ApplyEyeMakeup_g01_c01/ 1 0
        The shape of the return clip is 3 x clip_len x crop_size x crop_size
        '''
        self.datalist = self.get_datalist(datalist_file)
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.split = split
        self.clips_with_label = self.get_clip_list(datalist_file)
        self.test_sample_number = test_sample_number

    def __len__(self):
        if self.split == "training":
            return len(self.datalist)
        else:
            return sum([len(clips) for clips in clip_per_label])

    def __getitem__(self, idx):

        if self.split == "training":
            data = self.datalist[idx]
            frame_dir, start_frame, label = data[0], int(data[1]), data[2]
            np_mean = np.load("ucf101_volume_mean_official.npy")
            clip = self.load_frames(frame_dir, start_frame)
            clip = self.crop(clip)
            clip = self.random_flip(clip)
            clip, label = self.to_tensor(clip, label)
        else:
            clip_with_label = self.clips_with_label[idx]
            for frame_idx in range(self.test_sample_number):
                clip = self.load_frames(clips_in_label[int(clip_idx)], frame_idx*self.clip_len+1)
                clip = self.crop(clip)
                sample_clips.append(clip)

            # sample clips - 10 x 16 x 3 x 112 x 112
            clip, label = self.to_tensor(np.array(sample_clips), int(clip_idx))

        sample = {'clip': clip, 'label': label}

        return sample

    def get_datalist(self, datalist_file):
        datalist = list(open(datalist_file, 'r'))
        for i in range(len(datalist)):
            datalist[i] = datalist[i].strip('\n').split()

        return datalist

    def get_clip_list(self, cliplist_file):
        """Args: /data2/UCF101/ucf101_jpegs_256/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01/  0"""
        """
		clip_per_class[i] = [list of path for video clip]
		"""
        datalist = list(open(datalist_file, 'r'))
        clips_with_label = []
        for data in datalist:
            path, label = ata.strip('\n').split(' ')[0], int(data.strip('\n').split(' ')[1])
            clips_with_label.append({'path':path, 'label':label})
        return clips_with_label

    def load_frames(self, frame_dir, start_frame):
        clip = []
        for i in range(self.clip_len):
            frame_path = os.path.join(
                frame_dir, "frame" + "{:06}.jpg".format(start_frame+i))
            frame_origin = cv2.imread(frame_path)
            frame_resize = cv2.resize(frame_origin, (171, 128))
            clip.append(frame_resize)
        clip = np.array(clip)
        return clip

    def crop(self, clip):
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
            frame = frame[crop_x:(crop_x + crop_size),
                          crop_y:(crop_y+crop_size), :]
            crop_clip.append(frame)
        return np.array(crop_clip).astype(np.uint8)

    def normalize(self, clip, np_mean):
        norm_clip = []
        for i in range(len(clip)):
            norm_clip.append(clip[i] - np_mean[i])
        return np.array(norm_clip).astype(np.uint8)

    def random_flip(self, clip):
        flip_clip = []
        mirror = np.random.randint(0, 2)
        if self.split == "training":
            for i in range(len(clip)):
                if mirror == 0:
                    flip_clip.append(cv2.flip(clip[i], 1))
                else:
                    flip_clip.append(clip[i])
        else:
            flip_clip = clip

        return np.array(flip_clip).astype(np.uint8)

    def to_tensor(self, clip, label):
        if self.split == "training":
            # 3 x 16 x 112 x 112
            return torch.from_numpy(clip.transpose((3, 0, 1, 2))), torch.from_numpy(np.array(label).astype(np.int64))
        else:
            # 10 x 3 x 16 x 112 x 112
            return torch.from_numpy(clip.transpose((0, 4, 1, 2, 3))), torch.tensor(label)


def show_batch(clips):

    batch_size = clips.shape[0]
    for i in range(batch_size):
        video = clips[i].numpy().transpose(1, 2, 3, 0)
        for frame in video:
            cv2.imshow('img', frame)
            k = cv2.waitKey(0)
            if k == 27:
                cv2.destroyAllWindows()



