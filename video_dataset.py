from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from video_transforms import RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor
from torchvision import transforms
import torch

new_height = 128
new_width = 171
crop_size= 112


class ucf101(Dataset):

    def __init__(self, rgb_list, transform=None):

        rgb_lines = open(rgb_list, 'r')
        self.rgb_lines = list(rgb_lines)
        self.transform = transform


    def __len__(self):
        return len(self.rgb_lines)

    def __getitem__(self, index):


        rgb_line = self.rgb_lines[index].strip('\n').split()
        img_dir = rgb_line[0]

        start_frame = int(rgb_line[1])
        tmp_label = int(rgb_line[2])

        video_clip = []

        for i in range(16):

            cur_img_path = os.path.join(img_dir,
                                        "frame" + "{:06}.jpg".format(start_frame + i))

            img = cv2.imread(cur_img_path)
            img = cv2.resize(img, (171, 128))

            video_clip.append(img)

        video_clip = np.array(video_clip)




        sample = {'clip': video_clip, 'label': tmp_label}


        if self.transform:
            sample = self.transform(sample)

        return sample


def show_batch(video_clips):

    batch_size = video_clips.shape[0]


    for i in range(batch_size):
        cur_video = video_clips[i]


        cur_video = cur_video.numpy().transpose(1, 2, 3, 0)

        for img in cur_video:
            cv2.imshow("img", img)
            k = cv2.waitKey(0)
            if k == 27:
                cv2.destroyAllWindows()


if __name__ == '__main__':

    rgb_list_path  = 'list/rgb_train_ucf101.list'


    ucf101_dataset = ucf101(rgb_list=rgb_list_path, transform=None)

    

    sample, label = ucf101_dataset[100]['clip'], ucf101_dataset[100]['label']
    


    trans_ucf101_dataset = ucf101(rgb_list=rgb_list_path, transform=transforms.Compose([RandomCrop(112),
                                                                                        RandomHorizontalFlip(0.5),
                                                                                       ToTensor()]))



    dataloader = DataLoader(trans_ucf101_dataset, batch_size=1, shuffle=True, num_workers=1)


    import time
    s_time = time.time()

    for i_batch, sample_batched in enumerate(dataloader):

        video_clips = sample_batched['clip']

        labels = sample_batched['label']
        print(labels)
        show_batch(video_clips)
        print("OK")

        if i_batch == 0:
            break



