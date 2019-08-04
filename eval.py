import torch.optim as optim
import torch.nn as nn
import torch
import model
import os
from dataset import UCF101DataSet
import numpy as np

test_list = 'list/test_ucf101.list'
batch_size = 12
num_classes = 101
model_dir = 'models'
model_name = 'c3d-new.pth-60000'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def eval():

    model_path = os.path.join(model_dir, model_name)
    c3d = model.C3D(num_classes)
    device = get_default_device()

    if device == torch.device('cpu'):
        c3d.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        c3d.load_state_dict(torch.load(model_path))

    c3d.to(device, non_blocking=True, dtype=torch.float)
    c3d.eval()

    testset = UCF101DataSet(datalist_file=test_list,
                            clip_len=16, crop_size=112, split="testing")
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=10)

    total_predict_label = []
    total_accuracy = []

    for (i, data) in enumerate(testloader, 0):

        # inputs - 12 x 10 x x 16 x 3 x 112 x 112
        outputs = []
        inputs, labels = data['clip'].to(
            device, dtype=torch.float), data['label'].to(device)
        for j,sample in enumerate(inputs):
            print(sample.shape)
            sample_outputs = c3d(sample)
            _, output_idx = sample_outputs.max(1)
            print(output_idx)            
            output = torch.mean(sample_outputs, dim=0)
            outputs.append(output)
        outputs = torch.stack(outputs)
        _, outputs = outputs.max(1)
        total = labels.size(0)
        print(labels)
        print(outputs)
        correct = (outputs == labels).sum().item()
        accuracy = float(correct) / float(total)
        print("iteration %d, accuracy = %g" % (i, accuracy))
        total_predict_label.append(outputs)
        total_accuracy.append(accuracy)

    total_accuracy = np.array(total_accuracy)

    total_predict_label = np.array(total_predict_label)
    print("Final accuracy", np.mean(total_accuracy))


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def main():
    eval()


if __name__ == "__main__":
    main()
