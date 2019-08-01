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
model_name = 'c3d-data.pth-60000'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def eval():
	
	model_path = os.path.join(model_dir,model_name)
	c3d = model.C3D(num_classes)
	device = get_default_device()

	if device == torch.device('cpu'):
		c3d.load_state_dict(torch.load(model_path,map_location='cpu'))
	else:
		c3d.load_state_dict(torch.load(model_path))
	
	c3d.to(device,non_blocking=True,dtype=torch.float)
	c3d.eval()
	
	for name, param in c3d.named_parameters():
		print (name, param)

	testset = UCF101DataSet(datalist_file=test_list, clip_len=16, crop_size=112,split="testing")
	testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=4) 

	total_predict_label = []
	total_accuracy = [] 

	for (i, data) in enumerate(testloader, 0):
		inputs, labels = data['clip'].to(device,dtype=torch.float), data['label'].to(device)
		outputs = c3d(inputs)
		#outputs = nn.Softmax(dim=1)(outputs)
		_, outputs = outputs.max(1)
		total = labels.size(0)
		correct = (outputs == labels).sum().item()
		accuracy = float(correct) / float(total)
		print("iteration %d, accuracy = %g" % (i, accuracy))
		total_predict_label.append(outputs)
		total_accuracy.append(accuracy)

	total_accuracy = np.array(total_accuracy)

	total_predict_label = np.array(total_predict_label)
	#np.savetxt('results/total_accuracy.txt', total_accuracy, fmt = "%.6f")
	#np.save('results/total_accuray.npy', total_accuracy)
	#np.savetxt('results/predict_label_total.txt', total_predict_label, fmt="%d")
	#np.save('results/predict_label_total.npy', total_predict_label)
	print(model_name)
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



