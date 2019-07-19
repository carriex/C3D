import torch.optim as optim 
import torch.nn as nn 
import torch
import model
import os
from dataset import UCF101DataSet

base_lr = 0.003
momentum = 0.9 
batch_size = 1
num_classes = 101
num_epoches = 16
weight_decay = 0.005
train_list = 'list/train_ucf101.list'
model_dir = 'models'
model_name = 'c3d.pth'

def train():

	#create the network 
	c3d = model.C3D(num_classes)

	device = get_default_device()
	#import input data
	trainset = UCF101DataSet(datalist_file=train_list, clip_len=16, crop_size=112,split="training",device=device)
	trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=2)
	

	#c3d.to(device, non_blocking=True,dtype=torch.float)

	#define loss function (Cross Entropy loss)
	criterion = nn.CrossEntropyLoss()


	#define optimizer 
	optimizer = optim.SGD(c3d.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)

	#lr is divided by 10 after every 4 epoches 
	
	scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=4, gamma=0.1)


	for epoch in range(num_epoches):
		
		running_loss = 0.0 

		for i, data in enumerate(trainloader, 0):
			step = epoch * len(trainloader) + i
			inputs, labels = data['clip'], data['label'] 
			print(inputs.device) 
			optimizer.zero_grad()
			
			outputs = c3d(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			nn.utils.clip_grad_value_(c3d.parameters(),1)
			optimizer.step()

			running_loss +=loss.item()
			print('Step %d, loss: %.3f' %(i, loss.item()))

			_, predict_label = outputs.max(1)
			total = labels.size(0)
			correct = (outputs == labels.float()).sum().item()
			accuracy = correct / total
			print("iteration %d, accuracy = %.3f" % (i, accuracy))

			if i % batch_size == (batch_size - 1):
				print('[%d, %5d] loss: %.3f' %
					(epoch + 1, i + 1, running_loss / 30))
				running_loss = 0.0 
			if step % 10000 == 9999:
				torch.save(c3d.state_dict(),os.path.join(model_dir,'%s-%d'%(model_name, step+1)))

		scheduler.step()
	print('Finished Training')

def get_default_device():
	if torch.cuda.is_available():
		return torch.device('cuda')
	else:
		return torch.device('cpu')



def main():
	train()

if __name__ == "__main__":
	main()





