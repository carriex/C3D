import torch.optim as optim 
import torch.nn as nn 
import torch
import model
import os
from tensorboardX import SummaryWriter
from video_dataset import ucf101
from video_transforms import RandomHorizontalFlip, RandomCrop, ToTensor
from torchvision import transforms


base_lr = 0.003
momentum = 0.9 
batch_size = 30
num_classes = 101
num_epoches = 18
weight_decay = 0.005
train_list = 'list/rgb_train_ucf101.list'
model_dir = 'models'
model_name = 'c3d-video.pth'

#c3d-video.pth is trained with jl's data

os.environ["CUDA_VISIBLE_DEVICES"] = '3'


#accuracy: 0.2942098158326102

def train():

	#create the network 
	c3d = model.C3D(num_classes)
	train_params = [{'params': c3d.get_1x_lr_params(), 'lr': base_lr},
					{'params': c3d.get_2x_lr_params(), 'lr': base_lr * 2}]


	device = get_default_device()
	#import input data
	trainset = ucf101(rgb_list=train_list, transform=transforms.Compose([RandomCrop(112),
                                                                            RandomHorizontalFlip(-1),
                                                                            ToTensor()])) # ToTensor()

	trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=10)
	

	c3d.to(device, non_blocking=True,dtype=torch.float)
	c3d.train()

	#define loss function (Cross Entropy loss)
	criterion = nn.CrossEntropyLoss()
	criterion.to(device)


	#define optimizer 
	optimizer = optim.SGD(train_params, lr=base_lr, momentum=momentum, weight_decay=weight_decay)

	#lr is divided by 10 after every 4 epoches 
	
	scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=6, gamma=0.1)
	writer = SummaryWriter()


	for epoch in range(num_epoches):
		
		running_loss = 0.0
		running_accuracy = 0.0
		scheduler.step()
		
		#(lr1, lr2) = optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']
		#print(epoch, " Learning rate,", lr1, lr2)
		#writer.add_scalar('Weight learning rate/epoch',lr1, epoch)
		#writer.add_scalar('Bias learning rate/epoch',lr2, epoch)

		for i, data in enumerate(trainloader, 0):
			step = epoch * len(trainloader) + i
			inputs, labels = data['clip'].to(device,dtype=torch.float), data['label'].to(device) 
			optimizer.zero_grad()

			
			outputs = c3d(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			#print(len([ w for w in c3d.conv1.weight.grad.view(5184) if w > 1 or w < -1 ]))
			nn.utils.clip_grad_value_(c3d.parameters(),1)
			optimizer.step()

			running_loss +=loss.item()
			print('Step %d, loss: %.3f' %(i, loss.item()))
			writer.add_scalar('Train/Loss', loss.item(),step)
			

			outputs = nn.Softmax(dim=1)(outputs)
			_, predict_label = outputs.max(1)
			correct = (predict_label == labels).sum().item()
			accuracy = float(correct) / float(batch_size)
			running_accuracy+=accuracy
			writer.add_scalar('Train/Accuracy', accuracy,step)


			print("iteration %d, accuracy = %.3f" % (i, accuracy))

			if i % 100 == 99:
				print('[%d, %5d] loss: %.3f' %
					(epoch + 1, i + 1, running_loss / 100))
				print('[%d, %5d] accuracy: %.3f' %
					(epoch + 1, i + 1, running_accuracy / 100))
				running_loss = 0.0 
				running_accuracy = 0.0
			if step % 10000 == 9999:
				torch.save(c3d.state_dict(),os.path.join(model_dir,'%s-%d'%(model_name, step+1)))

		
	print('Finished Training')
	writer.close()

def get_default_device():
	if torch.cuda.is_available():
		return torch.device('cuda:0')
	else:
		return torch.device('cpu')



def main():
	train()

if __name__ == "__main__":
	main()




