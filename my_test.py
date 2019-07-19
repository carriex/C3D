import torch
import torch.optim as optim 
import torch.nn as nn
import numpy as np  
from model import C3D

base_lr = 0.003
momentum = 0.9 
batch_size = 3
num_classes = 101
num_epoches = 16
weight_decay = 5e-4

c3d = C3D(101).float()
#fake data 
input_tensor = []
labels = []
for i in range(30): # 30 x 3 images 
	input_tensor.append(torch.randn(3, 3, 16, 112, 112))
	labels.append(torch.randint(101,(3,)))

#loss function 
criterion = nn.CrossEntropyLoss()
#gradient descent 
optimizer = optim.SGD(c3d.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
#update lr 
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=4, gamma=0.1)

for epoch in range(num_epoches):
	running_loss = 0.0 
	for i, data in enumerate(input_tensor):
		optimizer.zero_grad()
		outputs = c3d(data)
		loss = criterion(outputs, labels[i])
		loss.backward()
		nn.utils.clip_grad_value_(c3d.parameters(),1)
		optimizer.step()

		running_loss +=loss.item()
		print(loss.item())
		if i % 30 == 29:
			print('[%d, %5d] loss: %.3f' %
				(epoch + 1, i + 1, running_loss / 30))
			running_loss = 0.0 
	scheduler.step()
print('Finished training')


