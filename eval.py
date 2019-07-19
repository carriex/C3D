import torch.optim as optim 
import torch.nn as nn 
import torch
import model
import os
from dataset import UCF101DataSet

test_list = 'list/test_ucf101.list'
batch_size = 12
num_classes = 101 
model_dir = 'models'
model_name = 'c3d.pth-30'

def eval():
	
	model_path = os.path.join(model_dir,model_name)
	c3d = model.C3D(num_classes).float()
	c3d.load_state_dict(torch.load(model_path))

	testset = UCF101DataSet(datalist_file=test_list, clip_len=16, crop_size=112,split="testing")
	testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=4) 

	total_predict_label = []
	total_accuracy = [] 

	for (i, data) in enumerate(testloader, 0):
		inputs, labels = data['clip'], data['label']
		_, outputs = c3d(inputs.float()).max(1)
		total = labels.size(0)
		correct = (outputs == labels).sum().item()
		accuracy = correct / total
		print("iteration %d, accuracy = %g" % (i, accuracy))
		total_predict_label.append(outputs)
		total_accuracy.append(accuracy)

	total_accuracy = np.array(total_accuracy)
	total_predict_label = np.array(total_predict_label)
	np.savetxt('results/total_accuracy.txt', total_accuracy, fmt = "%.6f")
    np.save('results/total_accuray.npy', total_accuracy)
    np.savetxt('results/predict_label_total.txt', total_predict_label, fmt="%d")
    np.save('results/predict_label_total.npy', total_predict_label)
    print(np.mean(total_accuracy))





def main():
	eval()

if __name__ == "__main__":
	main()



