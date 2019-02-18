import sys
import os
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


def read_folder(path):
	files =os.listdir(path)
	for name in files:
		if name.find(" ") != -1:
			os.rename(path+'/'+name, path+'/'+name.replace(' ', '_'))

path_train = 'fruits_360/fruits-360/Training'
path_test = 'fruits_360/fruits-360/Test'

#call the function to rename subfolders
read_folder(path_train)
read_folder(path_test)

train_dataset=datasets.ImageFolder(path_train, transform= transforms.ToTensor())
train_loader= DataLoader(train_dataset, batch_size=4, shuffle= True)
test_dataset=datasets.ImageFolder(path_test, transform= transforms.ToTensor())
test_loader= DataLoader(test_dataset, batch_size=4, shuffle= True)

class Net(nn.Module):
	'''7.Definethelayersinthenetwork'''
	def __init__(self):
		super(Net,self).__init__()

		#1input imagechannel,6outputchannels,5x5squareconvolutionkernel
		self.conv1=nn.Conv2d(3,6,5)
		self.conv2=nn.Conv2d(6,30,5)

		#anaffineoperation:y=Wx+b
		self.fc1=nn.Linear(30*22*22,300)#(sizeofinput,sizeofoutput)
		self.fc2=nn.Linear(300,90)
		self.fc3=nn.Linear(90,83)

		'''Implementtheforwardcomputationofthenetwork'''

	def forward(self,x):
		#Maxpoolingovera(2,2)window
		x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
		#Ifthesizeisasquareyoucanonlyspecifyasinglenumber
		x=F.max_pool2d(F.relu(self.conv2(x)),2)
		x=x.view(-1,self.num_flat_features(x))
		x=F.relu(self.fc1(x))
		x=F.relu(self.fc2(x))
		x=self.fc3(x)
		return x

	def num_flat_features(self,x):
		size=x.size()[1:]
		#alldimensionsexceptthebatchdimension
		num_features=1
		for s in size:
			num_features*=s
		return num_features

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net=Net().to(device)
print(net)

#Define the loss function
criterion = nn.CrossEntropyLoss()
#Define the optimizer
optimizer=optim.SGD(net.parameters(),lr=0.0001,momentum=0.9)

epochs=5
def train(epochs):
	for epoch in range(epochs):
		running_loss=0.0
		for data in enumerate(train_loader, 0):
			images, labels = data
		for i, (images, labels) in enumerate(train_loader,0):
			#get the input
			images = images.to(device)
			labels = labels.to(device)
			# clear the parameter gradients
			optimizer.zero_grad()

			#forward+backward+optimize
			outputs=net(images)
			loss=criterion(outputs,labels)
			loss.backward()
			optimizer.step()

			running_loss+=loss.item()
			if i % 2000 == 1999: #printevery2000mini-batches
				print ('[%d,%5d] loss: %.3f' % (epoch+1,i+1,running_loss/2000))
				running_loss=0.0
				print ('Finished Training')


train(epochs)
net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
	correct = 0
	total = 0
	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
	print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
