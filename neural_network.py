import numpy as np 
import os
import cv2
import matplotlib.pyplot as plt
import math
import copy
import warnings
import matplotlib
from PIL import Image
import scipy.misc
import imageio
warnings.filterwarnings("error")
mom_rate = 0.20

os.getcwd()
os.chdir('1_gray/')
path = '1_gray/'
os.listdir(path)

def activation_fuc(x):
	
	relu_out = np.zeros(x.shape)
	for i in range(x.shape[-1]):
		for j in range(x.shape[0]):
			for k in range(x.shape[1]):
				if x[j,k,i] > 0:
					relu_out[j,k,i] = x[j,k,i]
					#relu_out[j,k,i] = 0.01 * x[j,k,i]
	return relu_out

def HSVtoRGB(input_image):
	TransMat = np.array([[1.0,0.0,1.14075],[1.0,-0.3455,-0.7169],[1.0,1.779,0.0]])
	output_image = np.zeros(input_image.shape)

	#input_image[:,:,1:3] -= 128

	for i in range(input_image.shape[0]):
		for j in range(input_image.shape[1]):
			output_image[i,j,:] = np.dot(TransMat, input_image[i,j,:])

	output_image = np.where(output_image < 255, output_image, 255)
	output_image = np.where(output_image > 0, output_image, 0)
	return output_image

def RGBtoHSV(input_image):
	TransMat = np.array([[0.299, 0.587, 0.114],[-0.169, -0.331, 0.5],[0.5, -0.419, -0.081]])
	output_image = np.zeros(input_image.shape)
	for i in range(input_image.shape[0]):
		for j in range(input_image.shape[1]):
			output_image[i,j,:] = np.dot(TransMat, input_image[i,j,:])

	#output_image[:,:,1:3] += 128
	return output_image

class Neuron:
	def __init__(self, bias, n, size, n_pre, name,layer,type = 0):
		self.layer = layer
		self.epoch = 0
		self.name = name
		#self.weights = weights
		self.learningrate = 0.000002
		self.bias = bias

		self.filternumber = n
		self.size = size

		#self.filter = (np.random.normal(n_pre, n, size, size)) * (1/np.sqrt(32*n_pre))
		if type == 0:
			self.filter = np.random.uniform(-1.0, 1.0, (size, size, n_pre, n)) * np.sqrt(6/(n_pre ** self.layer + n ** (self.layer)))
		else:
			self.filter = (np.random.normal(-1.0, 1.0, (size, size, n_pre, n)) ) * np.sqrt(6/(n_pre ** self.layer + n ** (self.layer)))
		self.inputs = []
		self.n_pre = n_pre
		self.output = np.zeros((32, 32, n))
		self.filter_dir = np.zeros(self.filter.shape)
		self.random = 0.02
		self.momentum = 0

	def feedfoward(self, inputs):
		features = np.zeros((inputs.shape[0], inputs.shape[1], self.filternumber))
		self.inputs = inputs

		pand_inputs = np.zeros((inputs.shape[0]+2, inputs.shape[1]+2, inputs.shape[2]))
		
		pand_inputs[1:inputs.shape[0]+1,1:inputs.shape[1]+1, :] = inputs[:,:,:]

		for k in range(self.filternumber):
			#cur_filter = self.filter[i, :]
			for i in range(1, inputs.shape[0]+1):
				for j in range(1, inputs.shape[1]+1):
					drop_out = np.random.rand()
					if drop_out >= self.random:
						a = pand_inputs[i-1:i+2, j-1:j+2, :] * self.filter[:, :, :, k]#self.conv(inputs, i, j, k) #+ self.bias					
						features[i-1,j-1,k] = a.sum() + self.bias
					else:
						features[i-1,j-1,k] = 0.0

		self.output = features
		#output = np.dot(self.weights, inputs) + self.bias
		#print(features)
		
		return activation_fuc(features)

	def backprogration(self, loss, epoch, partner, type = 0, have_part = 0):
		#self.epoch = epoch
		self.learningrate = self.learningrate / 1.3
		count = 0

		filter_dir = np.zeros((self.filter.shape))
		loss_prev = np.zeros((self.inputs.shape))
	
		#dir_act = np.where(self.output < 0, self.output, 1)
		#dir_act = np.where(self.output >= 0, dir_act, 0.01)
	
		#loss = dir_act * loss

		
		self.back_conv(loss)

		#print("loss")
		#print(loss)

		self.filter_dir = self.filter_dir/show
		self.momentum = mom_rate * self.momentum - (self.filter_dir)

		self.filter -= (self.learningrate * self.momentum)
		self.bias -= 0.1 * self.learningrate * self.momentum.sum()

		
		if have_part == 1:
			partner.back_conv(loss)
			partner.momentum = mom_rate * partner.momentum - (partner.filter_dir) 
			partner.filter -=  0.5 * partner.learningrate * (partner.momentum)
			partner.filter_dir = np.zeros(partner.filter.shape)
			partner.bias -= 0.05 * partner.learningrate * partner.momentum.sum()
		
		
		#print("\nfilter derivative")
		#print(self.filter_dir)
		'''
		print("\n sum")
		print(self.filter_dir.sum())
		'''

		self.filter_dir = np.zeros(self.filter.shape)

		if type == 0:
			pad_input = np.zeros((loss.shape[0]+self.size -1, loss.shape[1]+self.size - 1, loss.shape[2]))
			pad_input[int((self.size-1)/2):loss.shape[0]+int((self.size-1)/2), int((self.size-1)/2):loss.shape[1]+int((self.size-1)/2),:] = loss

			for layer in range(loss_prev.shape[2]):
				for row in range(loss_prev.shape[0]):
					for col in range(loss_prev.shape[1]):
						loss_prev[row,col,layer] = self.lossback_(layer, row, col, pad_input)

			return loss_prev

	def back_conv(self, output_loss):
		pad_input = np.zeros((self.inputs.shape[0]+self.size-1, self.inputs.shape[1]+self.size-1, self.inputs.shape[2]))
		
		pad_input[int((self.size-1)/2):self.inputs.shape[0]+int((self.size-1)/2), int((self.size-1)/2):self.inputs.shape[1]+int((self.size-1)/2), :] = self.inputs
		limit = 70.0
		

		for i in range(pad_input.shape[2]):
			for j in range(output_loss.shape[2]):
				for row in range(self.size):
					for col in range(self.size):
						a = ((pad_input[(row):row+image_size,(col):col+image_size, i] * output_loss[:,:,j])).sum()
						'''
						print(pad_input[i,(row):row+image_size,(col):col+image_size])
						print("\n***")
						print(output_loss[j,:,:])
						print(a)
						cv2.waitKey(0)
						'''
						if a > limit:
							self.filter_dir[row,col,i,j] = limit
						elif a < -limit:
							self.filter_dir[row,col,i,j] = -limit
						else:	
							self.filter_dir[row,col,i,j] = a


	def lossback_(self, layer, row, col, out_loss):
		derivative = 0
		derivative = ((out_loss[row:row+self.size, col:col+self.size, :] * np.rot90(self.filter[:,:,layer,:], 2, axes = (1,2)))).sum()
		if derivative > 90.0:
			return 90.0
		elif derivative < -90.0:
			return -90.0
		else:
			return derivative



class output(Neuron):
	def __init__(self, bias, n, size, n_pre, name, layer):
		self.layer = layer
		self.epoch = 0
		self.name = name
		self.learningrate = 0.0000002
		self.bias = bias
		self.filternumber = n
		self.size = size
		self.filter = (np.random.normal(0,1.0,(size, size, n_pre, n))) * np.sqrt(6/(n_pre ** self.layer + n ** (self.layer)))
		#self.filter = np.random.uniform(-2.0, 2.0, (n_pre, n, size, size)) * np.sqrt(6/(n_pre ** self.layer + n ** (self.layer)))
		self.inputs = []
		self.n_pre = n_pre
		self.output = np.zeros((32, 32, n))
		self.filter_dir = np.zeros(self.filter.shape)
		self.features = np.zeros((32,32,self.filternumber))
		self.momentum = 0

	def feedfoward(self, inputs):
		features = np.zeros((inputs.shape[0], inputs.shape[1], self.filternumber))
		self.inputs = inputs

		pand_inputs = np.zeros((inputs.shape[0]+int((self.size-1)), inputs.shape[1]+int((self.size-1)), inputs.shape[2]))

		pand_inputs[int((self.size-1)/2):inputs.shape[0]+int((self.size-1)/2), int((self.size-1)/2):inputs.shape[1]+int((self.size-1)/2),:] = inputs

		for k in range(self.filternumber):
			#cur_filter = self.filter[i, :]
			for i in range(int((self.size-1)/2), inputs.shape[0]+int((self.size-1)/2)):
				for j in range(int((self.size-1)/2), inputs.shape[1]+int((self.size-1)/2)):

					a = pand_inputs[i-int((self.size-1)/2):i+int((self.size-1)/2)+1, j-int((self.size-1)/2):j+int((self.size-1)/2)+1,:] * self.filter[:, :, :, k]#self.conv(inputs, i, j, k) #+ self.bias
					features[i-int((self.size-1)/2),j-int((self.size-1)/2),k] = a.sum() + self.bias

		self.features = features
		tanh_out = np.zeros(features.shape)
		for i in range(features.shape[-1]):
			for j in range(features.shape[0]):
				for k in range(features.shape[1]):
					try:
						tanh_out[j,k,i] = (np.exp(features[j,k,i]) - np.exp(-features[j,k,i]))/(np.exp(features[j,k,i]) + np.exp(-features[j,k,i]))
					except RuntimeWarning:
						if features[j,k,i] > 0:
							tanh_out[j,k,i] = 1.0
						elif features[j,k,i] < 0:
							tanh_out[j,k,i] = -1.0
					else:
						tanh_out[j,k,i] = (np.exp(features[j,k,i]) - np.exp(-features[j,k,i]))/(np.exp(features[j,k,i]) + np.exp(-features[j,k,i]))

		self.output = copy.deepcopy(tanh_out)
		return tanh_out

	def backprogration(self, loss, epoch, partner, type = 0):
		self.learningrate = self.learningrate / 1.3
		count = 0
		filter_dir = np.zeros((self.filter.shape))
		loss_prev = np.zeros((self.inputs.shape))

		#loss = loss * 255
		
		sig_dir = (1 - (self.output ** 2)) * -1 * loss 

		self.back_conv(sig_dir)
	
		self.filter_dir = self.filter_dir/show
		self.momentum = mom_rate * self.momentum - (self.filter_dir)
		self.filter +=  self.learningrate * (self.momentum)

		
		partner.back_conv(sig_dir)
		partner.momentum = mom_rate * partner.momentum - (partner.filter_dir) 
		partner.filter +=  0.5 * partner.learningrate * (partner.momentum)
		partner.filter_dir = np.zeros(partner.filter.shape)
		
		self.bias += 0.01 * self.learningrate * self.momentum.sum() 
		partner.bias += 0.005 * partner.learningrate * partner.momentum.sum() 

		#print("filter_dir")
		#print(self.filter_dir)

		self.filter_dir = np.zeros(self.filter.shape)


		if type == 0:
			pad_input = np.zeros((loss.shape[0]+int((self.size-1)), loss.shape[1]+int((self.size-1)),loss.shape[2]))
			pad_input[int((self.size-1)/2):loss.shape[0]+int((self.size-1)/2), int((self.size-1)/2):loss.shape[1]+int((self.size-1)/2),:] = sig_dir

			for layer in range(loss_prev.shape[2]):
				for row in range(loss_prev.shape[0]):
					for col in range(loss_prev.shape[1]):
						loss_prev[row, col, layer] = self.lossback_(layer, row, col, pad_input)

			return loss_prev

class NeuronNetwork:
	def __init__(self):
		self.h1 = Neuron(1, 64, 3, 1, 'h1', 1)
		self.h2 = Neuron(1, 128, 3, 64, 'h2', 2)
		self.h3 = Neuron(1, 256, 3, 128, 'h3', 3)	
		self.h4 = Neuron(1, 128, 3, 256, 'h4', 4)
		self.h5 = Neuron(1, 64, 3, 128, 'h5', 5, type = 1)
		self.output = output(1, 2, 63, 64, 'output', 6)
		
		self.orig_loss = np.zeros((image_size,image_size,2))
		self.regulation_factor = (0.00000001* (abs(self.output.filter)).sum() + 0.0000001*(abs(self.h5.filter).sum() + abs(self.h4.filter).sum() + abs(self.h1.filter).sum() + abs(self.h2.filter).sum() + abs(self.h3.filter).sum()))

	def loss(self, result, true_result):
		true_result_YUV = RGBtoHSV(true_result)
		loss = np.zeros((image_size, image_size, 2))
	
		#result = np.reshape(result, (32,32,-1))
		self.orig_loss = ((true_result_YUV[:,:,1:3]/128.0) - result)
		self.orig_loss -= self.regulation_factor
		#self.orig_loss = np.where(abs(self.orig_loss) > 1, self.orig_loss, 0)

		loss = self.orig_loss**2/2.0
		self.regulation_factor = (0.00000001* (abs(self.output.filter)).sum() + 0.0000001*(abs(self.h5.filter).sum() + abs(self.h4.filter).sum() + abs(self.h1.filter).sum() + abs(self.h2.filter).sum() + abs(self.h3.filter).sum()))

		#accuracy_train = true_result_YUV[:,:,1:3] - np.round(result)
		#accuracy_train = np.where(abs(accuracy_train) > 1, accuracy_train, 0)

		return loss, self.orig_loss

	def forward(self, inputs):
		output_h1 = self.h1.feedfoward(inputs)
		#print("\n***h1***")
		#print(output_h1[2,10:15,:])
		output_h2 = self.h2.feedfoward(output_h1)
		#print("\n***h2***")
		#print(output_h2[2,10:15,:])
		output_h3 = self.h3.feedfoward(output_h2)
		#print("\n***h3***")
		#print(output_h3[2,10:15,:])
		output_h4 = self.h4.feedfoward(output_h3 )

		output_h5 = self.h5.feedfoward(output_h4 + 0.5 * output_h2)

		result = self.output.feedfoward(output_h5 + 0.5 * output_h1)
		#print("\n***result***")
		#print(result[2,10:15,:])

		return result

	def update(self, loss, epoch):
		#loss = np.reshape(loss, (2,32,32))

		loss_h5 = self.output.backprogration(loss, epoch, self.h1)
		loss_h4 = self.h5.backprogration(loss_h5, epoch, self.h2, have_part = 1)
		loss_h3 = self.h4.backprogration (loss_h4, epoch, self.h2, have_part = 0)
		loss_h2 = self.h3.backprogration(loss_h3, epoch, self.h3, have_part = 0)
		loss_h1 = self.h2.backprogration(loss_h2, epoch, self.h3, have_part = 0)
		self.h1.backprogration(loss_h1, epoch, self.h1, type = 1 , have_part = 0)
		
	def train(self):
		count = 0
		global mom_rate
		loss = np.zeros((32,32,2))
		loss_show = np.zeros((32,32,2))
		epoch = 3
		count_loop = 0
		
		accuracy_train = 0
		for loop in range(epoch):
			print(f"{loop}th epoch")
			count = 0
			for i in os.listdir(path):
				count += 1
				
				imgrey = cv2.imread('1/'+i)
				#imgrey = cv2.imread('/Users/yukino/Downloads/documents of classes/520/project4/1/1_32.jpg')
				if i == ".DS_Store":
					continue
				
				imgrey_ = RGBtoHSV(imgrey)[:,:,0:1]
				#cv2.imshow('image',imgrey_)
				#cv2.waitKey(0)
				result = self.forward(copy.deepcopy(imgrey_))
				#result = result + imgrey_
				imtruth = cv2.imread('1/'+i)
				#imtruth = cv2.imread('/Users/yukino/Downloads/documents of classes/520/project4/1/1_32.jpg')
				
				temp = np.concatenate((imgrey_, (result * 128)), axis = 2)
				
				temp = HSVtoRGB(temp)
				
				#temp = cv2.cvtColor(copy.deepcopy(temp),cv2.COLOR_YUV2BGR)
				temp = np.round(temp)
				a = str(int(count_loop / show))+".jpg"
				for i in range(32):
					for j in range(32):
						for k in range(3):
							if abs(imtruth[i,j,k] - temp[i,j,k]) <= 1:
								accuracy_train += 1

				if count_loop % show == 0:	
					cv2.imwrite('/resultpic/'+ a, temp)

				b,a= self.loss(result, imtruth)
				#print(a)

				loss += a
				loss_show += b

			
				count_loop += 1
				if count % show == 0:
					mom_rate = mom_rate / 1.05
					evaluation = np.sqrt(loss_show.sum() / (show))
					print(f"{int(count / show)}th the error is {evaluation}")
					print(f"{int(count / show)}th the accuracy is {accuracy_train/(show*32.0*32.0*3.0)}")

					self.update((loss/show), count_loop)

					print(self.regulation_factor)

					f = open('/error.txt','a')
					f.write(str(evaluation) + '\n')
					f.close()

					temp = accuracy_train/(show*32.0*32.0*3.0)
					f = open('/accuracy.txt','a')
					f.write(str(temp) + '\n')
					f.close()

					loss = np.zeros((32,32,2))
					loss_show = np.zeros((32,32,2))
					accuracy_train = 0.0
					
				
		file_abs = "/parameter.txt"
		with open(file_abs, "w") as f:
			for i in range(self.h1.filter.shape[0]):
				for j in range(self.h1.filter.shape[1]):
					for k in range(self.h1.filter.shape[2]):
						for m in range(self.h1.filter.shape[3]):
							f.write(str(self.h1.filter[i,j,k,m]) + ' ')
						f.write('\n')
					f.write('\n')
				f.write('\n')

			f.write('\n')

			for i in range(self.h2.filter.shape[0]):
				for j in range(self.h2.filter.shape[1]):
					for k in range(self.h2.filter.shape[2]):
						for m in range(self.h2.filter.shape[3]):
							f.write(str(self.h2.filter[i,j,k,m]) + ' ')
						f.write('\n')
					f.write('\n')
				f.write('\n')

			f.write('\n')

			for i in range(self.h3.filter.shape[0]):
				for j in range(self.h3.filter.shape[1]):
					for k in range(self.h3.filter.shape[2]):
						for m in range(self.h3.filter.shape[3]):
							f.write(str(self.h3.filter[i,j,k,m]) + ' ')
						f.write('\n')
					f.write('\n')
				f.write('\n')

			f.write('\n')

			for i in range(self.output.filter.shape[0]):
				for j in range(self.output.filter.shape[1]):
					for k in range(self.output.filter.shape[2]):
						for m in range(self.output.filter.shape[3]):
							f.write(str(self.output.filter[i,j,k,m]) + ' ')
						f.write('\n')
					f.write('\n')
				f.write('\n')

		print("train finished \n")
		count = 0
		loss = np.zeros((32,32,2))
		for i in os.listdir('/1 2/'):
			count += 1
				
			imgrey = cv2.imread('/1 2/'+i)
			#imgrey = cv2.imread('/Users/yukino/Downloads/documents of classes/520/project4/1/1_32.jpg')
			if i == ".DS_Store":
				continue
			
			imgrey_ = RGBtoHSV(imgrey)[:,:,0:1]
			#cv2.imshow('image',imgrey_)
			#cv2.waitKey(0)
			result = self.forward(copy.deepcopy(imgrey_))
			#result = result + imgrey_
			imtruth = cv2.imread('/1 2/'+i)
			#imtruth = cv2.imread('/Users/yukino/Downloads/documents of classes/520/project4/1/1_32.jpg')
			
			temp = np.concatenate((imgrey_, (result * 128)), axis = 2)
			
			temp = HSVtoRGB(temp)
			
			#temp = cv2.cvtColor(copy.deepcopy(temp),cv2.COLOR_YUV2BGR)
			temp = np.round(temp)
			a = str(int(count_loop / show))+".jpg"
			for i in range(32):
				for j in range(32):
					for k in range(3):
						if abs(imtruth[i,j,k] - temp[i,j,k]) <= 1:
							accuracy_train += 1

			if count_loop % 50 == 0:
				cv2.imwrite('/testresultpic/'+ a, temp)
			
			loss_show += ((RGBtoHSV(imtruth)[:,:,1:] - result) ** 2) * 0.5

		
			count_loop += 1
			if count % 50 == 0:
				evaluation = np.sqrt(loss_show.sum() / (show))
				print(f"{int(count / show)}th the error is {evaluation}")
				print(f"{int(count / show)}th the accuracy is {accuracy_train/(show*32.0*32.0*3.0)}")

				print(self.regulation_factor)

				f = open('/test_error.txt','a')
				f.write(str(evaluation) + '\n')
				f.close()

				temp = accuracy_train/(show*32.0*32.0*3.0)
				f = open('/test_accuracy.txt','a')
				f.write(str(temp) + '\n')
				f.close()

				loss = np.zeros((32,32,2))
				loss_show = np.zeros((32,32,2))
				accuracy_train = 0.0

image_size = 32
show = 20.0

colornet = NeuronNetwork( )
colornet.train()

'''
a = output(1.0, 1,5,1,'test', 0)
print(a.filter)
test_input = np.full((1,5,5), 1)
test_true_res = np.full((1,5,5), 127.5)
result = a.feedfoward(test_input)
#print(result)
for i in range(1000):
	test_input = np.full((1,5,5), 1)
	test_true_res = np.full((1,5,5), 127.5)
	result = a.feedfoward(test_input)
	update = a.backprogration(test_true_res - result, 1)

	print("\nresult")
	print(result)
	print("\n loss")
	print(test_true_res - result)
	print("\nupdate")
	print(update)
	cv2.waitKey(0)

print("\nfilter")
print(a.filter)
print("\nerror")
print(test_true_res - result)
print("\nupdate")
print(update)
print("\nresult")
print(result)
'''













