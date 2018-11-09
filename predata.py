import os
import numpy as np
from PIL import Image
import pickle

f_list = []
for f in os.listdir('./dataset'):
	if f.endswith('jpg'):
		f_list.append(os.path.join('./dataset', f))
output = np.array(Image.open(f_list[0]).resize((128,128)).convert('L'))
output = output[np.newaxis,:,:]	

for i in f_list[1:]:
	ima = np.array(Image.open(i).resize((128,128)).convert('L'))
	ima = ima[np.newaxis,:,:]
	output = np.concatenate((output, ima))
output = output[:,:,:,np.newaxis]


file = open('images.pickle', 'wb')
pickle.dump(output, file)
file.close()

output = []
for i in f_list:
	if i.endswith('0.jpg'):
		lab = 0
	else:	
		lab = 1
	output.append(int(lab))
output = np.array(output, dtype = np.uint8)

num_labels = output.shape[0]
index_offset = np.arange(num_labels) * 2
labels_one_hot = np.zeros((num_labels, 2))
labels_one_hot.flat[index_offset + output.ravel()] = 1
output = labels_one_hot

file = open('labels.pickle', 'wb')
pickle.dump(output, file)
file.close()
