import tensorflow as tf
import os
import numpy as np
from scipy.stats import multivariate_normal

from network.siamese_net import SiameseNet, TripletLoss
from dataloader import create_batch_generator


class Config():
	training_dir = "/home/gani/DL_Mylearning/object_tracking/cars-dataset/object-tracking-crops-data/crops/"
	testing_dir = "/home/gani/DL_Mylearning/object_tracking/cars-dataset/object-tracking-crops-data/crops_test/"
	train_batch_size = 128
	train_number_epochs = 250
	NUM_CLASSES = 184
	image_height = 128
	image_width = 128
	channels = 3

def get_gaussian_mask():
	#128 is image size
	x, y = np.mgrid[0:1.0:128j, 0:1.0:128j] #128 is input size.
	xy = np.column_stack([x.flat, y.flat])
	mu = np.array([0.5,0.5])
	sigma = np.array([0.22,0.22])
	covariance = np.diag(sigma**2)
	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
	z = z.reshape(x.shape)

	z = z / z.max()
	z  = z.astype(np.float32)
	z = np.stack((z,)*3, axis=-1)
	mask = tf.convert_to_tensor(z)

	return mask

# initialize the model
net = SiameseNet(Config.NUM_CLASSES)

# create the data loader
batch_generator = create_batch_generator(Config.training_dir, Config.image_width, Config.train_batch_size)

# initialize the loss
criterion = TripletLoss(margin=1)

# create the Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

# check if cuda is available
print(" Is GPU available : ", tf.test.is_gpu_available(cuda_only=True))

#Multiply each image with mask to give attention to center of the image.
gaussian_mask = get_gaussian_mask()

iteration_number = 0
counter = []
loss_history = []

@tf.function
def train_step(anchor, positive, negative, net, criterion, optimizer):
	with tf.GradientTape() as tape:
		anchor_out,positive_out,negative_out = net.call(anchor, positive, negative)
		loss = criterion(anchor_out, positive_out, negative_out)
	gradients = tape.gradient(loss, net.trainable_variables)
	optimizer.apply_gradients(grads_and_vars=zip(gradients, net.trainable_variables))

	return loss

for epoch in range(Config.train_number_epochs):
	for i, (anchor, positive, negative) in enumerate(batch_generator):
		anchor, positive, negative = anchor * gaussian_mask, positive * gaussian_mask, negative * gaussian_mask

		loss = train_step(anchor, positive, negative, net, criterion, optimizer)

		if i %100 == 0 :
			print("Epoch number {}\n Current loss {}\n".format(epoch,loss.numpy()))
			iteration_number +=10
			counter.append(iteration_number)
			loss_history.append(loss.numpy())

	if epoch % 5 == 0:
		if not os.path.exists('ckpts/'):
			os.mkdir('ckpts')
		net.save_weights(filepath='ckpts/model_'+str(epoch), save_format='pb')


































