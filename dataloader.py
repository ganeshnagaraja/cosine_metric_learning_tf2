import numpy as np
import PIL.Image
import random
import matplotlib.pyplot as plt
import PIL.ImageOps
import glob
import tensorflow as tf
import os
from functools import partial


class SiameseDataset():

	def __init__(self, imagesFolerDataset, new_size):
		self.imagesFolderDataset = imagesFolerDataset
		self.images_list = []
		self.new_size = new_size
		for subdir, dirs, files in os.walk(self.imagesFolderDataset):
			# file = os.path.join(subdir, file) for file in files
			for file in files:
				class_num = int(file.split("_")[-1].split(".")[0])
				file = os.path.join(subdir, file)
				tuple = (file, class_num)
				self.images_list.append(tuple)

	def __len__(self):
		return len(self.images_list)

	def generate(self):

		for index in range(len(self.images_list)):

			img0_tuple = random.choice(self.images_list)

			while True:
				img1_tuple = random.choice(self.images_list)
				if img0_tuple[1] != img1_tuple[1]:
					break

			# Selecting positive image.
			anchor_image_name = img0_tuple[0].split('/')[-1]
			anchor_class_name = img0_tuple[0].split('/')[-2]

			all_files_in_class = glob.glob(self.imagesFolderDataset + anchor_class_name + '/*')
			all_files_in_class = [x for x in all_files_in_class if x != img0_tuple[0]]

			if len(all_files_in_class)==0:
				positive_image = img0_tuple[0]
			else:
				positive_image = random.choice(all_files_in_class)

			if anchor_class_name != positive_image.split('/')[-2]:
				print("Error")

			anchor = PIL.Image.open(img0_tuple[0])
			negative = PIL.Image.open(img1_tuple[0])
			positive = PIL.Image.open(positive_image)

			anchor = anchor.convert("RGB")
			negative = negative.convert("RGB")
			positive = positive.convert("RGB")

			anchor = np.array(anchor.resize((self.new_size, self.new_size)), dtype=np.float32) / 255.0
			negative = np.array(negative.resize((self.new_size, self.new_size)), dtype=np.float32) / 255.0
			positive = np.array(positive.resize((self.new_size, self.new_size)), dtype=np.float32) / 255.0

			yield anchor, positive, negative

def create_batch_generator(images_dir, new_size, batch_size):
	cars = SiameseDataset(images_dir, new_size)

	generator = partial(cars.generate)
	dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32, tf.float32))

	dataset = dataset.shuffle(50).batch(batch_size)

	return dataset



# batch_generator = create_batch_generator(Config.training_dir, new_size=128, batch_size=Config.train_batch_size)
#
# for i, (anchor, positive, negative) in enumerate(batch_generator):
# 	print("-------------------------------------------yo ------------------------", i)
	# print(anchor)
	# print(positive)
	# print(negative)
