__author__ = 'moonkey'

import os
import numpy as np
from PIL import Image
from collections import Counter
import pickle as cPickle
import random, math
from Attention.data_util.bucketdata import BucketData
import cv2


class DataGen(object):
	GO = 1
	EOS = 2

	def __init__(self, evaluate = False,
				 valid_target_len = float('inf'),
				 img_width_range = (12, 320),
				 word_len = 30):
		"""
		:param data_root:
		:param annotation_fn:
		:param lexicon_fn:
		:param img_width_range: only needed for training set
		:return:
		"""

		img_height = 32		 

		if evaluate:
			self.bucket_specs = [(int(math.floor(64 / 4)), int(word_len + 2)), (int(math.floor(108 / 4)), int(word_len + 2)),
								 (int(math.floor(140 / 4)), int(word_len + 2)), (int(math.floor(256 / 4)), int(word_len + 2)),
								 (int(math.floor(img_width_range[1] / 4)), int(word_len + 2))]
		else:
			self.bucket_specs = [(int(64 / 4), 9 + 2), (int(108 / 4), 15 + 2),
							 (int(140 / 4), 17 + 2), (int(256 / 4), 20 + 2),
							 (int(math.ceil(img_width_range[1] / 4)), word_len + 2)]

		self.bucket_min_width, self.bucket_max_width = img_width_range
		self.image_height = img_height
		self.valid_target_len = valid_target_len

		self.bucket_data = {i: BucketData()
							for i in range(self.bucket_max_width + 1)}

	def clear(self):
		self.bucket_data = {i: BucketData()
							for i in range(self.bucket_max_width + 1)}

	
	def gen(self, cropImg):
		img_bw, word= self.read_data(cropImg)
		width = img_bw.shape[-1]
		b_idx = min(width, self.bucket_max_width)
		bs = self.bucket_data[b_idx].append(img_bw, word, os.path.join(''))
		
		b = self.bucket_data[b_idx].flush_out(
				self.bucket_specs,
				valid_target_length=self.valid_target_len,
				go_shift=1)
		if b is not None:
			yield b	 
		
		self.clear()

	def read_data(self, cropImg):			
		img = np.array(cropImg)
		h = img.shape[0]
		w = img.shape[1]
		print('w:{},h:{}'.format(w,h))
		aspect_ratio = float(w) / float(h)
		if aspect_ratio < float(self.bucket_min_width) / self.image_height:			   
			tempmg = cv2.resize(img, (self.bucket_min_width, self.image_height))		   
		elif aspect_ratio > float(
				self.bucket_max_width) / self.image_height:			   
			tempmg = cv2.resize(img,(self.bucket_max_width, self.image_height))
		elif h != self.image_height:			
			tempmg = cv2.resize(img, (int(aspect_ratio * self.image_height), self.image_height))
		else:
			tempmg = img.copy()

		img_bw =  cv2.cvtColor(tempmg,cv2.COLOR_BGR2GRAY)	  
		#img.convert('L')
		img_bw = np.asarray(img_bw, dtype=np.uint8)
		img_bw = img_bw[np.newaxis, :]
		
		# 'a':97, '0':48
		word = [self.GO]
		'''for c in lex:
			assert 96 < ord(c) < 123 or 47 < ord(c) < 58
			word.append(
				ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3)'''
		word.append(self.EOS)
		word = np.array(word, dtype=np.int32)
		# word = np.array( [self.GO] +
		# [ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3
		# for c in lex] + [self.EOS], dtype=np.int32)

		return img_bw, word


def test_gen():
	print('testing gen_valid')
	# s_gen = EvalGen('../../data/evaluation_data/svt', 'test.txt')
	# s_gen = EvalGen('../../data/evaluation_data/iiit5k', 'test.txt')
	# s_gen = EvalGen('../../data/evaluation_data/icdar03', 'test.txt')
	s_gen = EvalGen('../../data/evaluation_data/icdar13', 'test.txt')
	count = 0
	for batch in s_gen.gen(1):
		count += 1
		print(str(batch['bucket_id']) + ' ' + str(batch['data'].shape[2:]))
		assert batch['data'].shape[2] == img_height
	print(count)


if __name__ == '__main__':
	test_gen()
