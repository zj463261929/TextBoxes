#coding=utf-8
__author__ = 'moonkey'

import os
import numpy as np
from PIL import Image
from collections import Counter
import pickle as cPickle
import random, math
from Attention_chinese.data_util.bucketdata import BucketData
import codecs
import cv2
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

class DataGen(object):
	GO = 1
	EOS = 2
	#获得字对应的索引
	words =[]
	labels = []
	annotation_path1 = "word_label.txt"
	if os.path.isfile(annotation_path1):
		with codecs.open(annotation_path1, 'rb', "utf-8") as ann_file:
			lines = ann_file.readlines()
			for l in lines:
				word, lex = l.strip().split() 
				words.append(word)
				labels.append(lex)
				#print (word.encode("utf-8"), lex)
	else:
		print ("File does not exist:{}".format(annotation_path1))
		
	
	def __init__(self,evaluate = False,
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
	
	def word_label(self, c): #获得word对应的label
		if c in self.words:
			label = self.words.index(c) #list.index()必须是list中包含的值，不然会抛出异常。
			if label > -1:
				return self.labels[label]
			
	def label_word(self, c):#获得label对应的word
		c = str(c)
		if c in self.labels:
			#print ("words len:{}".format(len(self.words)))
			label = self.labels.index(c)
			#print ("label index:{}".format(label))
			if label > -1:
				return self.words[label]
			
	
	def read_data(self, cropImg):
		img = np.array(cropImg)
		'''
		h = img.shape[0]
		w = img.shape[1]
		print('w:{},h:{}'.format(w,h))
		aspect_ratio = float(w) / float(h)
		if aspect_ratio < float(self.bucket_min_width) / self.image_height:
			tempmg = cv2.resize(img, (self.bucket_min_width, self.image_height))		   
		elif aspect_ratio > float(self.bucket_max_width) / self.image_height:
			tempmg = cv2.resize(img,(self.bucket_max_width, self.image_height))
		elif h != self.image_height:			
			tempmg = cv2.resize(img, (int(aspect_ratio * self.image_height), self.image_height))'''

		img_bw =  cv2.cvtColor(cropImg,cv2.COLOR_BGR2GRAY)	  
		#img.convert('L')
		img_bw = np.asarray(img_bw, dtype=np.uint8)
		img_bw = img_bw[np.newaxis, :]

		# 'a':97, '0':48
		word = [self.GO]
		#print "\n"
		'''for c in lex:
			l = c.encode("raw_unicode_escape")
			#print l
			#print lex.encode("utf-8")
			#print l.decode("raw_unicode_escape").encode("utf-8")
			
			if 1:#c in self.words: 
				cc = c.encode("raw_unicode_escape")
				#print ("c:{}".format(c.encode("utf-8")))  #.encode("raw_unicode_escape")
				if cc.find("\u", 0) > -1:
					#print ll.encode("raw_unicode_escape")
					label = self.words.index(c)	
					if label > -1:
						#print label
						#print self.labels[label]
						word.append(self.labels[label])
						#print c.encode("utf-8")
				elif 96 < ord(c) < 123 or 47 < ord(c) < 58 or 64 < ord(c) < 91:
					c = c.lower()
					#print ("ord(c):{}".format(ord(c)))
					label = self.words.index(c)
					if label > -1:
						#print self.labels[label]
						word.append(self.labels[label])
						#print c.encode("utf-8")
			else:
				#print ("c111111 :{}".format(c.encode("utf-8"))) 
				#print ("111111111111")
				word = []
				return img_bw, word 
			
			
			#print word '''
		
		
		word.append(self.EOS)
		word = np.array(word, dtype=np.int32)
		
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
