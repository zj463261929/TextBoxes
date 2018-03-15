#coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import xml.dom.minidom
import cv2
# %matplotlib inline

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '/opt/zhangjing/TextBoxes'  # this file is expected to be in {caffe_root}/examples
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
#print sys.path




# function pick = nms(boxes,threshold,type)
def nms(boxes, threshold):
	if boxes.size==0:
		return np.empty((0,3))
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	s = boxes[:,4]
	area = (x2-x1+1) * (y2-y1+1)
	I = np.argsort(s)
	pick = np.zeros_like(s, dtype=np.int16)
	counter = 0
	while I.size>0:
		i = I[-1]
		pick[counter] = i
		counter += 1
		idx = I[0:-1]
		xx1 = np.maximum(x1[i], x1[idx])
		yy1 = np.maximum(y1[i], y1[idx])
		xx2 = np.minimum(x2[i], x2[idx])
		yy2 = np.minimum(y2[i], y2[idx])
		w = np.maximum(0.0, xx2-xx1+1)
		h = np.maximum(0.0, yy2-yy1+1)
		inter = w * h		 
		o = inter / (area[i] + area[idx] - inter)
		I = I[np.where(o<=threshold)]
	pick = pick[0:counter]
	return pick


#import exp_config	 

import sys, argparse, logging
import numpy as np
from PIL import Image
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
sys.path.insert(0, '/opt/zhangjing/TextBoxes/examples/TextBoxes')
	
from Attention.model.model import Model


'''parameters = process_args(args, defaults)
logging.basicConfig(
	level=logging.DEBUG,
	format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
	filename=parameters.log_path)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)'''

#def text_recognize(cropImg):
print("construct attention model")
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess: 
	model = Model(
			phase = 'test',									
			output_dir = 'results',
			batch_size = 1,
			initial_learning_rate = 1.0,
			num_epoch = 1000,
			steps_per_checkpoint = 500,
			target_vocab_size = 39, 
			model_dir = 'examples/TextBoxes/Attention/train/english',
			target_embedding_size = 10,
			attn_num_hidden = 512,
			attn_num_layers = 2,
			clip_gradients = True,
			max_gradient_norm = 5.0,
			load_model = True,
			valid_target_length = float('inf'),
			gpu_id=0,
			use_gru=True,
			session = sess)	  
	
				
	
	#orignial horizontal proposal
	model_def = '/opt/zhangjing/TextBoxes/examples/TextBoxes/deploy.prototxt'
	model_weights = '/opt/zhangjing/TextBoxes/models/VGGNet/text/longer_conv_300x300/VGG_text_longer_conv_300x300_iter_150000.caffemodel'

	#horizontal proposal and vertical proposal
	'''model_def = '/opt/zhangjing/TextBoxes/jobs1/deploy.prototxt'
	model_weights = '/opt/zhangjing/TextBoxes/jobs1/VGG_text_longer_conv_300x300_iter_260000.caffemodel'''

	#vertical proposal
	#model_def = '/opt/zhangjing/TextBoxes/jobs2/deploy.prototxt'
	#model_weights = '/opt/zhangjing/TextBoxes/jobs2/VGG_text_longer_conv_300x300_iter_190000.caffemodel'

	#scales=((300,300),)
	#scales=((1600,1600),)

	# IMPORTANT: If use mutliple scales in the paper, you need an extra non-maximum superession for the results
	scales=((300,300),(700,700),(700,500),(700,300),(1600,1600))
	#scales=((300,300),)

	import caffe
	#caffe.set_device(0)
	caffe.set_mode_cpu()

	net = caffe.Net(model_def,		# defines the structure of the model
					model_weights,	# contains the trained weights
					caffe.TEST)		# use test mode (e.g., don't perform dropout)

	# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
	print(net.blobs['data'].data.shape)

	#test_list=open('/opt/zhangjing/TextBoxes/data/icdar13/test_list.txt')
	save_dir='/opt/zhangjing/TextBoxes/examples/TextBoxes/test_bb/'
	orig_image_dir = '/opt/zhangjing/TextBoxes/examples/TextBoxes/test'


	from os import path
	files = [x for x in os.listdir(orig_image_dir) if path.isfile(orig_image_dir+os.sep+x)]
	for line in files:
		recttotal = []
		image_path=orig_image_dir+os.sep+line
		save_detection_path=save_dir+'res_'+line[0:len(line)-4]+'.txt'
		image=caffe.io.load_image(image_path)
		saveimge = cv2.imread(image_path)
		image_height,image_width,channels=image.shape
		detection_result=open(save_detection_path,'wt')
		print line
		print scales
		for scale in scales:
			image_resize_height = scale[0]
			image_resize_width = scale[1]
			transformer = caffe.io.Transformer({'data': (1,3,image_resize_height,image_resize_width)})
			transformer.set_transpose('data', (2, 0, 1))
			transformer.set_mean('data', np.array([104,117,123])) # mean pixel
			transformer.set_raw_scale('data', 255)	# the reference model operates on images in [0,255] range instead of [0,1]
			transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

			net.blobs['data'].reshape(1,3,image_resize_height,image_resize_width)
			transformed_image = transformer.preprocess('data', image)
			net.blobs['data'].data[...] = transformed_image

			# Forward pass.
			detections = net.forward()['detection_out']

			# Parse the outputs.
			det_label = detections[0,0,:,1]
			det_conf = detections[0,0,:,2]
			det_xmin = detections[0,0,:,3]
			det_ymin = detections[0,0,:,4]
			det_xmax = detections[0,0,:,5]
			det_ymax = detections[0,0,:,6]

			top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

			top_conf = det_conf[top_indices]
			top_xmin = det_xmin[top_indices]
			top_ymin = det_ymin[top_indices]
			top_xmax = det_xmax[top_indices]
			top_ymax = det_ymax[top_indices]

			#print top_ymax;

			#plt.clf()
			#plt.imshow(image)
			#currentAxis = plt.gca()

			for i in xrange(top_conf.shape[0]):
				xmin = int(round(top_xmin[i] * image.shape[1]))
				ymin = int(round(top_ymin[i] * image.shape[0]))
				xmax = int(round(top_xmax[i] * image.shape[1]))
				ymax = int(round(top_ymax[i] * image.shape[0]))
				xmin = max(1,xmin)
				ymin = max(1,ymin)
				xmax = min(image.shape[1]-1, xmax)
				ymax = min(image.shape[0]-1, ymax)
				score = top_conf[i]
				#result=str(xmin)+','+str(ymin)+','+str(xmax)+','+str(ymax)+'\r\n'
				#detection_result.write(result)
				#print result

				name = '%.2f'%(score)
				coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
				color = 'b'
				#currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
				#currentAxis.text(xmin, ymin, name, bbox={'facecolor':'white', 'alpha':0.5})
				
				if score>0.6:
					recttotal.append([xmin, ymin,xmax,ymax,score])
				#cv2.rectangle(saveimge,(xmin, ymin),(xmax,ymax),(0,255,0),3)
		#print len(recttotal)
		nprect=np.array(recttotal)
		
		nmsIndex=nms(nprect,0.2)
		if len(nmsIndex)>0:
			total_boxes = nprect[nmsIndex,:]
			print("detect num: {}".format(total_boxes.shape[0]))		
			
			strlist=[]
			for i in xrange(total_boxes.shape[0]):
				#print("detect %d: %d,%d,%d,%d" % (i,total_boxes[i][0], total_boxes[i][1],total_boxes[i][2],total_boxes[i][3]))
				#crop = saveimge[int(total_boxes[i][1]):int(total_boxes[i][3]), int(total_boxes[i][0]):int(total_boxes[i][2])]
				dd = 5
				crop = saveimge[int(total_boxes[i][1])-dd:int(total_boxes[i][3])+dd, int(total_boxes[i][0])-dd:int(total_boxes[i][2])+dd]
				cropImg = crop.copy()
				#cv2.rectangle(saveimge,(int(total_boxes[i][0]), int(total_boxes[i][1])),(int(total_boxes[i][2]),int(total_boxes[i][3])),(0,255,0),3) 
				cv2.rectangle(saveimge,(int(total_boxes[i][0])-dd, int(total_boxes[i][1])-dd),(int(total_boxes[i][2])+dd,int(total_boxes[i][3])+dd),(0,255,0),3) 
				
				width = cropImg.shape[1]  
				height = cropImg.shape[0]
				if width<10 or height<10:
					continue
				#print('w1:{},h1:{}'.format(width,height))
				result = model.launch(cropImg)
				#result = text_recognize(cropImg) 
				#print(result)
				text_res =result
				#text_res =result[0]
				#print(text_res)
				str1=''
				for ii in range(len(text_res)):
					if text_res[ii] > 2:
						if text_res[ii] < 13:
							str1=str1+chr(text_res[ii]+48-3)
						else:
							str1=str1+chr(text_res[ii]+97-13)					  
					else:
						break
				print("**************result*************") 
				print(str1) 
				print("*********************************")				  
				strlist.append(str1)				  
				cv2.putText(saveimge, str1, (int(total_boxes[i][0]), int(total_boxes[i][1])), 0, 1, (255, 0 ,0), 1)
			'''for i in xrange(total_boxes.shape[0]):
				cv2.putText(saveimge, strlist[i], (int(total_boxes[i][0]), int(total_boxes[i][1])), 0, 1, (255, 0 ,0), 1)'''
			
		detection_result.close()
		cv2.imwrite('/opt/zhangjing/TextBoxes/examples/TextBoxes/test_vi/'+line,saveimge)	  
		#plt.savefig('/home/TextBoxes/data/TextBoxes/visu_icdar13/'+image_name)
		
	#test_list.close()
	print('success')

