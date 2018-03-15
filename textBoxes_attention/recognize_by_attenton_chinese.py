#coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import xml.dom.minidom
import cv2
import codecs
from PIL import Image,ImageDraw,ImageFont

font = ImageFont.truetype('simsun.ttc',100)
# %matplotlib inline

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '/opt/yushan/TextBoxes'  # this file is expected to be in {caffe_root}/examples
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
    
from Attention_chinese.model.model import Model
#from Attention.data_util.data_gen import DataGen

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

#获得字对应的索引
words =[]
labels = []
annotation_path1 = "/opt/zhangjing/TextBoxes/examples/TextBoxes/word_label.txt"  #最好给绝对路径
if os.path.isfile(annotation_path1):
    with codecs.open(annotation_path1, 'rb', "utf-8") as ann_file:
        lines = ann_file.readlines()
        for l in lines:
            word, lex = l.strip().split() 
            words.append(word)
            labels.append(lex)
            #print (word.encode("utf-8"), lex)
else:
    print ("File does not exist!:{}".format(annotation_path1))
        
def label_word(c):#获得label对应的word
        c = str(c)
        if c in labels:
            #print ("words len:{}".format(len(self.words)))
            label = labels.index(c)
            #print ("label index:{}".format(label))
            if label > -1:
                return words[label]
                
target_vocab_size = len(words)
#def text_recognize(cropImg):
print("construct attention model")

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess: 
    model = Model(
            phase = 'test',                                 
            output_dir = 'results',
            batch_size = 1,
            initial_learning_rate = 1.0,
            num_epoch = 100,
            steps_per_checkpoint = 1000,
            target_vocab_size = 2610, 
            model_dir = 'examples/TextBoxes/Attention_chinese/train/chinese',
            target_embedding_size = 20,
            attn_num_hidden = 128,
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
    model_weights = '/opt/zhangjing/TextBoxes/jobs1/VGG_text_longer_conv_300x300_iter_260000.caffemodel' '''

    #vertical proposal
    #model_def = '/opt/zhangjing/TextBoxes/jobs2/deploy.prototxt'
    #model_weights = '/opt/zhangjing/TextBoxes/jobs2/VGG_text_longer_conv_300x300_iter_190000.caffemodel'

    #scales=((300,300),)
    #scales=((1600,1600),)

    # IMPORTANT: If use mutliple scales in the paper, you need an extra non-maximum superession for the results
    scales=((300,300),(700,700),(700,500),(700,300),(1600,1600)) #多尺度测试
    #scales=((300,300),)

    import caffe
    #caffe.set_device(0)
    caffe.set_mode_cpu()

    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    print(net.blobs['data'].data.shape)

    #test_list=open('/opt/yushan/TextBoxes/data/icdar13/test_list.txt')
    save_dir='/opt/zhangjing/TextBoxes/examples/TextBoxes/test_bb/'
    orig_image_dir = '/opt/zhangjing/TextBoxes/examples/TextBoxes/test_chinese'


    from os import path
    files = [x for x in os.listdir(orig_image_dir) if path.isfile(orig_image_dir+os.sep+x)]
    num = 0
    for line in files:
        num = num + 1
        recttotal = []
        image_path=orig_image_dir+os.sep+line
        save_detection_path=save_dir+'res_'+line[0:len(line)-4]+'.txt'
        image=caffe.io.load_image(image_path)
        saveimge = cv2.imread(image_path)
        image_height,image_width,channels=image.shape
        saveimge1 = Image.open(image_path)
        '''image = np.array(saveimge,dtype=np.float32)
        image_height,image_width,channels  = image.shape[0], image.shape[1], image.shape[2]'''
        detection_result=open(save_detection_path,'wt')
        print line
        print scales
        for scale in scales:
            image_resize_height = scale[0]
            image_resize_width = scale[1]
            transformer = caffe.io.Transformer({'data': (1,3,image_resize_height,image_resize_width)})
            transformer.set_transpose('data', (2, 0, 1))
            transformer.set_mean('data', np.array([104,117,123])) # mean pixel
            transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
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
            str_all = ""
            draw = ImageDraw.Draw(saveimge1)
            for i in xrange(total_boxes.shape[0]):
                #print("detect %d: %d,%d,%d,%d" % (i,total_boxes[i][0], total_boxes[i][1],total_boxes[i][2],total_boxes[i][3]))
                #crop = saveimge[int(total_boxes[i][1]):int(total_boxes[i][3]), int(total_boxes[i][0]):int(total_boxes[i][2])]
                dd = 20
                crop = saveimge[int(total_boxes[i][1])-dd:int(total_boxes[i][3])+dd, int(total_boxes[i][0])-dd:int(total_boxes[i][2])+dd]
                crop1 = crop.copy()
                cv2.rectangle(saveimge,(int(total_boxes[i][0])-dd, int(total_boxes[i][1])-dd),(int(total_boxes[i][2])+dd,int(total_boxes[i][3])+dd),(0,255,0),3)
                #dd = 0
                draw.rectangle(((int(total_boxes[i][0])-dd, int(total_boxes[i][1])-dd),(int(total_boxes[i][2])+dd,int(total_boxes[i][3])+dd)),outline = "red") 
                '''w = abs(int(total_boxes[i][3] - total_boxes[i][1]))
                h = abs(int(total_boxes[i][2] - total_boxes[i][0]))
                crop = saveimge.crop( (int(total_boxes[i][1]), int(total_boxes[i][0]), w,h) )'''
                
                #cv2.imwrite('/opt/yushan/TextBoxes/examples/TextBoxes/test_vi_chinese/'+"crop_"  + str(i) + line,crop1) 
                #cropImg = crop.copy() 
                width = crop1.shape[1]  
                height = crop1.shape[0]
                if width<10 or height<10:
                    continue
                cropImg = cv2.resize(crop1,(100,32))
                #cv2.imwrite('/opt/yushan/TextBoxes/examples/TextBoxes/test_vi_chinese/'+"resize_" + str(i) +line,cropImg) 
                result = model.launch(cropImg)
                #result = text_recognize(cropImg) 
                #print(result)
                text_res = result
                #text_res =result[0]
                #print(text_res)
                str2 = ""
                isOK = False
                for ii in range(len(text_res)):
                    if text_res[ii] > 2:
                        if text_res[ii] < target_vocab_size:
                            c = label_word(text_res[ii])
                            str2 += c
                            print (c.encode("utf-8"))
                        isOK = True
                    else:
                        break
                #strlist.append(str2)
                #strlist.append(" ")

                if isOK==True:
                    draw.text( (int(total_boxes[i][0]),int(total_boxes[i][3])), str2,(255,0,0),font=font)
            print("{} in {}".format(num,len(files)))
            print("**************result*************")  
        detection_result.close()
        #cv2.imwrite('/opt/zhangjing/TextBoxes/examples/TextBoxes/test_vi/'+line,saveimge)   
        #plt.savefig('/home/TextBoxes/data/TextBoxes/visu_icdar13/'+image_name)
        saveimge1.save('/opt/zhangjing/TextBoxes/examples/TextBoxes/test_vi_chinese/'+line,'JPEG')
    #test_list.close()
    print('success')

