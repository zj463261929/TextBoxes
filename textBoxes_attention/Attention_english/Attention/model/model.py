# -*- coding: utf-8 -*-
"""Visual Attention Based OCR Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random, time, os, shutil, math, sys, logging
#import ipdb
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from PIL import Image
import tensorflow as tf
#import keras.backend as K
#from tensorflow.models.rnn.translate import data_utils

from .cnn import CNN
from .seq2seq_model import Seq2SeqModel
from Attention.data_util.data_gen import DataGen
from tqdm import tqdm

try:
    import distance
    distance_loaded = True
except ImportError:
    distance_loaded = False

distance_loaded = False

class Model(object):

    def __init__(self,
            phase,                      
            output_dir,
            batch_size,
            initial_learning_rate,
            num_epoch,
            steps_per_checkpoint,
            target_vocab_size, 
            model_dir, 
            target_embedding_size,
            attn_num_hidden, 
            attn_num_layers,
            clip_gradients,
            max_gradient_norm,
            session,
            load_model,
            gpu_id,
            use_gru,
            evaluate=False,
            valid_target_length=float('inf'),
            reg_val = 0 ):
            
        gpu_device_id = '/gpu:' + str(gpu_id)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        logging.info('loading data')
        # load data
        
        batch_size = 1
        self.s_gen = DataGen(evaluate=True)


        buckets = self.s_gen.bucket_specs
        logging.info('buckets')
        logging.info(buckets)
        if use_gru:
            logging.info('ues GRU in the decoder.')
        print("use gru {}".format(use_gru))
        # variables
        self.img_data = tf.placeholder(tf.float32, shape=(None, 1, 32, None), name='img_data')
        self.zero_paddings = tf.placeholder(tf.float32, shape=(None, None, 512), name='zero_paddings')
        
        self.decoder_inputs = []
        self.encoder_masks = []
        self.target_weights = []
        for i in xrange(int(buckets[-1][0] + 1)):
            self.encoder_masks.append(tf.placeholder(tf.float32, shape=[None, 1],
                                                    name="encoder_mask{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                    name="weight{0}".format(i)))
      
        self.reg_val = reg_val
        self.sess = session
        self.evaluate = evaluate
        self.steps_per_checkpoint = steps_per_checkpoint 
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.buckets = buckets
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.global_step = tf.Variable(0, trainable=False)
        self.valid_target_length = valid_target_length
        self.phase = phase        
        self.learning_rate = initial_learning_rate
        self.clip_gradients = clip_gradients
       
        if phase == 'train':
            self.forward_only = False
        elif phase == 'test':
            self.forward_only = True
        else:
            assert False, phase

        with tf.device(gpu_device_id):
            cnn_model = CNN(self.img_data, not self.forward_only)#True) #
            self.conv_output = cnn_model.tf_output()
            self.concat_conv_output = tf.concat(axis=1, values=[self.conv_output, self.zero_paddings])

            self.perm_conv_output = tf.transpose(self.concat_conv_output, perm=[1, 0, 2])

        with tf.device(gpu_device_id):
            self.attention_decoder_model = Seq2SeqModel(
                encoder_masks = self.encoder_masks,
                encoder_inputs_tensor = self.perm_conv_output, 
                decoder_inputs = self.decoder_inputs,
                target_weights = self.target_weights,
                target_vocab_size = target_vocab_size, 
                buckets = buckets,
                target_embedding_size = target_embedding_size,
                attn_num_layers = attn_num_layers,
                attn_num_hidden = attn_num_hidden,
                forward_only = self.forward_only,
                use_gru = use_gru)

        if not self.forward_only:

            self.updates = []
            self.summaries_by_bucket = []
            with tf.device(gpu_device_id):
                params = tf.trainable_variables()
                # Gradients and SGD update operation for training the model.
                opt = tf.train.AdadeltaOptimizer(learning_rate=initial_learning_rate)
                for b in xrange(len(buckets)):
                    if self.reg_val > 0:
                        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                        logging.info('Adding %s regularization losses', len(reg_losses))
                        logging.debug('REGULARIZATION_LOSSES: %s', reg_losses)
                        loss_op = self.reg_val * tf.reduce_sum(reg_losses) + self.attention_decoder_model.losses[b]
                    else:
                        loss_op = self.attention_decoder_model.losses[b]

                    gradients, params = zip(*opt.compute_gradients(loss_op, params))
                    if self.clip_gradients:
                        gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
                    # Add summaries for loss, variables, gradients, gradient norms and total gradient norm.
                    summaries = []
                   
                    summaries.append(tf.summary.scalar("loss", loss_op))
                    summaries.append(tf.summary.scalar("total_gradient_norm", tf.global_norm(gradients)))
                    all_summaries = tf.summary.merge(summaries)
                    self.summaries_by_bucket.append(all_summaries)
                    # update op - apply gradients
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        self.updates.append(opt.apply_gradients(zip(gradients, params), global_step=self.global_step))

        self.saver_all = tf.train.Saver(tf.all_variables())

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and load_model:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            #self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.saver_all.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            self.sess.run(tf.initialize_all_variables())
        #self.sess.run(init_new_vars_op)


    # train or test as specified by phase
    def launch(self, cropImg):
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        writer = tf.summary.FileWriter(self.model_dir, self.sess.graph)
        if self.phase == 'test':
            if not distance_loaded:
                logging.info('Warning: distance module not installed. Do whole sequence comparison instead.')
            else:
                logging.info('Compare word based on edit distance.')
            num_correct = 0
            num_total = 0
                        
            #如果之前没有处理90K的数据的标签，标签为字典的索引，这里需要字典中的值
            #f=open("/opt/datasets/data/str/mnt/ramdisk/max/90kDICT32px/lexicon.txt","rb")
            #ll=f.readlines()
            
            for batch in self.s_gen.gen(cropImg):
                # Get a batch and make a step.
                start_time = time.time()
                bucket_id = batch['bucket_id']
                img_data = batch['data']
                zero_paddings = batch['zero_paddings']
                decoder_inputs = batch['decoder_inputs']
                target_weights = batch['target_weights']
                encoder_masks = batch['encoder_mask']
                file_list = batch['filenames']
                real_len = batch['real_len']
               
                grounds = [a for a in np.array([decoder_input.tolist() for decoder_input in decoder_inputs]).transpose()]
                _, step_loss, step_logits, step_attns = self.step(encoder_masks, img_data, zero_paddings, decoder_inputs, target_weights, bucket_id, self.forward_only)
                curr_step_time = (time.time() - start_time)
                step_time += curr_step_time / self.steps_per_checkpoint
                logging.info('step_time: %f, loss: %f, step perplexity: %f'%(curr_step_time, step_loss, math.exp(step_loss) if step_loss < 300 else float('inf')))
                loss += step_loss / self.steps_per_checkpoint
                current_step += 1
                step_outputs = [b for b in np.array([np.argmax(logit, axis=1).tolist() for logit in step_logits]).transpose()]
                

                for idx, output, ground in zip(range(len(grounds)), step_outputs, grounds):
                    flag_ground,flag_out = True, True
                    num_total += 1
                    output_valid = []
                    ground_valid = []
                    for j in range(1,len(ground)):
                        s1 = output[j-1]
                        s2 = ground[j]
                        if s2 != 2 and flag_ground:
                            ground_valid.append(s2)
                        else:
                            flag_ground = False
                        if s1 != 2 and flag_out:
                            output_valid.append(s1)
                        else:
                            flag_out = False
                    if distance_loaded:
                        num_incorrect = distance.levenshtein(output_valid, ground_valid)
                       
                        num_incorrect = float(num_incorrect) / len(ground_valid)
                        num_incorrect = min(1.0, num_incorrect)
                    else:
                        print(output_valid, ground_valid)
                        if output_valid == ground_valid:
                            num_incorrect = 0
                        else:
                            num_incorrect = 1
                        
                    num_correct += 1. - num_incorrect
                logging.info('%f out of %d correct' %(num_correct, num_total))
              
        return output_valid
        
    # step, read one batch, generate gradients
    def step(self, encoder_masks, img_data, zero_paddings, decoder_inputs, target_weights,
               bucket_id, forward_only):
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                    " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                    " %d != %d." % (len(target_weights), decoder_size))
        
        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        input_feed[self.img_data.name] = img_data
        input_feed[self.zero_paddings.name] = zero_paddings
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        for l in xrange(int(encoder_size)):
            try:
                input_feed[self.encoder_masks[l].name] = encoder_masks[l]
            except Exception as e:
                pass
                #ipdb.set_trace()
    
        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
    
        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed  = [self.updates[bucket_id],  # Update Op that does SGD.
                    #self.gradient_norms[bucket_id],  # Gradient norm.
                    self.attention_decoder_model.losses[bucket_id],
                             self.summaries_by_bucket[bucket_id]]
            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.attention_decoder_model.outputs[bucket_id][l])
        else:
            output_feed = [self.attention_decoder_model.losses[bucket_id]]  # Loss for this batch.
            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.attention_decoder_model.outputs[bucket_id][l])
            
    
        outputs = self.sess.run(output_feed, input_feed)
        if not forward_only:
            return outputs[2], outputs[1], outputs[3:(3+self.buckets[bucket_id][1])], None  # Gradient norm summary, loss, no outputs, no attentions.
        else:
            return None, outputs[0], outputs[1:(1+self.buckets[bucket_id][1])], outputs[(1+self.buckets[bucket_id][1]):]  # No gradient norm, loss, outputs, attentions.


    