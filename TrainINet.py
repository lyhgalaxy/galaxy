# -*- coding: utf-8 -*-
# /usr/bin/env/python3

from utils.IAgeData_v1 import prepare_dataset
from losses.face_losses import arcface_loss

from nets.AgeNet import inference as inference_AgeNet
# from nets.inception_resnet_v1 import inference as inference_AgeNet

from verification import evaluate
from scipy.optimize import brentq
from utils.common import train
from scipy import interpolate
from datetime import datetime
from sklearn import metrics
import tensorflow as tf
import numpy as np
import argparse
import time
import os
slim = tf.contrib.slim

def get_parser():       
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=12, type=int,help='epoch to train the network')
    # parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--class_number', default=100,type=int, 
                        help='age: 100, gender: 2')
    parser.add_argument('--embedding_size',
                        help='Dimensionality of the embedding.',type=int, default=128)
    parser.add_argument('--weight_decay', default=5e-5,type=float, help='L2 weight regularization.')
    parser.add_argument('--lr_step', help='Number of epochs for learning rate piecewise.', default=[3, 6, 9, 12])
    parser.add_argument('--train_batch_size',type=int, default=512, help='batch size to train network')
    parser.add_argument('--test_batch_size',
                        help='Number of images to process in a batch in the test set.',type=int, default=100)
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='', help='evluate datasets base path')
    parser.add_argument('--eval_nrof_folds',
                        help='Number of folds to use for cross validation. Mainly used for testing.',
                        type=int, default=10)
    parser.add_argument('--tfrecords_file_path', default='', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--out_path', default='./out', help='') 
    parser.add_argument('--saver_maxkeep', default=50, help='tf.train.Saver max keep ckpt files')
    #parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--summary_interval', default=400, help='interval to save summary')
    parser.add_argument('--ckpt_interval', type=int,default=2000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', type=int,default=2000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=50, help='intervals to save ckpt file')
    parser.add_argument('--pretrained_model', type=str, default='', 
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--optimizer', type=str, 
                        choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM','ADABOUND','AMSBOUND'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.999)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
                        help='Loss based on the norm of the activations in the prelogits layer.', default=2e-5)
    parser.add_argument('--prelogits_norm_p', type=float,
                        help='Norm to use for prelogits norm loss.', default=1.0)
    parser.add_argument('--network',default='mobilefacenet',help='' )    
    parser.add_argument('--channel_div',type=int,default=1,help='' )
    parser.add_argument('--gpu_id',type=str,default="0",help='' )
    parser.add_argument('--chanel_div',type=int,default=1,help='' )
    parser.add_argument('--lr_list',default=[0.1, 0.01, 0.001, 0.0001, 0.00001],help='' )
    parser.add_argument('--some_label',default='',help='' )
    parser.add_argument('--root_dir',default='',help='' )
    parser.add_argument('--train_txt',default='',help='' )
    # parser.add_argument('--use_bn_scale',type=bool,default=True,help='' )
    
    
    args = parser.parse_args()
    return args

args = get_parser()

lr_list_str =args.lr_list.split(',')
lr_list = [float(lr) for lr in lr_list_str]

lr_step_str = args.lr_step.split(',')
lr_step = [int(lr) for lr in lr_step_str]



print('learning rate list: ',lr_list)
# print(type(lr_list))

if __name__ == '__main__':    
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    out_path = args.out_path+'-{}-cd{}-{}'.format(args.network,args.channel_div,subdir)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    with open(os.path.join(out_path,'readme.txt'),'w+') as f:
        config_info = 'Train Time: ' +'{}'.format(subdir)+'\n'
        config_info += 'Same Label: ' + args.some_label+'\n'
        config_info += 'Train Net: ' + args.network+'\n'
        config_info += 'Channel Div: ' + str(args.channel_div)+'\n'
        config_info += 'Train Data: '+args.tfrecords_file_path+'\n'
        config_info += 'Val Data: '+str(args.eval_datasets)+'\n'
        f.write(config_info)
        
       
    with tf.Graph().as_default():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        # create log dir        
        log_dir = os.path.join(os.path.expanduser(out_path), 'logs')
        if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
            os.makedirs(log_dir)

        # define global parameters
        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        epoch = tf.Variable(name='epoch', initial_value=-1, trainable=False)
        inputs = tf.placeholder(name='img_inputs', shape=[None, *args.image_size, 3], dtype=tf.float32)   
        # labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
        
        age_labels = tf.placeholder(name='age_labels',shape=[None,], dtype=tf.int64)
        # gender_labels = tf.placeholder(name='gender_labels', shape=[None, ], dtype=tf.int64)
        
        
        phase_train_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=None, name='phase_train')

        # set train datasets     
        dataset, steps = prepare_dataset(args)
        print('shapes:', dataset.output_shapes)
        print('types:', dataset.output_types)
        print('steps:', steps)
        data_it = dataset.make_one_shot_iterator()
        next_data = data_it.get_next()

        
        
        # # set val datasets        
        # ver_list = []
        

        
        # pretrained model path
        pretrained_model = None
        if args.pretrained_model:
            pretrained_model = os.path.expanduser(args.pretrained_model)
            print('Pre-trained model: %s' % pretrained_model)

        # identity the input, for inference
        inputs = tf.identity(inputs, 'input')
        # print("input.shape",input.shape)
        
        if args.network == "AgeNet":
        
            age_prelogits, net_points = inference_AgeNet(inputs, bottleneck_layer_size=args.embedding_size, phase_train=phase_train_placeholder, weight_decay=args.weight_decay)
            
            # age_logits, gender_logits, net_points = inference_AgeNet(inputs, keep_probability=0.8,
                                                                     # phase_train=True, weight_decay=0.00005)
        else :
            print("NOT FOUND NETWORK...")

        # record the network architecture     
        hd_path=os.path.join(log_dir,"Net_Arch.txt")        
        hd = open(hd_path, 'w')
        for key in net_points.keys():
            info = '{}:{}\n'.format(key, net_points[key].get_shape().as_list())
            hd.write(info)
        hd.close()

        
        
        # pre_age = tf.nn.l2_normalize(age_prelogits, 1, 1e-10, name='pre_age')

        # Norm for the prelogits
        # eps = 1e-5
        # prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=args.prelogits_norm_p, axis=1))
        # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * args.prelogits_norm_loss_factor)

        # # inference_loss, logit = cos_loss(prelogits, labels, args.class_number)
        # w_init_method = slim.initializers.xavier_initializer()
        # inference_loss, logit = arcface_loss(embedding=embeddings, labels=labels, w_init=w_init_method, out_num=args.class_number)
        # tf.add_to_collection('losses', inference_loss)

        # # total losses
        # regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # total_loss = tf.add_n([inference_loss] + regularization_losses, name='total_loss')

        # define the learning rate schedule
        # print('labels==',age_labels)
        # print('pre_age==',age_prelogits)
        # 解决ValueError: Rank mismatch: Rank of labels (received 4) should equal rank of
        age_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=age_labels, 
                                                                           logits=age_prelogits)
        # age_cross_entropy= tf.nn.softmax_cross_entropy_with_logits(
        # logits=age_prelogits, labels=age_labels)                                                                  
        age_cross_entropy_mean = tf.reduce_mean(age_cross_entropy)
        
        # gender_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gender_labels,
                                                                              # logits=gender_prelogits)
        # gender_cross_entropy_mean = tf.reduce_mean(gender_cross_entropy)

        # l2 regularization
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(
            [age_cross_entropy_mean] + regularization_losses)

        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_prelogits), age_), axis=1)
        print(age_labels)
        print(age)
        
        abs_loss = tf.losses.absolute_difference(age_labels, age)

        # gender_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(gender_logits, gender_labels, 1), tf.float32))
        learning_rate = tf.train.piecewise_constant(epoch, boundaries=lr_step, values=lr_list,
                                         name='lr_schedule')

        
                     
        # define sess
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping, gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        
        sess = tf.Session(config=config)

        # calculate accuracy
        pred = tf.nn.softmax(age_prelogits)
        correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), tf.cast(age_labels, tf.int64)), tf.float32)
        Accuracy_Op = tf.reduce_mean(correct_prediction)   
        
        # summary writer
        summary_path = os.path.join(out_path,'summary')
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        summary = tf.summary.FileWriter(summary_path, sess.graph)
        summaries = []
        # add train info to tensorboard summary        
        summaries.append(tf.summary.scalar("age_cross_entropy", age_cross_entropy_mean))
        # summaries.append(tf.summary.scalar("gender_cross_entropy", gender_cross_entropy_mean))
        summaries.append(tf.summary.scalar("total loss", total_loss))
        summaries.append(tf.summary.scalar("train_abs_age_error", abs_loss))
        # summaries.append(tf.summary.scalar("gender_accuracy", gender_acc))
        summaries.append(tf.summary.scalar('leraning_rate', learning_rate))
        summary_op = tf.summary.merge(summaries)

        
                
        # train op
        train_op = train(total_loss, global_step, args.optimizer, learning_rate, args.moving_average_decay,
                         tf.global_variables(), summaries, args.log_histograms)
        inc_global_step_op = tf.assign_add(global_step, 1, name='increment_global_step')
        # inc_epoch_op = tf.assign_add(epoch, 1, name='increment_epoch')

        # record trainable variable    
        hd_path=os.path.join(log_dir,"trainable_var.txt")
        hd = open(hd_path, "w")
        for var in tf.trainable_variables():
            hd.write(str(var))
            hd.write('\n')
        hd.close()

        # saver to load pretrained model or save model
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.saver_maxkeep)
        # init all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # load pretrained model
        if pretrained_model:
            print('Restoring pretrained model: %s' % pretrained_model)
            ckpt = tf.train.get_checkpoint_state(pretrained_model)
            print(ckpt)
            saver.restore(sess, ckpt.model_checkpoint_path)
       
        count = 0
        total_accuracy = {}
        for i in range(args.max_epoch):        
            # sess.run(iterator.initializer)
            # _ = sess.run(inc_epoch_op)
            t0 = time.time()
            while True:
                try:
                  
                    images_train, ages_train = sess.run(next_data)
                    # print(len(images_train), len(ages_train), images_train.shape, np.min(images_train), np.max(images_train))                    
                    feed_dict = {inputs: images_train, 
                                age_labels: ages_train, 
                                phase_train_placeholder: True}                    
                    
                    start = time.time()
                    
                    
                    # _, total_loss_val, inference_loss_val, reg_loss_val, _, acc_val = \
                    # sess.run([train_op, total_loss, inference_loss, regularization_losses, inc_global_step_op, Accuracy_Op],
                             # feed_dict=feed_dict)
                                                 
                    _, total_loss_val, reg_loss_val, _, acc_val = \
                    sess.run([train_op, total_loss, regularization_losses, inc_global_step_op, Accuracy_Op],
                             feed_dict=feed_dict)
                             
                             
                    end = time.time()
                    pre_sec = args.train_batch_size/(end - start)

                    count += 1
                    # print training information
                    if count > 0 and count % args.show_info_interval == 0:
                        print('epoch %d, total_step %d, total loss is %.2f ,reg_loss is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
                              (i, count, total_loss_val, np.sum(reg_loss_val), acc_val, pre_sec))

                    # save summary
                    if count > 0 and count % args.summary_interval == 0:                      
                        feed_dict = {inputs: images_train, 
                                age_labels: ages_train, 
                                phase_train_placeholder: True}      
                        summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                        summary.add_summary(summary_op_val, count)

                        
                        
                    if count > 0 and count % args.validate_interval == 0:
                        print('\nIteration', count, 'testing...')
                        for db in range(1):
                        # for db_index in range(len(ver_list)):
                            # start_time = time.time()
                            # data_sets, issame_list = ver_list[db_index]
     
                            
                            # emb_array = np.zeros((data_sets.shape[0], args.embedding_size))
                            
                            # nrof_batches = data_sets.shape[0] // args.test_batch_size
                            
                            # for index in range(nrof_batches): # actual is same multiply 2, test data total
                                # start_index = index * args.test_batch_size
                                # end_index = min((index + 1) * args.test_batch_size, data_sets.shape[0])

                                # feed_dict = {inputs: data_sets[start_index:end_index, ...], phase_train_placeholder: False}
                                
                                
                                
                                
                                # emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                            # tpr, fpr, accuracy, val, val_std, far = evaluate(emb_array, issame_list, nrof_folds=args.eval_nrof_folds)
                            
                            
                            
                            
                            # duration = time.time() - start_time

                            # print("total time %.3fs to evaluate %d images of %s" % (duration, data_sets.shape[0], ver_name_list[db_index]))
                            # print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
                            
                            # print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
                            # print('fpr and tpr: %1.3f %1.3f' % (np.mean(fpr, 0), np.mean(tpr, 0)))

                            # auc = metrics.auc(fpr, tpr)
                            # print('Area Under Curve (AUC): %1.3f' % auc)
                            # # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
                            # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr,fill_value="extrapolate")(x), 0., 1.)
                            # print('Equal Error Rate (EER): %1.3f\n' % eer)
                              

                              
                            # with open(os.path.join(log_dir, '{}_result.txt'.format(ver_name_list[db_index])), 'at') as f:
                                # f.write('%d\t%.5f\t%.5f\n' % (count, np.mean(accuracy), val))
                            
                            
                            
                            # save .ckpt
                          
                            # acc_s = np.mean(accuracy)
                            acc_s = 0.9
                            filename = '{}_iter_{:d}_{:.4f}'.format(args.network,count,acc_s) + '.ckpt'
                            file_path = os.path.join(out_path,'ckpt')
                            if not os.path.exists(file_path):
                                os.makedirs(file_path)
                            f_name = os.path.join(file_path,filename)
                            saver.save(sess, f_name)
                            print("save done...")
                                                        
                            # if ver_name_list == 'lfw' and np.mean(accuracy) > 0.992:
                                # ckpt_best_path= os.path.join(out_path,'ckpt_best')
                                # if not os.path.exists(ckpt_best_path):
                                    # os.makedirs(ckpt_best_path)
                                # print('best accuracy is %.5f' % np.mean(accuracy))
                                # filename = '{}_iter_best_{:d}'.format(args.network,count) + '.ckpt'
                                # f_name = os.path.join(ckpt_best_path, filename)
                                # saver.save(sess, f_name)

                except tf.errors.OutOfRangeError:
                    print("End of epoch %d" % i)
                    f.close()
                    break
                    
            t1 = time.time()
            epoch_time = t1-t0
            print("-------------epoch_time=====" ,epoch_time)

