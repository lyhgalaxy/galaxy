import math
import os
import time
from datetime import datetime
import numpy as np
import argparse
import os
import cv2    
import tensorflow as tf 
import sys
from tensorflow.python.platform import gfile
from tqdm import tqdm
 


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_dir', default='/disk1/Dataset/Asian_280k/Asian_112x112_folder', help='')
    parser.add_argument('--use_model', default='model.pb',help='') 
    parser.add_argument('--save_dir', default='./save_out', help='')    
 
    args = parser.parse_args()
    return args

def get_graph():
    model = args.test_model
    graph = tf.get_default_graph()
    graph_def = graph.as_graph_def()
    graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
    tf.import_graph_def(graph_def, name='graph')
    summaryWriter = tf.summary.FileWriter('./log/', graph)
    model_name = 'model.pb'

    
def create_graph(model_name):
    # print point
    with tf.gfile.FastGFile(model_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


class Matcher():  
    def __init__(self,gpus='0',weights_file=''):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        self.load_model(weights_file)
        self.inputs_placeholder = self.graph.get_tensor_by_name("input:0")
        self.embeddings = self.graph.get_tensor_by_name("embeddings:0")
        self.embedding_size = self.embeddings.get_shape()[1]

     
    def load_model(self,model):
        # 加载模型
        model_exp = os.path.expanduser(model)
        if (os.path.isfile(model_exp)):
            print('Model filename: %s' % model_exp)
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                with tf.gfile.FastGFile(model_exp, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')
                    self.sess = tf.Session(graph=detection_graph)
                    self.graph = detection_graph
        else:       
            print('Model directory: %s' % model_exp)
            meta_file, ckpt_file = get_model_filenames(model_exp)
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
            saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
            self.sess = tf.get_default_session()
            
            
    def generate_img(self,img_path):
        im=cv2.imread(img_path)            
        im = cv2.resize(im,(112,112),interpolation=cv2.INTER_LINEAR).astype(np.float32)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im[:,:,::-1]
        im -= 127.5
        im *= 0.0078125
        im = np.expand_dims(im,0)
            
        feed_dict = {self.inputs_placeholder: im}
        embeddings_val = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return embeddings_val

        
    def main(self,img_dir,save_dir):
        labels = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))] 
        label_n = 0   # 训练标签
        img_n = 0   # 训练图片名称
        label_val_n = 0  # 验证标签
        img_val_n =0  # 验证图片名
        for label in tqdm(labels):            
            folder = os.path.join(img_dir, label)      
            src_imgs = [os.path.join(folder, f) for f in os.listdir(folder) if
                       os.path.isfile(os.path.join(folder, f)) and f.lower().endswith('.jpg')]

            num = len(src_imgs)
            all_thr = 0   # 所有距离之和
            # op = 0    
            max_thr = 0   # 最大距离
            min_thr = 10   # 最小距离
            embs_list = []   # 输出特征向量列表
            add_emb = np.array(0)   # 设置初始特征向量和
  
            for n in range(num):
                # 图片及其特征向量组成一个子列表
                emb_list=[]
                # 图片路径
                img_path = src_imgs[n]   
                emb_list.append(img_path)
                img_emb = self.generate_img(img_path) 
                # 将同一label下的所有特征向量相加
                add_emb = np.add(add_emb,img_emb)
                emb_list.append(img_emb)
                embs_list.append(emb_list)
            # 求平均特征向量
            avg_emb = np.divide(add_emb,num)
            # emb_n与avg_emb平均距离列表
            diff_lists = []
            for n in range(num):
                diff_list = []
                emb_n = embs_list[n][1]
                diff_list.append(embs_list[n][0])
                # 获取欧氏距离
                diff = np.subtract(emb_n, avg_emb)                
                dist = np.sum(np.square(diff), 1)
                # 去除无用向量，降维度
                # result = np.squeeze(output_data)
                theta = dist[0]
                diff_list.append(theta)
                diff_lists.append(diff_list)
                all_thr+=theta
                # op+=1
                if theta > max_thr:
                    max_thr = theta
                if theta < min_thr:
                    min_thr = theta

            if num >2:
                # 去掉最大值、最小值之后求平均距离（阈值）
                avg_thr = (all_thr-max_thr-min_thr)/(num-2)
                # print('avg_thr================================================= ',avg_thr)
            else:
                continue
                
            if avg_thr < 0.5: 
                s_cnt = 0
                for n in range(num):
                    if diff_lists[n][1] <= avg_thr*1.5:
                        s_cnt+=1
            else:
                continue
            
            if s_cnt >=20:
                # 保存满足要求的图片作为训练
                for n in range(num):
                    if diff_lists[n][1] <= avg_thr*1.5:
                        # print('thr================================================= ',diff_lists[n][1])
                        img = cv2.imread(diff_lists[n][0])                        
                        save_path = os.path.join(save_dir,'train',str(label_n))
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        save_img = os.path.join(save_path,'{}.jpg'.format(img_n))
                        cv2.imwrite(save_img,img)
                        img_n+=1
                label_n+=1
             
            else:
                # 保存满足要求的图片作为验证
                for n in range(num):
                    if diff_lists[n][1] <= avg_thr*1.5:
                        # print('thr================================================= ',diff_lists[n][1])
                        img = cv2.imread(diff_lists[n][0])                        
                        save_path = os.path.join(save_dir,'val',str(label_val_n))
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        save_img = os.path.join(save_path,'{}.jpg'.format(img_val_n))
                        cv2.imwrite(save_img,img)
                        img_val_n+=1
                label_val_n+=1       
  

  
if __name__ == '__main__':    
    args = get_parser()
    img_dir = args.img_dir
    save_dir = args.save_dir
    use_model = args.use_model
    matcher = Matcher(weights_file=use_model)  
    matcher.main(img_dir,save_dir)
    
    
    
    
    
    
    
    
    


