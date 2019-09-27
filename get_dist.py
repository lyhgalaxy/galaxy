import os
import sys
import numpy as np
import argparse 
from tqdm import tqdm
from random import shuffle
import shutil 
from datetime import datetime 


def get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dist_txt', default='./filter_all.txt')
    parser.add_argument('--save_txt', default='./filter_all2.txt')
    parser.add_argument('--save_dir', default='/disk1/Dataset/DataFaceMatcher/train_data/asian_add_emore/filter_imgs_new')
    parser.add_argument('--img_dir', default='./asian_imgs')
    parser.add_argument('--emore_dir', default='/disk1/Dataset/DataFaceMatcher/train_data/faces_emore/faces_emore_imgs')
    parser.add_argument('--asian_dir', default='/disk1/Dataset/DataFaceMatcher/train_data/asian_add_emore/asian_imgs')
    parser.add_argument('--set_thr', default=1.35,type=float)
     
    args = parser.parse_args()
    return args

  
def mv_imgs(args):   
    with open(args.save_txt,'r') as fr:
        img_lines = fr.readlines()        
    for line in tqdm(img_lines):
        line_info = line.split(' ')
        label = line_info[1]
        dist = float(line_info[2])
        src_dir = os.path.join(args.img_dir,label)
        try:
            if dist >= 1.26: 
                pass
                # shutil.move(src_dir,os.path.join(args.save_dir,'1.26-1.35'))
            elif 1 <= dist < 1.26:
                dst_dir = os.path.join(args.save_dir,'1-1.26',label)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                shutil.move(src_dir,dst_dir)
            else:
                dst_dir = os.path.join(args.save_dir,'0-1',label)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                shutil.move(src_dir,dst_dir)
        except:
            continue

def move_same(args):
    fw = open(args.save_txt,'w')
    with open(args.dist_txt,'r') as fr:
        dst_lines = fr.readlines()
        
    all_num = len(dst_lines)
    id_list = list(range(all_num))

    while len(id_list):
        # print(len(id_list),end='\r')
        id1 = id_list[0]
        dst_line = dst_lines[id1]
        id_list.remove(id1)
        
        dst_line_info = dst_line.split(' ')
        dst_label = dst_line_info[1]       
        min_dist = float(dst_line_info[2])
       
        single_label = True
        id = 0
        while id < len(id_list):
            id2 = id_list[id]
            now_line = dst_lines[id2]                   
            now_line_info = now_line.split(' ')
            now_label = now_line_info[1]      
            now_dist = float(now_line_info[2])
            id += 1
            if now_label==dst_label:
                # print('=====',now_label,dst_label)
                id_list.remove(id2)
                id -= 1
                if now_dist<min_dist:
                    min_dist = now_dist
                    save_line = now_line
                    single_label = False
                                            
        if single_label:            
            save_line = dst_line    
                 
        if min_dist < args.set_thr:
            fw.write(save_line)
                      
    fw.close()                
  



def get_big_num(args):   
    with open('./single_all_new.txt','r') as fr:
        img_lines = fr.readlines()
        
    i = 0        
    for line in tqdm(img_lines):
        line_info = line.split(' ')
        emore_label = line_info[0]
        asian_label = line_info[1]        
        dist = float(line_info[2])
        
        src_emore = os.path.join(args.emore_dir,emore_label)
        src_asian = os.path.join(args.asian_dir,asian_label)

    
        i+=1
        try:
            # if dist < 1.26: 
                # shutil.rmtree(src_asian)
            # continue
            # sys.exit(0)
            if dist > 1.3:  
                save_dir = os.path.join(args.save_dir,'1.3-1.35',str(i))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                emore_save_path = os.path.join(save_dir,emore_label)
                asian_save_path = os.path.join(save_dir,asian_label)
                shutil.copytree(src_emore,emore_save_path)
                shutil.copytree(src_asian,asian_save_path)
                # shutil.move(src_asian,save_dir)
                shutil.rmtree()
                
                
            elif 1.26 < dist<= 1.3:  
                save_dir = os.path.join(args.save_dir,'1.26-1.3',str(i))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                emore_save_path = os.path.join(save_dir,emore_label)
                asian_save_path = os.path.join(save_dir,asian_label)
                shutil.copytree(src_emore,emore_save_path)                
                shutil.copytree(src_asian,asian_save_path)
                # shutil.move(src_asian,save_dir)
            elif 1.24 < dist <= 1.26:  
                save_dir = os.path.join(args.save_dir,'1.24-1.26',str(i))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                emore_save_path = os.path.join(save_dir,emore_label)
                asian_save_path = os.path.join(save_dir,asian_label)
                shutil.copytree(src_emore,emore_save_path)
                shutil.copytree(src_asian,asian_save_path)
                # shutil.move(src_asian,save_dir)
            elif 1.0 < dist <= 1.24:  
                save_dir = os.path.join(args.save_dir,'1.0-1.24',str(i))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                emore_save_path = os.path.join(save_dir,emore_label)
                asian_save_path = os.path.join(save_dir,asian_label)
                shutil.copytree(src_emore,emore_save_path)
                shutil.copytree(src_asian,asian_save_path)
                # shutil.move(src_asian,save_dir)
            elif 0.7<=dist<=1.0:
                save_dir = os.path.join(args.save_dir,'0.7-1',str(i))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                emore_save_path = os.path.join(save_dir,emore_label)
                asian_save_path = os.path.join(save_dir,asian_label)
                shutil.copytree(src_emore,emore_save_path)
                shutil.copytree(src_asian,asian_save_path)
                # shutil.move(src_asian,save_dir)
            elif dist<0.7:
                save_dir = os.path.join(args.save_dir,'0-0.7',str(i))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                emore_save_path = os.path.join(save_dir,emore_label)
                asian_save_path = os.path.join(save_dir,asian_label)
                shutil.copytree(src_emore,emore_save_path)
                shutil.copytree(src_asian,asian_save_path)
                # shutil.move(src_asian,save_dir)
                           
        except Exception as e:
            print(e)
            continue
                        

    
            
def copy_imgs(args):   
    with open('./single_all_new.txt','r') as fr:
        img_lines = fr.readlines()
        
    i = 0        
    for line in tqdm(img_lines):
        line_info = line.split(' ')
        emore_label = line_info[0]
        asian_label = line_info[1]        
        dist = float(line_info[2])
        
        src_emore = os.path.join(args.emore_dir,emore_label)
        src_asian = os.path.join(args.asian_dir,asian_label)

    
        i+=1
        try:
            # if dist < 1.26: 
                # shutil.rmtree(src_asian)
            # continue
            # sys.exit(0)
            if dist > 1.3:  
                save_dir = os.path.join(args.save_dir,'1.3-1.35',str(i))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                emore_save_path = os.path.join(save_dir,emore_label)
                asian_save_path = os.path.join(save_dir,asian_label)
                shutil.copytree(src_emore,emore_save_path)
                shutil.copytree(src_asian,asian_save_path)
                # shutil.move(src_asian,save_dir)
                shutil.rmtree()
                
                
            elif 1.26 < dist<= 1.3:  
                save_dir = os.path.join(args.save_dir,'1.26-1.3',str(i))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                emore_save_path = os.path.join(save_dir,emore_label)
                asian_save_path = os.path.join(save_dir,asian_label)
                shutil.copytree(src_emore,emore_save_path)                
                shutil.copytree(src_asian,asian_save_path)
                # shutil.move(src_asian,save_dir)
            elif 1.24 < dist <= 1.26:  
                save_dir = os.path.join(args.save_dir,'1.24-1.26',str(i))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                emore_save_path = os.path.join(save_dir,emore_label)
                asian_save_path = os.path.join(save_dir,asian_label)
                shutil.copytree(src_emore,emore_save_path)
                shutil.copytree(src_asian,asian_save_path)
                # shutil.move(src_asian,save_dir)
            elif 1.0 < dist <= 1.24:  
                save_dir = os.path.join(args.save_dir,'1.0-1.24',str(i))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                emore_save_path = os.path.join(save_dir,emore_label)
                asian_save_path = os.path.join(save_dir,asian_label)
                shutil.copytree(src_emore,emore_save_path)
                shutil.copytree(src_asian,asian_save_path)
                # shutil.move(src_asian,save_dir)
            elif 0.7<=dist<=1.0:
                save_dir = os.path.join(args.save_dir,'0.7-1',str(i))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                emore_save_path = os.path.join(save_dir,emore_label)
                asian_save_path = os.path.join(save_dir,asian_label)
                shutil.copytree(src_emore,emore_save_path)
                shutil.copytree(src_asian,asian_save_path)
                # shutil.move(src_asian,save_dir)
            elif dist<0.7:
                save_dir = os.path.join(args.save_dir,'0-0.7',str(i))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                emore_save_path = os.path.join(save_dir,emore_label)
                asian_save_path = os.path.join(save_dir,asian_label)
                shutil.copytree(src_emore,emore_save_path)
                shutil.copytree(src_asian,asian_save_path)
                # shutil.move(src_asian,save_dir)
                           
        except Exception as e:
            print(e)
            continue
                        

    
def get_dist(args,split1=False,split2=False):
    fw = open(args.save_txt,'w')
    with open(args.dist_txt,'r') as fr:
        dist_lines = fr.readlines()
    id = 0
    same_num = 0
    min_dist = 2.0
    all_num = len(dist_lines)
    while id < all_num-1:
        cont_for = 0
        bb_s=False
        while cont_for < 10000:
            new_line = dist_lines[id]         
            cont_for+=1            
            new_line_info = new_line.split(' ')
            new_label = new_line_info[0]        
            new_dist = float(new_line_info[2])
            if not bb_s:
                bb_s = True
                min_dist = new_dist
                last_label = new_label 
                save_line = new_line
                continue
            elif bb_s and new_label==last_label:
                id += 1
                if new_dist<min_dist:
                    min_dist = new_dist
                    save_line = new_line                    
                continue
            else:
                if min_dist < args.set_thr:
                    fw.write(save_line)
                    same_num += 1
                break
    fw.close()                
    print('same_num==',same_num)            
            




            
    
if __name__ == '__main__':
    args=get_parser()
    # test()
    # move_same(args)
    copy_imgs(args)
    # with open('./all.txt','r') as fr:
        # dist_lines = fr.readlines()
    # i = 0
    # for line in dist_lines:
        # line_info = line.split(' ')       
        # dist = float(line_info[2])
        # if dist < 1.26:
            # i += 1
            
    # print('============',i)
        
        

