import os
from tqdm import tqdm
import shutil 


def get_num():
    # img_dir='/disk1/Dataset/DataFaceMatcher/train_data/faces_emore/faces_emore_imgs'
    img_dir='./filter_face1'

    i = 0
    m = 0
    labels = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
    print('folder==',len(labels))
            
    for label in tqdm(labels):            
        folder = os.path.join(img_dir, label)      
        # imgs = [ f for f in os.listdir(folder) if
                # os.path.isfile(os.path.join(folder, f)) and f.lower().endswith('.jpg')]
        imgs = [f for f in os.listdir(folder)]
        # m += len(imgs)        
        if len(imgs) < 5:
            i += 1
            m += len(imgs)
            # dst_dir = os.path.join('./remove_imgs10')
            shutil.rmtree(folder)
            # if not os.path.exists(dst_dir):
                # os.makedirs(dst_dir)
            # shutil.move(folder,dst_dir)
    print("num==",i,m)

  

    
def get_num1():    
    with open('./single_all_new.txt','r') as fr:
        img_lines = fr.readlines() 
    m = 0
    for line in img_lines:
        line_info = line.split(' ')
        dist = float(line_info[2])
        label = line_info[1]
        if dist<0.7:
            label_path = os.path.join('/disk1/Dataset/DataFaceMatcher/train_data/asian_2.8m/asian_imgs',label)
            imgs = [f for f in os.listdir(label_path)]
            m += len(imgs)
    print('===========', m)
    

def add_imgs():
    img_dir='/disk1/Dataset/DataFaceMatcher/train_data/asian_add_emore/filter_imgs_new/0-0.5'
    i = 0
    m = 0
    labels = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
    

    for label in tqdm(labels):         
        folder = os.path.join(img_dir, label) 
        label_q = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        label_q_path1 = os.path.join(folder, label_q[0])
        label_q_path2 = os.path.join(folder, label_q[1])
        label_q1 = label_q[0].split('_')[0]
        label_q2 = label_q[1].split('_')[0]

        if label_q1=='emore':
            label_n = label_q[0]
        elif label_q2=='emore':
            label_n = label_q[1]
        save_dir = os.path.join('/disk1/Dataset/DataFaceMatcher/train_data/asian_add_emore/','0.5imgs',label_n)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        imgs1 = [f for f in os.listdir(label_q_path1) if
                os.path.isfile(os.path.join(label_q_path1, f)) and f.lower().endswith('.jpg')]
        for img in imgs1:
            img_path = os.path.join(label_q_path1,img)
            shutil.copy(img_path,save_dir)
            
        imgs2 = [f for f in os.listdir(label_q_path2) if
                os.path.isfile(os.path.join(label_q_path2, f)) and f.lower().endswith('.jpg')]
        for img in imgs2:
            img_path = os.path.join(label_q_path2,img)
            shutil.copy(img_path,save_dir)
        i += 1
        m = m+len(imgs1)+len(imgs2)
    print(i,m)
    




def add_imgs2():
    img_dir='/disk1/Dataset/DataFaceMatcher/train_data/asian_add_emore/0.5imgs'
    labels = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
    rm_img_dir = '/disk1/Dataset/DataFaceMatcher/train_data/asian_add_emore/asian_emore_all'
    
    label_q_num = 0

    for label in tqdm(labels):         
        rm_dir = os.path.join(rm_img_dir,label)
        print(rm_dir)
        shutil.rmtree(rm_dir)
        label_q_num+=1
        try:
           print(rm_dir)
           shutil.rmtree(rm_dir)
           label_q_num+=1
        except:
            pass
 
        
    print(label_q_num)
        
    
if __name__ == '__main__':
    add_imgs2()
    
 



































 
    
    
