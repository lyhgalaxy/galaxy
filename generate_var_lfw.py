# 生成lfw格式的人脸对代码
import glob
import os.path
import numpy as np
import os
 
 

def get_name_list(img_dir):
    # 将同一类图片归类
    imgs = [ f for f in os.listdir(img_dir) if
        os.path.isfile(os.path.join(img_dir, f)) and f.lower().endswith('.jpg')]
        
    all_num = len(imgs)
    id_list = list(range(all_num))
    all_list = []
    while len(id_list):
        # print(len(id_list),end='\r')
        same_list = []
        id1 = id_list[0]
        dst_img = imgs[id1]
        id_list.remove(id1)        
        dst_img_info = dst_img.split('_')
        dst_label = dst_img_info[0]       
        same_list.append(dst_img)
        id = 0        
        while id < len(id_list):
            id2 = id_list[id]
            now_img = imgs[id2]                   
            now_img_info = now_img.split('_')
            now_label = now_img_info[0]      

            id += 1
            if now_label==dst_label:
                id_list.remove(id2)
                id -= 1
                same_list.append(now_img)
        all_list.append(same_list)
    return all_list
                                            

                                            
def create_match_my(img_dir):
    matched_result = set()
    k = 0
    all_list = get_name_list(img_dir)
    for j in range(24):
        for file_list in all_list: 
            if len(file_list) >= 2:
                length = len(file_list)
                random_number1 = np.random.randint(length)
                random_number2 = np.random.randint(length)
                while random_number1 == random_number2:
                    random_number1 = np.random.randint(length)
                    random_number2 = np.random.randint(length)
                base_name1 = os.path.basename(file_list[random_number1 % length])
                base_name2 = os.path.basename(file_list[random_number2 % length])
                if(file_list[random_number1%length] != file_list[random_number2%length]):            
                    matched_result.add(base_name1 + ' ' + base_name2 + ' 1')
                    if len(matched_result) > 10000:
                        break
                    k = k + 1
    return matched_result, k
 
 
def create_unmatch_my(img_dir):
    """不同类的匹配对"""
    unmatched_result = set()
    k = 0
    while len(unmatched_result) <10000:
        sub_dirs = get_name_list(img_dir)
        length_of_dir = len(sub_dirs)
        for j in range(24):
            for i in range(1, length_of_dir):
                class1 = sub_dirs[i]
                random_number = np.random.randint(length_of_dir)
                while random_number == 0 | random_number == i:
                    random_number = np.random.randint(length_of_dir)
                class2 = sub_dirs[random_number]
      
                file_list1 = class1
                file_list2 = class2

                if file_list1 and file_list2:
                    base_name1 = os.path.basename(file_list1[j % len(file_list1)])
                    base_name2 = os.path.basename(file_list2[j % len(file_list2)])
            
         
                    s = base_name1+' '+ base_name2+' 0'
                    if(s not in unmatched_result):
                        unmatched_result.add(s)
                        if len(unmatched_result) > 10000:
                            break
                    k = k + 1
    return unmatched_result, k                                            
                                            
                                                                                     
 
def create_match_content():
    matched_result = set()
    k = 0
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    while len(matched_result) < 3000:
        for sub_dir in sub_dirs[1:]:
            extensions = 'jpg'
            file_list = []
            dir_name = os.path.basename(sub_dir)
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extensions)
            # glob.glob(file_glob)获取指定目录下的所有图片
            file_list.extend(glob.glob(file_glob))
            if not file_list: continue
            if len(file_list) >= 2:
                label_name = dir_name
                length = len(file_list)
                random_number1 = np.random.randint(length)
                random_number2 = np.random.randint(length)
                while random_number1 == random_number2:
                    random_number1 = np.random.randint(length)
                    random_number2 = np.random.randint(length)
                base_name1 = os.path.basename(file_list[random_number1 % length])
                base_name2 = os.path.basename(file_list[random_number2 % length])
                if(file_list[random_number1%length] != file_list[random_number2%length]):
                    base_name1 = label_name+'/'+base_name1          
                    base_name2 = label_name+'/'+base_name2
            
                    matched_result.add(base_name1 + ' ' + base_name2 + ' 1')
                    # print(label_name + ' ' + get_real_str(base_name1) + ' ' + get_real_str(base_name2))
                    k = k + 1
    return matched_result, k
 
 
def create_unmatch_content():
    """不同类的匹配对"""
    unmatched_result = set()
    k = 0
    while len(unmatched_result) <3000:
        sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
        length_of_dir = len(sub_dirs)
        for j in range(24):
            for i in range(1, length_of_dir):
                class1 = sub_dirs[i]
                random_number = np.random.randint(length_of_dir)
                while random_number == 0 | random_number == i:
                    random_number = np.random.randint(length_of_dir)
                class2 = sub_dirs[random_number]
                class1_name = os.path.basename(class1)
                class2_name = os.path.basename(class2)
                extensions = 'jpg'
                file_list1 = []
                file_list2 = []
                file_glob1 = os.path.join(INPUT_DATA, class1_name, '*.' + extensions)
                file_list1.extend(glob.glob(file_glob1))
                file_glob2 = os.path.join(INPUT_DATA, class2_name, '*.' + extensions)
                file_list2.extend(glob.glob(file_glob2))
                if file_list1 and file_list2:
                    base_name1 = os.path.basename(file_list1[j % len(file_list1)])
                    base_name2 = os.path.basename(file_list2[j % len(file_list2)])            
                    s = class2_name + '/' + base_name2 + ' ' + class1_name + '/' + base_name1+' 0'
                    if(s not in unmatched_result):
                        unmatched_result.add(s)
                        if len(unmatched_result) > 3000:
                            break
                    k = k + 1
    return unmatched_result, k
 
 
if __name__ == '__main__':


    # 这个是直接生成比对txt，不是为了生成.bin
    # TODO 图片数据文件夹
    INPUT_DATA = '/disk1/Dataset/DataFaceMatcher/val_data/no_filter_val/compare_ver_data_1000x10'
    txt_path = '/disk1/Dataset/DataFaceMatcher/val_data/test.txt'
    model_use = 'my_img'
    if model_use=='my_img':
        if os.path.isfile(txt_path):
            os.remove(txt_path)
        result, k1 = create_match_my(INPUT_DATA)
        print(k1)
        # print(result)
        result_un, k2 = create_unmatch_my(INPUT_DATA)
        print(k2)
        # print(result_un)
        file = open(txt_path, 'w')
        result1 = list(result)
        result2 = list(result_un)

        file.write('100 100\n')
        for i in range(100):
            for pair in result1[i*100:i*100+100]:
                file.write(pair + '\n')
            for pair in result2[i*100:i*100+100]:
                file.write(pair + '\n')
        file.close()
            
    
    
    elif model_use=='lfw':
    
        if os.path.isfile(txt_path):
            os.remove(txt_path)
        result, k1 = create_match_content()
        print(k1)
        # print(result)
        result_un, k2 = create_unmatch_content()
        print(k2)
        # print(result_un)
        file = open(txt_path, 'w')
        result1 = list(result)
        result2 = list(result_un)

        file.write('10 300\n')
        for i in range(10):
            for pair in result1[i*300:i*300+300]:
                file.write(pair + '\n')
            for pair in result2[i*300:i*300+300]:
                file.write(pair + '\n')
        file.close()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
