import os
import random
import shutil

'''
해당 코드는 서로 다른 작업을 한 두사람의 데이터셋으로부터 제공받은 데이터셋에 맞게 파일을 임의 분류하는 작업을 위한 코드.
서로 다른 작업을 했기때문에 폴더의 구성이 다르다.1

(1)
먼저 첫번째 폴더에는 아래와 같은 이름의 폴더가 있고 이는 원본 52살의 여성의 원본으로부터 예상되는 60살의 이미지를 생성했음을 의미.
000002_female_Asian_52_generated_60
해당 폴더에는 새로운 원본이 되는 60살의 이미지가 있고 정상착용한 N장의 마스크 착용이미지가 있다.

(2)
동시에 'step2_incorrect_masked_01'과 'step2_incorrect_masked_02'라는 복수의 폴더명에
각각 '000002_female_Asian_52_generated_60'라는 이름의 incorrect mask이미지가 있다.

해당 코드의 기능은 (1)에서 노말 1장과 마스크착용한 N장의 이미지에서 5장을 임의 추출하여 기존 train데이터셋의 구성과 같이
000002_female_Asian_52_generated_60 라는 폴더명에
normal, mask1, mask2, mask3, mask4, mask5로 이름을 바꿔서 넣어주고
(2)에서 복수의 폴더에서 임의의 incorrect이미지 1장을 추출 파일명은 incorrect로 바꾼 후 000002_female_Asian_52_generated_60에 넣는다.
'''

data_dir = 'stage_1_224x224'

file_list = os.listdir(data_dir)
file_list.sort()


i_need_new_gen = 2358  #2358 #1280 #1078  #62 #1342

#f로 끝나는것은 female, m으로 끝나는것은 male을 의미
file_youngf = []
file_middlef = []
file_oldf = []

file_youngm = []
file_middlem = []
file_oldm = []


female_cnt = 0
male_cnt = 0

for idx,count_f_age in enumerate(file_list):
    tmp_fname_l = count_f_age.split('_')
    if int(tmp_fname_l[-1]) <30:
        if 'female' in tmp_fname_l:
            file_youngf.append(idx)
        else:
            file_youngm.append(idx)
    elif int(tmp_fname_l[-1]) <60:
        if 'female' in tmp_fname_l:
            file_middlef.append(idx)
        else:
            file_middlem.append(idx)
    else:
        if 'female' in tmp_fname_l:
            file_oldf.append(idx)
        else:
            file_oldm.append(idx)

#######################수정수정
# all_f = len(file_young)
all_f = len(file_middlef)
# all_f = len(file_old)

already_get = [-1]

def already_getfunc(file_len):
    f_idx = -1
    while f_idx in already_get:
        f_idx = random.randint(0,file_len)
    return f_idx

for i in range(i_need_new_gen//2):
    #########################################그떄그떄 바꿔줄것
    f_listup_f = random.sample(range(0,len(file_oldf)),i_need_new_gen//2)
    
    for f_idx_female in f_listup_f:
        #######################################################
        idxxxx = file_oldf[f_idx_female]
        img_path = os.path.join(data_dir,file_list[idxxxx])
        print(img_path)
        #shutil.copytree(img_path,f'./get_gen/{file_list[idxxxx]}')
        os.mkdir(f"get_gen/{file_list[idxxxx]}")
        
        file_list2 = os.listdir(img_path)
        file_list2.sort()
        candidates = random.sample(range(1,len(file_list2)-1),5)
        cnt_mask = 0
        for candidate in candidates:
            mask_line = ['mask1','mask2','mask3','mask4','mask5']
            new_split = file_list2[candidate].split('_')
            new_link = file_list2[candidate]
            img_path2 = os.path.join(img_path,new_link)
            if 'masked' in new_split:
                shutil.copy(img_path2,f"get_gen/{file_list[idxxxx]}")
                os.rename(f"get_gen/{file_list[idxxxx]}/{new_link}", f"get_gen/{file_list[idxxxx]}/{mask_line[cnt_mask]}.jpg")
                file_count = os.listdir(f"get_gen/{file_list[idxxxx]}")
                cnt_mask += 1
        
        
        shutil.copy(f"{data_dir}/{file_list[idxxxx]}/{file_list[idxxxx]}_aligned.jpg",f"get_gen/{file_list[idxxxx]}")
        os.rename(f"get_gen/{file_list[idxxxx]}/{file_list[idxxxx]}_aligned.jpg", f"get_gen/{file_list[idxxxx]}/normal.jpg")
        
        incorrect_data_dir = 'incorrect'
        file_list3 = os.listdir(incorrect_data_dir)
        incorrect_idx = random.randint(0,len(file_list3)-1)
        shutil.copy(f"{incorrect_data_dir}/{file_list3[incorrect_idx]}/{file_list[idxxxx]}_aligned.jpg",f"get_gen/{file_list[idxxxx]}")
        os.rename(f"get_gen/{file_list[idxxxx]}/{file_list[idxxxx]}_aligned.jpg", f"get_gen/{file_list[idxxxx]}/incorrect.jpg")
    break

for i in range(i_need_new_gen//2):
    #########################################그떄그떄 
    f_listup_m = random.sample(range(0,len(file_oldm)),i_need_new_gen//2)
    
    for f_idx_male in f_listup_m:
        #######################################################
        idxxxx = file_oldm[f_idx_male]
        img_path = os.path.join(data_dir,file_list[idxxxx])
        print(img_path)
        os.mkdir(f"get_gen/{file_list[idxxxx]}")
        
        file_list2 = os.listdir(img_path)
        file_list2.sort()
        candidates = random.sample(range(1,len(file_list2)-1),5)
        cnt_mask = 0
        for candidate in candidates:
            mask_line = ['mask1','mask2','mask3','mask4','mask5']
            new_split = file_list2[candidate].split('_')
            new_link = file_list2[candidate]
            img_path2 = os.path.join(img_path,new_link)
            if 'masked' in new_split:
                shutil.copy(img_path2,f"get_gen/{file_list[idxxxx]}")
                os.rename(f"get_gen/{file_list[idxxxx]}/{new_link}", f"get_gen/{file_list[idxxxx]}/{mask_line[cnt_mask]}.jpg")
                cnt_mask +=1
        
        
        shutil.copy(f"{data_dir}/{file_list[idxxxx]}/{file_list[idxxxx]}_aligned.jpg",f"get_gen/{file_list[idxxxx]}")
        os.rename(f"get_gen/{file_list[idxxxx]}/{file_list[idxxxx]}_aligned.jpg", f"get_gen/{file_list[idxxxx]}/normal.jpg")
        
        incorrect_data_dir = 'incorrect'
        file_list3 = os.listdir(incorrect_data_dir)
        incorrect_idx = random.randint(0,len(file_list3)-1)
        shutil.copy(f"{incorrect_data_dir}/{file_list3[incorrect_idx]}/{file_list[idxxxx]}_aligned.jpg",f"get_gen/{file_list[idxxxx]}")
        os.rename(f"get_gen/{file_list[idxxxx]}/{file_list[idxxxx]}_aligned.jpg", f"get_gen/{file_list[idxxxx]}/incorrect.jpg")
    break