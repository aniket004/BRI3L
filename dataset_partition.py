import glob
import shutil
import os

src_dir_img = "/scratch0/aniket/NIPS_code/Illusion_dataset/Localization/ill_Loc/extra/total_img/img/"
dst_dir_img_train = "/scratch1/NIPS_code/Illusion_dataset/Localization/ill_Loc/train/img/"

src_dir_label = "/scratch0/aniket/NIPS_code/Illusion_dataset/Localization/ill_Loc/extra/total_img/label_bw/"
dst_dir_label_train = "/scratch1/NIPS_code/Illusion_dataset/Localization/ill_Loc/train/label_bw/"


dst_dir_img_test = "/scratch1/NIPS_code/Illusion_dataset/Localization/ill_Loc/test/img/"
dst_dir_label_test = "/scratch1/NIPS_code/Illusion_dataset/Localization/ill_Loc/test/label_bw/"



files_img = glob.iglob(os.path.join(src_dir_img, "*.png"))
files_label = glob.iglob(os.path.join(src_dir_label, "*.png"))

list1 = os.listdir(src_dir_img)
list2 = os.listdir(src_dir_label)


#for file in list_: 
    #name, ext = os.path.splitext(file) 
  
# transfer train images  
for i in range(1,15000):
    shutil.copy(src_dir_img+list1[i], dst_dir_img_train)
    shutil.copy(src_dir_label+list2[i], dst_dir_label_train)
    
  
for i in range(15001, len(list1)):
    shutil.copy(src_dir_img+list1[i], dst_dir_img_test)
    shutil.copy(src_dir_label+list2[i], dst_dir_label_test)
    



#for pngfile in glob.iglob(os.path.join(src_dir, "*.png")):
#    shutil.copy(jpgfile, dst_dir)
