#!/bin/sh  
# ============ get the file name ===========  
## train
#img_dir="/home/a/SR/train_data/291"  
#txt_path="/home/a/SR/train.txt"

## test
img_dir="/home/a/SR/test_data/Set"  
txt_path="/home/a/SR/test.txt"  
# clear 
: > $txt_path                                                                                                                                           
for file_a in ${img_dir}/*; do
    # path + name  
    echo $file_a >> $txt_path	
    # just name
    #temp_file=`basename $file_a`
    #echo $temp_file >> $txt_path
done  
