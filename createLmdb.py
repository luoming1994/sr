import sys
sys.path.append('/home/a/caffe/python')
import numpy as np
import lmdb
import caffe
from PIL import Image

IDX_FMT = '{:0>10d}'

def img2lmdb(paths_src,path_dst):
	in_db = lmdb.open(path_dst, map_size=int(1e12))
	#f = open(paths_src,'r')    
	img_list = open(paths_src, 'r').read().splitlines()

	# if file aready ,delete file

	with in_db.begin(write=True) as in_txn:
		for in_idx, in_ in enumerate(img_list):
			# load image:
			# - as np.uint8 {0, ..., 255}
			# - in BGR (switch from RGB)
			# - in Channel x Height x Width order (switch from H x W x C)
			im = np.array(Image.open(in_)) # or load whatever ndarray you need
			im = im[:,:,::-1]
			im = im.transpose((2,0,1))
			im_dat = caffe.io.array_to_datum(im)
			in_txn.put(IDX_FMT.format(in_idx), im_dat.SerializeToString())
	in_db.close()

def lh_img2lmdb(img_path,l_lmdb_path,h_lmdb_path):
    """
    img_path: a txtfile pathname which store path names of images 
    l_lmdb_path: lmdb file of low  pixel images
    h_lmdb_path: lmdb file of high pixel images
    """  
    img_list = open(img_path, 'r').read().splitlines()
    
    ### low pixel Image  resize(BICUBIC) X2
    in_db = lmdb.open(l_lmdb_path, map_size=int(1e12))   
    with in_db.begin(write=True) as in_txn:
        for in_idx, in_ in enumerate(img_list):
			# load image:
			# - as np.uint8 {0, ..., 255}
			# - in BGR (switch from RGB)
             # - in Channel x Height x Width order (switch from H x W x C)
             imH = Image.open(in_)
             org_size = imH.size
             re_size = org_size[0]/2,org_size[1]/2
             im2 = imH.resize(re_size, Image.ANTIALIAS)
             imX2 = im2.resize(org_size,Image.BICUBIC)
             
             im = np.array(imX2,dtype=np.float32) # or load whatever ndarray you need         
             im = im[:,:,::-1]
             im = im.transpose((2,0,1))	
             im_dat = caffe.io.array_to_datum(im)
             in_txn.put(IDX_FMT.format(in_idx), im_dat.SerializeToString())
    in_db.close()
    
    ### groundtruth Image
    hin_db = lmdb.open(h_lmdb_path, map_size=int(1e12))   
    with hin_db.begin(write=True) as hin_txn:
		for in_idx, in_ in enumerate(img_list):
			# load image:
			# - as np.uint8 {0, ..., 255}
			# - in BGR (switch from RGB)
			# - in Channel x Height x Width order (switch from H x W x C)
			im = np.array(Image.open(in_),dtype=np.float32) # or load whatever ndarray you need
			im = im[:,:,::-1]
			im = im.transpose((2,0,1))
			im_dat = caffe.io.array_to_datum(im)
			hin_txn.put(IDX_FMT.format(in_idx), im_dat.SerializeToString())
    hin_db.close()
    
    

if __name__ == '__main__':
    img_path = r'/home/a/SR/train.txt'
    l_lmdb_path = r"/home/a/SR/trainL_lmdb"
    h_lmdb_path = r"/home/a/SR/trainH_lmdb"
    lh_img2lmdb(img_path,l_lmdb_path,h_lmdb_path)
    