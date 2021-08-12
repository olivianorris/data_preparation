"""This script is the test script for Deep3DFaceRecon_pytorch
"""
import pathlib
import time#New19
import os
from options.test_options import TestOptions
from options.train_options import TrainOptions #New1
from data import create_dataset #New2
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch 
from data.flist_dataset import default_flist_reader
from scipy.io import loadmat, savemat
import argparse
import os.path as osp


def get_data_path(root='examples'):
    
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]

    return im_path, lm_path

def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm

def main(rank, train_opt, test_opt, name='examples'):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(test_opt)#New3
    model.setup(test_opt)#New7

    train_dataset = create_dataset(train_opt, rank=rank)#New5 #train_optで作成することで自動的にdataset/train/のパスが指定されている
    train_dataset_batches = len(train_dataset) // train_opt.batch_size#New6
    
    model.device = device
    model.parallelize()
    model.eval()
    #model.to('cuda')#New 撤廃
    #visualizer = MyVisualizer(opt)

    #im_path, lm_path = get_data_path(name)
    #lm3d_std = load_lm3d(opt.bfm_folder) 
    preprocess_start_time = time.time()
    print("preprocess started")
    #for i in range(len(im_path)):
    name_list = ['gt_feat', 'avgpool', 'coeff']
    for i, train_data in enumerate(train_dataset):
        batch_start_time = time.time()
        if i < 100:
            print("time form start of",i," th iteration is", time.time()-batch_start_time)
            
        #img_name = im_path[i].split(os.path.sep)[-1].replace('.png','').replace('.jpg','')
        #if not os.path.isfile(lm_path[i]):
        #    continue
        #im_tensor, lm_tensor = read_data(im_path[i], lm_path[i], lm3d_std)
        #data = {
        #    'imgs': im_tensor,
        #    'lms': lm_tensor
        #}
        """
        set_inputでの入力に対して、適切な位置にフォルダがあればパスすることで、保存する(適当にサンプリングして時間を短縮する)
        """
        model.set_input(train_data)  # unpack data from data loader #
        # test_str =  model.image_paths[12]

        # test_dir = pathlib.Path(test_str).parent
        # name = test_str.split('.')[:-1][0].split("/")[-1]
        
        # test_list = [osp.join(str(test_dir), i, name + '.txt') for i in name_list]
        # if i == 1:
        #     print("test_list: ",test_list)
        # #if os.path.isfile(test_list[0]) and os.path.isfile(test_list[1]) and os.path.isfile(test_list[2]):
        #         pass
        #else:
        """
        スキップ機能はflist_dataset.pyの__init__データローダで代替される
        そもそものパスのリストから削除するようにした
        """
        model.test()           # run inference
        #visuals = model.get_current_visuals()  # get image results
        #visualizer.display_current_results(visuals, 0, opt.epoch, dataset=name.split(os.path.sep)[-1], 
        #    save_results=True, count=i, name=img_name, add_image=False)

        #model.save_mesh(os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.obj')) # save reconstruction meshes
        #model.save_coeff(os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.mat')) # save predicted coefficients
        if i % 200:
            spent_time_per_batch = time.time()-batch_start_time
            spent_time_amount = time.time()- preprocess_start_time
            print("@th iteration finished in * s".replace("@",i).replace("*",spent_time_per_batch))
            print("\n@ image has been processed".replace("@",train_opt.batch_size*i))
            print("\nelapsed time from start: @".replace("@",spent_time_amount))

def data_summary():
    masks_path = 'datalist/train/masks.txt'
    imlist = []
    with open(masks_path, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)
    msk_paths = imlist
    #msk_paths = [os.path.join("./", i) for i in imlist]
    img_path = [msk_path.replace('mask/', '') for msk_path in msk_paths]
    lm_path = ['.'.join(msk_path.replace('mask', 'landmarks').split('.')[:-1]) + '.txt' for msk_path in msk_paths]
    gt_feat_path = [t.replace('landmarks','gt_feat') for t in lm_path]
    avgpool_path = [t.replace('landmarks','avgpool') for t in lm_path]
    coeff_path = [t.replace('landmarks','coeff') for t in lm_path]

    #lms_list, imgs_list, msks_list, gt_feat_list, avgpool_list,\
    #    coeff_list = check_list(msk_paths, img_path, lm_path, gt_feat_path, avgpool_path, coeff_path)
    
    write_list(lm_path, img_path, msk_paths, gt_feat_path, 
    avgpool_path, coeff_path, mode='train', save_folder='datalist', save_name='')
    
    # check if the path is valid
def check_list(rmsks_list, rimgs_list, rlms_list, rgt_feat_list, ravgpool_list, rcoeff_list):
    lms_list, imgs_list, msks_list, gt_feat_list, avgpool_list, coeff_list = [], [], [], [], [], []
    for i in range(len(rlms_list)):
        flag = 'false'
        lm_path = rlms_list[i]
        im_path = rimgs_list[i]
        msk_path = rmsks_list[i]
        gt_feat_path = rgt_feat_list[i]
        avgpool_path = ravgpool_list[i]
        coeff_path = rcoeff_list[i]
        
        if os.path.isfile(lm_path) and os.path.isfile(im_path) and os.path.isfile(msk_path) \
            and os.path.isfile(gt_feat_path) and os.path.isfile(avgpool_path) and os.path.isfile(coeff_path):
            flag = 'true'
            lms_list.append(rlms_list[i])
            imgs_list.append(rimgs_list[i])
            msks_list.append(rmsks_list[i])
            gt_feat_list.append(rgt_feat_list[i])
            avgpool_list.append(ravgpool_list[i])
            coeff_list.append(rcoeff_list[i])
        print(i, rlms_list[i], flag)
    return lms_list, imgs_list, msks_list, gt_feat_list, avgpool_list, coeff_list
def write_list(lms_list, imgs_list, msks_list, gt_feat_list, 
avgpool_list, coeff_list, mode='train',save_folder='datalist', save_name=''
):
    save_path = os.path.join(save_folder, mode)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, save_name + 'gt_feat.txt'), 'w') as fd:
        fd.writelines([i + '\n' for i in gt_feat_list])   
    
    with open(os.path.join(save_path, save_name + 'avgpool.txt'), 'w') as fd:
        fd.writelines([i + '\n' for i in avgpool_list])

    with open(os.path.join(save_path, save_name + 'coeff.txt'), 'w') as fd:
        fd.writelines([i + '\n' for i in coeff_list])

if __name__ == '__main__':
    train_opt = TrainOptions().parse() #New1
    test_opt = TestOptions().parse()  # get test options
    main(0, train_opt, test_opt) #New?
    data_summary()
