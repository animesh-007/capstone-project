import gradio as gr
import os
import shutil
import torch
from PIL import Image
import argparse
import pathlib
import cv2
from glob import glob


import PIL
import glob
import torch
import os.path
import numpy as np
from tqdm import tqdm
from shutil import copyfile

# os.system("git clone https://github.com/BadourAlBahar/pose-with-style")
# os.chdir("Thin-Plate-Spline-Motion-Model")
# os.system("mkdir checkpoints")
# os.system("wget -c https://cloud.tsinghua.edu.cn/f/da8d61d012014b12a9e4/?dl=1 -O checkpoints/vox.pth.tar")

# os.chdir("/workspace/U-2-Net")
from skimage import io, transform
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from U2Net.data_loader import RescaleT, ToTensor, ToTensorLab, SalObjDataset
from U2Net.model_u2n import U2NET # full size version 173.6 MB


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def save_output(image_name,pred,d_dir,outline_depth):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = PIL.Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=PIL.Image.BILINEAR)
    pb_np = np.array(imo)
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    image = cv2.cvtColor(np.array(imo), cv2.COLOR_BGR2GRAY)
    if outline_depth != 0:
      contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      cv2.drawContours(image, contours, -1, (255,255,255), outline_depth)
    cv2.imwrite(f'{d_dir}/{imidx}_sil.png',image)

model_dir = '/workspace/PWS/U2Net/u2net_human_seg.pth'
print('downloading 173.6 MB...')
net_u = U2NET(3,1)
if torch.cuda.is_available():
    net_u.load_state_dict(torch.load(model_dir))
    net_u.cuda()
else: net_u.load_state_dict(torch.load(model_dir, map_location='cpu'))
net_u.eval()



if_garment_transfer_then = ["upper_body", "lower_body", "face"]

examples = [['data/fashionWOMENDressesid0000262902_3back.png', 'data/fashionWOMENSkirtsid0000177102_1front.png', 'repose', 'upper_body'],
            ['data/fashionWOMENBlouses_Shirtsid0000635004_1front.png', 'data/fashionWOMENDressesid0000262902_3back.png', 'garment_transfer', 'upper_body'],
            ]

title = "Pose with Style: Detail-Preserving Pose-Guided Image Synthesis with Conditional StyleGAN"
DESCRIPTION = '''### Gradio demo for <b>Pose with Style: Detail-Preserving Pose-Guided Image Synthesis with Conditional StyleGAN</b>, SIGGRAPH Asia 2021. <a href='https://arxiv.org/abs/2109.06166'>[Paper]</a><a href='https://github.com/BadourAlBahar/pose-with-style'>[Github Code]</a>
<img id="overview" alt="overview" src="https://pose-with-style.github.io/images/teaser.jpg" />
'''
# FOOTER = '<img id="visitor-badge" alt="visitor badge" src="https://visitor-badge.glitch.me/badge?page_id=gradio-blocks.Image-Animation-using-Thin-Plate-Spline-Motion-Model" />'

# os.chdir("/workspace/PWS")

def inference(img, target, task, if_garment_transfer_then):
    os.system("rm -rf /workspace/PWS/test_data")
    os.system("rm -rf /workspace/PWS/result")
    # if not os.path.exists('test_data'):
    os.makedirs('/workspace/PWS/test_data', exist_ok=True)

    
    # if task == 'repose':
    #     img.save('test_data/source.png')
    #     target.save('test_data/target_iuv.png')
    #     source = "test_data/source.png"
    #     target = "test_data/target_iuv.png"
    # else:
    img.save('test_data/source.png')
    target.save('test_data/target.png')
    source = "test_data/source.png"
    target = "test_data/target.png"
    # resizing input image
    # import pdb; pdb.set_trace()
    
    img = cv2.imread(source, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    # w,h = img.size
    # if w > h: img = cv2.resize(img, (512, int((h*512)/w)))
    # elif h > w: img = cv2.resize(img, (int((w*512)/h), 512))
    # else: 
    img = cv2.resize(img, (512, 512))
    cv2.imwrite('/workspace/PWS/test_data/source.png', img)

    # resizing target image
    target = cv2.imread(target, cv2.IMREAD_COLOR)
    h, w, _ = target.shape
    # if w > h: target = cv2.resize(target, (512, int((h*512)/w)))
    # elif h > w: target = cv2.resize(target, (int((w*512)/h), 512))
    # else: 
    target = cv2.resize(target, (512, 512))
    # if task == 'repose':
    #     cv2.imwrite('/workspace/PWS/test_data/target_iuv.png', target)
    # else:
    cv2.imwrite('/workspace/PWS/test_data/target.png', target)

    # if task != 'repose':
    os.chdir('/workspace/detectron2/projects/DensePose')
    data_folder = '/workspace/PWS/test_data'
    os.system('rm -rf /workspace/PWS/test_data/.ipynb_checkpoints')
    os.system('rm -rf /workspace/PWS/test_data/source_sil.png /workspace/PWS/test_data/target_sil.png')
    os.system('rm -rf /workspace/PWS/test_data/source_iuv.png /workspace/PWS/test_data/target_iuv.png')

    data_folder = '/workspace/PWS/test_data'
    images_paths = glob.glob(data_folder + os.sep + '*')

    outline_depth = 5 # толщина контура
    test_salobj_dataset = SalObjDataset(img_name_list = images_paths, lbl_name_list = [], transform=transforms.Compose([RescaleT(320),ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)
    for i_test, data_test in tqdm(enumerate(test_salobj_dataloader)):
        print('\nStage 1/2\n')
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        inputs_test = Variable(inputs_test.cuda())
        d1,d2,d3,d4,d5,d6,d7= net_u(inputs_test)
        pred = normPRED(d1[:,0,:,:])
        save_output(images_paths[i_test],pred,data_folder,outline_depth)
        del d1,d2,d3,d4,d5,d6,d7
        # clear_output()

    for i in tqdm(images_paths):
        print('\nStage 2/2\n')
        os.system(f"MKL_SERVICE_FORCE_INTEL=1 python apply_net.py dump configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_s1x/165712116/model_final_844d15.pkl \
            {i} --output /workspace/dump.pkl -v")
        os.system(f"MKL_SERVICE_FORCE_INTEL=1 python pickle2png.py \
            --pickle_file /workspace/dump.pkl \
            --save_path {data_folder}")

    os.chdir('/workspace/PWS')
    
    if task == 'repose':
    #   img = img
        print('repose')
        os.system("MKL_SERVICE_FORCE_INTEL=1 python inference.py \
        --input_path ./test_data \
        --input_name source \
        --target_name target \
        --CCM_pretrained_model /workspace/PWS/checkpoints/CCM_epoch50.pt \
        --pretrained_model /workspace/PWS/checkpoints/posewithstyle.pt \
        --save_path /workspace/PWS/result")
    else: # 'Downsampled Image'
    ####  Resize the longer edge of the input image
        # if_garment_transfer_then = if_garment_transfer_then # "upper_body", "lower_body", "face"

        print('garment transfer')
        os.system(f"MKL_SERVICE_FORCE_INTEL=1 python garment_transfer.py \
        --input_path ./test_data \
        --input_name source \
        --target_name target \
        --CCM_pretrained_model /workspace/PWS/checkpoints/CCM_epoch50.pt \
        --pretrained_model /workspace/PWS/checkpoints/posewithstyle.pt \
        --part {if_garment_transfer_then} \
        --save_path /workspace/PWS/result")
    #   max_res = 512
    #   width, height = img.size
    #   if max(width,height) > max_res:
    #     scale = max_res /max(width,height)
    #     width = int(scale*width)
    #     height = int(scale*height)
    #     img = img.resize((width,height), Image.ANTIALIAS)
      
    # img.save("temp/image.jpg", "JPEG")

    # if task == 'Motion Deblurring':
    #   task = 'Motion_Deblurring'
    #   os.system("python demo_gradio.py --task 'Motion_Deblurring' --input_path 'temp/image.jpg' --result_dir './temp/'")
  
    # if task == 'Defocus Deblurring':
    #   task = 'Single_Image_Defocus_Deblurring'
    #   os.system("python demo_gradio.py --task 'Single_Image_Defocus_Deblurring' --input_path 'temp/image.jpg' --result_dir './temp/'")
  
    # if task == 'Denoising':
    #   task = 'Real_Denoising'
    #   os.system("python demo_gradio.py --task 'Real_Denoising' --input_path 'temp/image.jpg' --result_dir './temp/'")
  
    # if task == 'Deraining':
    #   os.system("python demo_gradio.py --task 'Deraining' --input_path 'temp/image.jpg' --result_dir './temp/'")
  
    return f'/workspace/PWS/result/final_image.png'

gr.Interface(
    inference,
    [
        gr.inputs.Image(type="pil", label="Input"),
        gr.inputs.Image(type="pil", label="Target"),
        gr.inputs.Radio(["repose", "Garment Transfer"], default="repose", label='task'),
        gr.inputs.Dropdown(choices=if_garment_transfer_then, type="value", default='upper_body', label='Body part to focus on')

    ],
    gr.outputs.Image(type="file", label="Output"),
    title=title,
    description=DESCRIPTION,
    # article=article,
    theme ="huggingface",
    examples=examples,
    allow_flagging=False,
    ).launch(debug=True,enable_queue=True, share=True)
  