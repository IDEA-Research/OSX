import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import torch
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
from config import cfg
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--img_path', type=str, default='input.png')
    parser.add_argument('--output_folder', type=str, default='output')

    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# load model
cfg.set_additional_args(encoder_setting='osx_l', decoder_setting='normal')
from OSX import get_model
from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
from common.utils.vis import render_mesh, save_obj
from common.utils.human_models import smpl_x
model_path = '../pretrained_models/osx_l.pth.tar'
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
transform = transforms.ToTensor()
original_img = load_img(args.img_path)
original_img_height, original_img_width = original_img.shape[:2]
os.makedirs(args.output_folder, exist_ok=True)

# detect human bbox with yolov5s
detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
with torch.no_grad():
    results = detector(original_img)
person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]
class_ids, confidences, boxes = [], [], []
for detection in person_results:
    x1, y1, x2, y2, confidence, class_id = detection.tolist()
    class_ids.append(class_id)
    confidences.append(confidence)
    boxes.append([x1, y1, x2 - x1, y2 - y1])
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
vis_img = original_img.copy()
for num, indice in enumerate(indices):
    bbox = boxes[indice]  # x,y,h,w
    bbox = process_bbox(bbox, original_img_width, original_img_height)
    img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
    img = transform(img.astype(np.float32))/255
    img = img.cuda()[None,:,:,:]
    inputs = {'img': img}
    targets = {}
    meta_info = {}

    # mesh recovery
    with torch.no_grad():
        out = model(inputs, targets, meta_info, 'test')

    mesh = out['smplx_mesh_cam'].detach().cpu().numpy()
    mesh = mesh[0]

    # save mesh
    save_obj(mesh, smpl_x.face, os.path.join(args.output_folder, f'person_{num}.obj'))

    # render mesh
    focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
    princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
    vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})

# save rendered image
cv2.imwrite(os.path.join(args.output_folder, f'render.jpg'), vis_img[:, :, ::-1])
