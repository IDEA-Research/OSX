import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
from config import cfg
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--img_path', type=str, default='input.png')
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--encoder_setting', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--decoder_setting', type=str, default='normal', choices=['normal', 'wo_face_decoder', 'wo_decoder'])
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')
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
cfg.set_additional_args(encoder_setting=args.encoder_setting, decoder_setting=args.decoder_setting, pretrained_model_path=args.pretrained_model_path)
from common.base import Demoer
demoer = Demoer()
demoer._make_model()
from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
from common.utils.vis import render_mesh, save_obj, vis_keypoints
from common.utils.human_models import smpl_x
model_path = args.pretrained_model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))

demoer.model.eval()

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
vis_mesh = original_img.copy()
vis_kpts = original_img.copy()
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
        out = demoer.model(inputs, targets, meta_info, 'test')

    mesh = out['smplx_mesh_cam'].detach().cpu().numpy()
    mesh = mesh[0]

    # save mesh
    save_obj(mesh, smpl_x.face, os.path.join(args.output_folder, f'person_{num}.obj'))

    # render mesh
    focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
    princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
    vis_mesh = render_mesh(vis_mesh, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})

    #get_2d_pts
    joint_proj = out['smplx_joint_proj'].detach().cpu().numpy()[0]
    joint_proj[:, 0] = joint_proj[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    joint_proj[:, 1] = joint_proj[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    joint_proj = np.concatenate((joint_proj, np.ones_like(joint_proj[:, :1])), 1)
    joint_proj = np.dot(bb2img_trans, joint_proj.transpose(1, 0)).transpose(1, 0)
    vis_kpts = vis_keypoints(vis_kpts, joint_proj)
# save rendered image
cv2.imwrite(os.path.join(args.output_folder, f'render.jpg'), vis_mesh[:, :, ::-1])
cv2.imwrite(os.path.join(args.output_folder, f'kpts.jpg'), vis_kpts[:, :, ::-1])