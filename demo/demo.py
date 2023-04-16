import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--img_path', type=str, default='input.png')
    parser.add_argument('--output_path', type=str, default='render_img.png')

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

# snapshot load
cfg.set_additional_args(encoder_setting='osx_l', decoder_setting='normal')
from OSX import get_model
from utils.preprocessing import load_img, process_bbox, generate_patch_image
import torch
from utils.vis import render_mesh, save_obj
from utils.human_models import smpl_x
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


# detect human bbox with yolov5s
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
classes = open('coco.names').read().split('\n')
with torch.no_grad():
    results = model(original_img)
person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]
class_ids, confidences, boxes = [], [], []

# Loop through each person detection
for detection in person_results:
    x1, y1, x2, y2, confidence, class_id = detection.tolist()
    class_ids.append(class_id)
    confidences.append(confidence)
    boxes.append([x1, y1, x2 - x1, y2 - y1])

# Apply non-max suppression to remove redundant detections
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Loop through each detected object and draw bounding box
for i in indices:
    i = i[0]
    box = boxes[i]
    x, y, w, h = box
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

# Show image with bounding boxes
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# prepare bbox
bbox = [193, 120, 516-193, 395-120] # xmin, ymin, width, height
bbox = process_bbox(bbox, original_img_width, original_img_height)
img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape) 
img = transform(img.astype(np.float32))/255
img = img.cuda()[None,:,:,:]
inputs = {'img': img}
targets = {}
meta_info = {}
with torch.no_grad():
    out = model(inputs, targets, meta_info, 'test')

mesh = out['smplx_mesh_cam'].detach().cpu().numpy()
np.save('mesh.npy', mesh)
mesh = mesh[0]
# save mesh
save_obj(mesh, smpl_x.face, '../demo/output.obj')

# render mesh
vis_img = img.cpu().numpy()[0].transpose(1,2,0).copy() * 255
focal = [cfg.focal[0] / cfg.input_body_shape[1] * cfg.input_img_shape[1], cfg.focal[1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]]
princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * cfg.input_img_shape[1], cfg.princpt[1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]]
rendered_img = render_mesh(vis_img[..., ::-1], mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
cv2.imwrite('../demo/render_cropped_img.jpg', rendered_img)

vis_img = original_img.copy()
focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
rendered_img = render_mesh(vis_img[..., ::-1], mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
cv2.imwrite('../demo/render_original_img.jpg', rendered_img)

