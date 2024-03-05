# Copyright (c) Tencent Inc. All rights reserved.
import os
import cv2
import time
import argparse
import os.path as osp
import numpy as np
from matplotlib import pyplot as plt

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmengine.utils import ProgressBar
from mmyolo.registry import RUNNERS
from transformers.models.owlvit.image_processing_owlvit import box_iou

# Removed unnecessary import
# import supervision as sv

BOUNDING_BOX_ANNOTATOR = None  # Define BOUNDING_BOX_ANNOTATOR object
LABEL_ANNOTATOR = None  # Define LABEL_ANNOTATOR object

"""
python image_demo.py ./configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth \
    /home/xuanlin/Downloads/imgs_test_detection 'towel,bowl,spoon,tomato can' --topk 100 --threshold 0.1 --output-dir demo_outputs

python image_demo.py ./configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth \
    ../example_imgs_openx 'eggplant,microwave,banana,fork,yellow towel,red towel,blue towel,red bowl,blue bowl,purple towel,steel bowl,white bowl,red spoon,green spoon,blue spoon,tomato can,strawberry,corn,yellow plate,red plate,cabinet,fridge,screwdriver,mushroom,plastic bottle,coke can,pepsi can,green chip bag,brown chip bag,blue chip bag,sponge,apple,orange' --topk 100 --threshold 0.1 --output-dir demo_outputs
"""

def show_box(box: np.ndarray, ax, label) -> None:
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                               facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)



def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image', help='image path, include image file or dir.')
    parser.add_argument(
        'text',
        help='text prompts, including categories separated by a comma or a txt file with each line as a prompt.'
    )
    parser.add_argument('--topk',
                        default=100,
                        type=int,
                        help='keep topk predictions.')
    parser.add_argument('--threshold',
                        default=0.0,
                        type=float,
                        help='confidence score threshold for predictions.')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device used for inference.')
    parser.add_argument('--show',
                        action='store_true',
                        help='show the detection results.')
    parser.add_argument('--annotation',
                        action='store_true',
                        help='save the annotated detection results as yolo text format.')
    parser.add_argument('--amp',
                        action='store_true',
                        help='use mixed precision for inference.')
    parser.add_argument('--output-dir',
                        default='demo_outputs',
                        help='the directory to save outputs')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def inference_detector(runner,
                       image_path,
                       texts,
                       max_dets,
                       score_thr,
                       output_dir,
                       use_amp=False,
                       show=False,
                       annotation=False):
    data_info = dict(img_id=0, img_path=image_path, texts=texts)
    data_info = runner.pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    tt = time.time()
    with autocast(enabled=use_amp), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        scores_sorted_ids = torch.argsort(-pred_instances['scores'])
        for idx in scores_sorted_ids:
            if pred_instances['scores'][idx] < 0.01:
                continue
            ious = box_iou(pred_instances['bboxes'][idx:idx+1], pred_instances['bboxes'])[0][0]
            ious[idx] = -1.0 # mask out self-IoU
            pred_instances['scores'][ious > 0.3] = 0.0
        pred_instances = pred_instances[
            pred_instances.scores.float() > score_thr]
    print("Time", time.time() - tt)
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    print(pred_instances)
    
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    anno_image = image.copy()
    plt.figure(figsize=(10, 10))
    plt.imshow(anno_image)
    ax = plt.gca()
    for (label, score, bbox) in zip(pred_instances['labels'], pred_instances['scores'], pred_instances['bboxes']):
        show_box(bbox, ax, texts[label][0] + f' {score:0.2f}')
    plt.savefig(osp.join(output_dir, osp.basename(image_path)))
    
    # detections = None  # Define detections object

    # labels = [
    #     f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
    #     zip(detections.class_id, detections.confidence)
    # ]

    # #label images
    # image = cv2.imread(image_path)
    # anno_image = image.copy()
    # image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    # image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    # cv2.imwrite(osp.join(output_dir, osp.basename(image_path)), image)


    if annotation:
        images_dict = {}
        annotations_dict = {}

        images_dict[osp.basename(image_path)] = anno_image
        annotations_dict[osp.basename(image_path)] = detections
        
        ANNOTATIONS_DIRECTORY =  os.makedirs(r"./annotations", exist_ok=True)

        MIN_IMAGE_AREA_PERCENTAGE = 0.002
        MAX_IMAGE_AREA_PERCENTAGE = 0.80
        APPROXIMATION_PERCENTAGE = 0.75
        
        sv.DetectionDataset(
        classes=texts,
        images=images_dict,
        annotations=annotations_dict
        ).as_yolo(
        annotations_directory_path=ANNOTATIONS_DIRECTORY,
        min_image_area_percentage=MIN_IMAGE_AREA_PERCENTAGE,
        max_image_area_percentage=MAX_IMAGE_AREA_PERCENTAGE,
        approximation_percentage=APPROXIMATION_PERCENTAGE
        )


    if show:
        cv2.imshow('Image', image)  # Provide window name
        k = cv2.waitKey(0)
        if k == 27:
            # wait for ESC key to exit
            cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # load text
    if args.text.endswith('.txt'):
        with open(args.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
    else:
        texts = [[t.strip()] for t in args.text.split(',')] + [[' ']]

    output_dir = args.output_dir
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    runner.call_hook('before_run')
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)
    runner.model.eval()

    if not osp.isfile(args.image):
        images = [
            osp.join(args.image, img) for img in os.listdir(args.image)
            if img.endswith('.png') or img.endswith('.jpg')
        ]
    else:
        images = [args.image]

    progress_bar = ProgressBar(len(images))
    for image_path in images:

        inference_detector(runner,
                           image_path,
                           texts,
                           args.topk,
                           args.threshold,
                           output_dir=output_dir,
                           use_amp=args.amp,
                           show=args.show,
                           annotation=args.annotation)
        progress_bar.update()
