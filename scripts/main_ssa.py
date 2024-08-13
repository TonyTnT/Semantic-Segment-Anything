import os
import argparse
import torch

from pipeline import semantic_segment_anything_inference, img_load
from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from configs.cityscapes_id2label import CONFIG as CONFIG_CITYSCAPES_ID2LABEL

import torch.distributed as dist
import torch.multiprocessing as mp
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12322'

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    parser.add_argument('--data_dir', help='specify the root path of images and masks')
    parser.add_argument('--sam_type', help='Support sam and mobile_sam')
    # parser.add_argument('--ckpt_path', default='ckp/sam_vit_h_4b8939.pth', help='specify the root path of SAM checkpoint')
    parser.add_argument('--out_dir', default='outs', help='the dir to save semantic annotations')
    parser.add_argument('--save_img', default=True, action='store_true', help='whether to save annotated images')
    parser.add_argument('--world_size', type=int, default=0, help='number of nodes')
    parser.add_argument('--dataset', type=str, default='ade20k', choices=['ade20k', 'cityscapes', 'foggy_driving'], help='specify the set of class names')
    parser.add_argument('--eval', default=False, action='store_true', help='whether to execute evalution')
    parser.add_argument('--gt_path', default=None, help='specify the path to gt annotations')
    parser.add_argument('--model', type=str, default='segformer', choices=['oneformer', 'segformer'], help='specify the semantic branch model')
    args = parser.parse_args()
    return args
    
def main(rank, args):
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    if args.sam_type == 'sam':
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        args.ckpt_path = 'ckp/sam_vit_h_4b8939.pth'
        model = sam_model_registry["vit_h"](checkpoint=args.ckpt_path).to(rank)
    elif args.sam_type == 'sam2':
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator as SamAutomaticMaskGenerator
        # checkpoint = "ckp/sam2_hiera_tiny.pt"
        # model_cfg = "sam2_hiera_t.yaml"
        checkpoint = "ckp/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        model = build_sam2(model_cfg, checkpoint).to(rank)
        
    elif args.sam_type == 'mobile_sam':
        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        args.ckpt_path = '/home/chenx2/MobileSAM/weights/mobile_sam.pt'
        model = sam_model_registry["vit_t"](checkpoint=args.ckpt_path).to(rank)

    mask_branch_model = SamAutomaticMaskGenerator(
        model=model,
        output_mode='coco_rle',
        # points_per_side=32,
        # points_per_batch=64,
        # pred_iou_thresh=0.88,
        # stability_score_thresh=0.95,
        # stability_score_offset=1.0,
        # box_nms_thresh=0.7,
        # crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        # point_grids=None,
        # min_mask_region_area=100,
    )
    print(f"[Model loaded] Mask branch ({args.sam_type}) is loaded.")
    # yoo can add your own semantic branch here, and modify the following code
    if args.model == 'oneformer':
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        if args.dataset == 'ade20k':
            semantic_branch_processor = OneFormerProcessor.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_large")
            semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_large").to(rank)
        elif args.dataset == 'cityscapes':
            semantic_branch_processor = OneFormerProcessor.from_pretrained(
                "shi-labs/oneformer_cityscapes_swin_large")
            semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_cityscapes_swin_large").to(rank)
        elif args.dataset == 'foggy_driving':
            semantic_branch_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_dinat_large")
            semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_cityscapes_dinat_large").to(rank)
        else:
            raise NotImplementedError()
    elif args.model == 'segformer':
        from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
        if args.dataset == 'ade20k':
            semantic_branch_processor = SegformerFeatureExtractor.from_pretrained(
                "nvidia/segformer-b5-finetuned-ade-640-640")
            semantic_branch_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b5-finetuned-ade-640-640").to(rank)
        elif args.dataset == 'cityscapes' or args.dataset == 'foggy_driving':
            semantic_branch_processor = SegformerFeatureExtractor.from_pretrained(
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
            semantic_branch_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to(rank)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    print('[Model loaded] Semantic branch (your own segmentor) is loaded.')
    if args.dataset == 'ade20k':
        filenames = [fn_ for fn_ in os.listdir(args.data_dir) if '.jpg' in fn_ or '.png' in fn_]
    elif args.dataset == 'cityscapes' or args.dataset == 'foggy_driving':
        sub_folders = [fn_ for fn_ in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, fn_))]
        filenames = []
        for sub_folder in sub_folders:
            filenames += [os.path.join(sub_folder, fn_.replace('.png', '')) for fn_ in os.listdir(args.data_dir + sub_folder) if '.png' in fn_]
    local_filenames = filenames[(len(filenames) // args.world_size + 1) * rank : (len(filenames) // args.world_size + 1) * (rank + 1)]
    print('[Image name loaded] get image filename list.')
    print('[SSA start] model inference starts.')

    for i, file_name in enumerate(local_filenames):
        print('[Runing] ', i, '/', len(local_filenames), ' ', file_name, ' on rank ', rank, '/', args.world_size)
        img = img_load(args.data_dir, file_name, args.dataset)
        if args.dataset == 'ade20k':
            id2label = CONFIG_ADE20K_ID2LABEL
        elif args.dataset == 'cityscapes' or args.dataset == 'foggy_driving':
            id2label = CONFIG_CITYSCAPES_ID2LABEL
        else:
            raise NotImplementedError()
        with torch.no_grad():
            semantic_segment_anything_inference(file_name, args.out_dir, rank, img=img, save_img=args.save_img,
                                   semantic_branch_processor=semantic_branch_processor,
                                   semantic_branch_model=semantic_branch_model,
                                   mask_branch_model=mask_branch_model,
                                   dataset=args.dataset,
                                   id2label=id2label,
                                   model=args.model)
        # torch.cuda.empty_cache()



if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if args.world_size > 1:
        mp.spawn(main,args=(args,),nprocs=args.world_size,join=True)
    else:
        main(0, args)