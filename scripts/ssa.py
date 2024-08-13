# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, List, Optional

import numpy as np
import torch
import time
import pycocotools.mask as maskUtils
from segformer import segformer_segmentation as segformer_func
from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL

from server_wrapper import (
    ServerMixin,
    image_to_str,
    host_model,
    send_request,
    str_to_image,
)

# try:
    
# except ModuleNotFoundError:
#     print("Could not import mobile_sam. This is OK if you are only using the client.")


class SSA:
    def __init__(
        self,
        model_name: str = 'sam2',
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.device = device

            
        try:
            if model_name == 'sam':
                from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
                args.ckpt_path = 'ckp/sam_vit_h_4b8939.pth'
                model = sam_model_registry["vit_h"](checkpoint=args.ckpt_path).to(self.device)
                
            elif model_name == 'sam2':
                from sam2.build_sam import build_sam2
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator as SamAutomaticMaskGenerator
                # checkpoint = "ckp/sam2_hiera_tiny.pt"
                # model_cfg = "sam2_hiera_t.yaml"
                checkpoint = "ckp/sam2_hiera_large.pt"
                model_cfg = "sam2_hiera_l.yaml"
                model = build_sam2(model_cfg, checkpoint).to(self.device)
            elif model_name == 'mobile_sam':
                from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
                args.ckpt_path = '/home/chenx2/MobileSAM/weights/mobile_sam.pt'
                model = sam_model_registry["vit_t"](checkpoint=args.ckpt_path).to(self.device)

        except ModuleNotFoundError:
            print(f"Could not import {model_name}.")

        model.eval()
        
        self.mask_branch_model = SamAutomaticMaskGenerator(
            model=model,
            output_mode='coco_rle',
            crop_n_points_downscale_factor=2
        )
        
        from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
        self.semantic_branch_processor = SegformerFeatureExtractor.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640")
        self.semantic_branch_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640").to(self.device)


    def segment_bbox(self, image: np.ndarray) -> np.ndarray:
        """Segments the object in the given bounding box from the image.

        Args:
            image (numpy.ndarray): The input image as a numpy array.

        Returns:
            np.ndarray: 

        """
        
        with torch.no_grad():
            result = self.semantic_segment_anything_inference(img=image,
                                   semantic_branch_processor=self.semantic_branch_processor,
                                   semantic_branch_model=self.semantic_branch_model,
                                   mask_branch_model=self.mask_branch_model,
                                   dataset='ade20k',
                                   id2label=CONFIG_ADE20K_ID2LABEL)
        return result
    
    
    def semantic_segment_anything_inference(self, img=None,
                                 semantic_branch_processor=None,
                                 semantic_branch_model=None,
                                 mask_branch_model=None,
                                 dataset=None,
                                 id2label=None):
        st = time.time()
        anns = {'annotations': mask_branch_model.generate(img)}
        ed = time.time()
        print(f"Running time: {ed - st} seconds")
        h, w, _ = img.shape
        class_names = []

        class_ids = segformer_func(img, semantic_branch_processor, semantic_branch_model, self.device)
      
        semantc_mask = class_ids.clone()
        anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)
        for ann in anns['annotations']:
            valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
            # get the class ids of the valid pixels
            propose_classes_ids = class_ids[valid_mask]
            num_class_proposals = len(torch.unique(propose_classes_ids))
            if num_class_proposals == 1:
                semantc_mask[valid_mask] = propose_classes_ids[0]
                ann['class_name'] = id2label['id2label'][str(propose_classes_ids[0].item())]
                ann['class_proposals'] = id2label['id2label'][str(propose_classes_ids[0].item())]
                class_names.append(ann['class_name'])
                # bitmasks.append(maskUtils.decode(ann['segmentation']))
                continue
            top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
            top_1_propose_class_names = [id2label['id2label'][str(class_id.item())] for class_id in top_1_propose_class_ids]

            semantc_mask[valid_mask] = top_1_propose_class_ids
            ann['class_name'] = top_1_propose_class_names[0]
            ann['class_proposals'] = top_1_propose_class_names[0]
            class_names.append(ann['class_name'])
            # bitmasks.append(maskUtils.decode(ann['segmentation']))

            del valid_mask
            del propose_classes_ids
            del num_class_proposals
            del top_1_propose_class_ids
            del top_1_propose_class_names
        return semantc_mask.cpu().numpy()


class SSAClient:
    def __init__(self, port: int = 12185):
        self.url = f"http://localhost:{port}/ssa"

    def segment_bbox(self, image: np.ndarray) -> np.ndarray:
        response = send_request(self.url, image=image)
        seg_mask_str = response["seg_mask"]
        seg_mask_str = str_to_image(seg_mask_str, shape=tuple(image.shape[:2]))

        return seg_mask_str



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12185)
    parser.add_argument("--model_name", type=str, default="sam2")
    args = parser.parse_args()

    print("Loading model...")

    class SSAServer(ServerMixin, SSA):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            seg_mask = self.segment_bbox(image)
            seg_mask_str = image_to_str(seg_mask)
            return {"seg_mask": seg_mask_str}

    ssa = SSAServer(model_name="mobile_sam")
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(ssa, name="ssa", port=args.port)
