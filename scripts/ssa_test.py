import os
from typing import Any, List, Optional
import cv2
import numpy as np

from server_wrapper import (
    ServerMixin,
    image_to_str,
    host_model,
    send_request,
    str_to_image,
)
class SSAClient:
    def __init__(self, port: int = 12185):
        self.url = f"http://localhost:{port}/ssa"

    def segment_bbox(self, image: np.ndarray) -> np.ndarray:
        response = send_request(self.url, image=image)
        seg_mask_str = response["seg_mask"]
        seg_mask_str = str_to_image(seg_mask_str)

        return seg_mask_str


if __name__ == "__main__":
    img = cv2.imread("test/hm3d.jpg", 1)
    client = SSAClient(port=12185)
    seg_mask = client.segment_bbox(img)
    cv2.imshow("seg_mask", seg_mask)
    cv2.waitKey(0)
