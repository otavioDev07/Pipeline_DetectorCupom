import cv2
import numpy as np
from rembg import remove

from src.filters.base_filter import BaseFilter
from src.log.log_body import LogBody


class FilterBackground(BaseFilter):

    @property
    def name(self) -> str:
        return 'background'

    def apply(self, image, log_body: LogBody):
        result = remove(image)  # retorna RGBA numpy array

        # composta os pixels transparentes em fundo branco e converte para BGR
        alpha = result[:, :, 3:4].astype(np.float32) / 255.0
        rgb = result[:, :, :3].astype(np.float32)
        white = np.full_like(rgb, 255.0)
        composited = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
        bgr = cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)

        log_body.filters_applied.append(self.name)
        return bgr, log_body
