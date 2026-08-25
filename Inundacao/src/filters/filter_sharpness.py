import cv2
import numpy as np

from src.filters.base_filter import BaseFilter
from src.filters.tools_filter import normalize
from src.log.log_body import LogBody

THRESHOLD = 0.235


class FilterSharpness(BaseFilter):

    def __init__(self):
        self.mask_percentage = 0.6

    @property
    def name(self) -> str:
        return 'sharpness'

    def apply(self, image, log_body: LogBody):
        factor = self._fft_factor(image)
        log_body.sharpness_factor = factor
        log_body.accepted = factor > THRESHOLD
        log_body.filters_applied.append(self.name)
        return image, log_body

    def _fft_factor(self, image) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return float(np.mean(normalize(self._magnitude_spectrum(gray))))

    def _magnitude_spectrum(self, gray):
        fshift = np.fft.fftshift(np.fft.fft2(gray))
        h, w = fshift.shape
        cx, cy = w // 2, h // 2
        ch, cw = int(h * self.mask_percentage), int(w * self.mask_percentage)
        cut = fshift[cy - ch // 2:cy + ch // 2, cx - cw // 2:cx + cw // 2]
        return np.log(1 + np.abs(cut)) * 255
