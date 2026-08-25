import cv2

from src.filters.base_filter import BaseFilter
from src.log.log_body import LogBody


class FilterClahe(BaseFilter):

    @property
    def name(self) -> str:
        return 'clahe'

    def apply(self, image, log_body: LogBody):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge([l_clahe, a, b])
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        log_body.filters_applied.append(self.name)

        return result, log_body
