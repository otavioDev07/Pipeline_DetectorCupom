from src.filters.filter_sharpness import FilterSharpness
from src.log.log_body import LogBody


class FilterService:

    def __init__(self):
        self.filters = [
            FilterSharpness(),
        ]

    def process(self, image, log_body: LogBody):
        result = image
        for f in self.filters:
            result, log_body = f.apply(result, log_body)
        return result, log_body
