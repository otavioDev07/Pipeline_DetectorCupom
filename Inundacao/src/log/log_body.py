import json


class LogBody:
    def __init__(self, message: str):
        self.message = message
        self.timer = 'seconds'
        self.config = {}

        self.error: str | None = None
        self.fail: bool = False
        self.accepted: bool | None = None
        self.sharpness_factor: float | None = None

        self.download: float | None = None
        self.decode_image: float | None = None
        self.filter_process: float | None = None
        self.encode_image: float | None = None
        self.upload: float | None = None

        self.filters_applied: list[str] = []

    def json(self):
        return {
            "message": json.loads(self.message),
            "timer": self.timer,
            "config": self.config,
            "error": self.error,
            "fail": self.fail,
            "accepted": self.accepted,
            "sharpness_factor": self.sharpness_factor,
            "download": self.download,
            "decode_image": self.decode_image,
            "filter_process": self.filter_process,
            "encode_image": self.encode_image,
            "upload": self.upload,
            "filters_applied": self.filters_applied,
        }

    def small_json(self):
        return self.json()


