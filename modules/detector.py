from ultralytics import YOLO
import supervision as sv
import config

class ObjectDetector:
    def __init__(self):
        self.model = YOLO(config.YOLO_MODEL)

    def detect(self, frame):
        results = self.model(
            frame,
            conf=config.YOLO_CONFIDENCE,
            verbose=False
            # No 'classes' filter — detects all 80 COCO classes
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections

    def split(self, detections):
        """Persons are class 0. Everything else is a potential abandoned object."""
        persons = detections[detections.class_id == config.PERSON_CLASS_ID]
        objects = detections[detections.class_id != config.PERSON_CLASS_ID]
        return persons, objects