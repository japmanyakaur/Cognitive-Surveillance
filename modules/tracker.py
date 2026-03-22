# modules/tracker.py
import supervision as sv

class PersonTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,  # lower = picks up person faster
            lost_track_buffer=60,             # frames to remember a lost person (2 sec at 30fps)
            minimum_matching_threshold=0.8,   # higher = stricter ID matching, less ID switching
            frame_rate=25
        )

    def update(self, detections):
        return self.tracker.update_with_detections(detections)