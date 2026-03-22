import supervision as sv

class PersonTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=120,        # remember person for 120 frames
            minimum_matching_threshold=0.8,
            frame_rate=25
        )

    def update(self, detections):
        return self.tracker.update_with_detections(detections)