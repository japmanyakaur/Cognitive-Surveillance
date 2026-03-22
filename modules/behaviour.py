# modules/behavior.py
import numpy as np
from modules.timestamp import frame_to_time, frames_to_seconds
import config

def get_center(box):
    """Get center point of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)


class BehaviorAnalyzer:
    def __init__(self, fps):
        self.fps = fps

        # person_id -> { entry_frame, last_position, associated_bag }
        self.person_log = {}

        # bag_id -> { last_center, owner_id, placed_frame, abandoned }
        self.bag_log = {}

        # Completed alert events
        self.alerts = []

    def update(self, frame_num, tracked_persons, bags):
        """
        Call this every frame.
        tracked_persons: sv.Detections with tracker_id
        bags: sv.Detections (untracked is fine)
        """
        current_time = frame_to_time(frame_num, self.fps)

        # --- Update person positions ---
        active_ids = set()
        for i, tracker_id in enumerate(tracked_persons.tracker_id):
            box = tracked_persons.xyxy[i]
            center = get_center(box)
            active_ids.add(tracker_id)

            if tracker_id not in self.person_log:
                self.person_log[tracker_id] = {
                    "entry_frame": frame_num,
                    "entry_time": current_time,
                    "last_position": center,
                    "associated_bag_idx": None,
                }
            else:
                self.person_log[tracker_id]["last_position"] = center

        # --- Associate bags with nearest person ---
        for bag_idx, bag_box in enumerate(bags.xyxy):
            bag_center = get_center(bag_box)

            # Find closest person
            min_dist = float("inf")
            closest_id = None
            for pid, pdata in self.person_log.items():
                if pid not in active_ids:
                    continue
                d = euclidean(bag_center, pdata["last_position"])
                if d < min_dist:
                    min_dist = d
                    closest_id = pid

            if bag_idx not in self.bag_log:
                self.bag_log[bag_idx] = {
                    "last_center": bag_center,
                    "owner_id": closest_id,
                    "placed_frame": frame_num,
                    "placed_time": current_time,
                    "abandoned": False,
                }
            else:
                self.bag_log[bag_idx]["last_center"] = bag_center

            # Update owner if person is still nearby
            if min_dist < config.DISTANCE_THRESHOLD:
                self.bag_log[bag_idx]["owner_id"] = closest_id
                self.bag_log[bag_idx]["placed_frame"] = frame_num  # reset timer

        # --- Check for abandoned bags ---
        new_alerts = []
        for bag_idx, bag_data in self.bag_log.items():
            if bag_data["abandoned"]:
                continue

            owner_id = bag_data["owner_id"]
            if owner_id is None:
                continue

            owner_active = owner_id in active_ids
            if owner_active:
                owner_pos = self.person_log[owner_id]["last_position"]
                dist = euclidean(bag_data["last_center"], owner_pos)
            else:
                dist = float("inf")  # Person left the frame

            time_unattended = frames_to_seconds(
                frame_num - bag_data["placed_frame"], self.fps
            )

            if dist > config.DISTANCE_THRESHOLD and time_unattended > config.TIME_THRESHOLD_SECONDS:
                self.bag_log[bag_idx]["abandoned"] = True

                entry_time = self.person_log[owner_id]["entry_time"] if owner_id in self.person_log else "unknown"
                alert = {
                    "bag_idx": bag_idx,
                    "owner_id": owner_id,
                    "entry_time": entry_time,
                    "placed_time": bag_data["placed_time"],
                    "alert_time": current_time,
                    "bag_center": bag_data["last_center"].tolist(),
                }
                self.alerts.append(alert)
                new_alerts.append(alert)

        return new_alerts

    def build_behavior_log(self, alert):
        """Build a text log to send to the LLM."""
        return (
            f"Person ID {alert['owner_id']} entered the scene at {alert['entry_time']}. "
            f"They placed a bag at {alert['placed_time']}. "
            f"The person moved away from the bag and the bag was left unattended. "
            f"An abandoned object alert was triggered at {alert['alert_time']}."
        )