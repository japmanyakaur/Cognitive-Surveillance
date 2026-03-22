# modules/behaviour.py
import numpy as np
from modules.timestamp import frame_to_time, frames_to_seconds
import config

def get_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)


class BehaviorAnalyzer:
    def __init__(self, fps):
        self.fps = fps
        self.person_log = {}
        self.object_log = {}    # renamed from bag_log
        self.alerts = []

    def update(self, frame_num, tracked_persons, objects):
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
                }
            else:
                self.person_log[tracker_id]["last_position"] = center

        # --- Associate objects with nearest person ---
        for obj_idx, obj_box in enumerate(objects.xyxy):
            obj_center = get_center(obj_box)

            # Get object class label
            cls_id = int(objects.class_id[obj_idx])
            obj_label = objects.data.get("class_name", {})
            # fallback — store class_id, resolve label in main
            obj_class_id = cls_id

            min_dist = float("inf")
            closest_id = None
            for pid, pdata in self.person_log.items():
                if pid not in active_ids:
                    continue
                d = euclidean(obj_center, pdata["last_position"])
                if d < min_dist:
                    min_dist = d
                    closest_id = pid

            if obj_idx not in self.object_log:
                self.object_log[obj_idx] = {
                    "last_center": obj_center,
                    "owner_id": closest_id,
                    "placed_frame": frame_num,
                    "placed_time": current_time,
                    "class_id": obj_class_id,
                    "abandoned": False,
                }
            else:
                self.object_log[obj_idx]["last_center"] = obj_center

            if min_dist < config.DISTANCE_THRESHOLD:
                self.object_log[obj_idx]["owner_id"] = closest_id
                self.object_log[obj_idx]["placed_frame"] = frame_num

        # --- Check for abandoned objects ---
        new_alerts = []
        for obj_idx, obj_data in self.object_log.items():
            if obj_data["abandoned"]:
                continue

            owner_id = obj_data["owner_id"]
            if owner_id is None:
                continue

            owner_active = owner_id in active_ids
            if owner_active:
                owner_pos = self.person_log[owner_id]["last_position"]
                dist = euclidean(obj_data["last_center"], owner_pos)
            else:
                dist = float("inf")

            time_unattended = frames_to_seconds(
                frame_num - obj_data["placed_frame"], self.fps
            )

            if dist > config.DISTANCE_THRESHOLD and time_unattended > config.TIME_THRESHOLD_SECONDS:
                self.object_log[obj_idx]["abandoned"] = True

                entry_time = self.person_log[owner_id]["entry_time"] \
                    if owner_id in self.person_log else "unknown"

                alert = {
                    "obj_idx": obj_idx,
                    "class_id": obj_data["class_id"],
                    "owner_id": owner_id,
                    "entry_time": entry_time,
                    "placed_time": obj_data["placed_time"],
                    "alert_time": current_time,
                    "obj_center": obj_data["last_center"].tolist(),
                    "time_unattended_sec": round(time_unattended, 1),
                }
                self.alerts.append(alert)
                new_alerts.append(alert)

        return new_alerts

    def build_behavior_log(self, alert, object_label="object"):
        """Build natural language log — object_label passed in from main."""
        return (
            f"Person ID {alert['owner_id']} entered the scene at {alert['entry_time']}. "
            f"They left a {object_label} unattended at {alert['placed_time']}. "
            f"The person moved away and did not return. "
            f"The {object_label} was left unattended for {alert['time_unattended_sec']} seconds. "
            f"An abandoned object alert was triggered at {alert['alert_time']}."
        )