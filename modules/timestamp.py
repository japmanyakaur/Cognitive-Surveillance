def frame_to_time(frame_num, fps):
    """Convert frame number to HH:MM:SS string."""
    total_seconds = int(frame_num / fps)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def frames_to_seconds(frame_num, fps):
    return frame_num / fps