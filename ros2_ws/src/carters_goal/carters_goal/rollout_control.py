from __future__ import annotations

from geometry_msgs.msg import Pose, PoseArray


ROLLOUT_FRAME_PREFIX = "rollout:"


def parse_rollout_id(frame_id: str) -> int | None:
    if not frame_id.startswith(ROLLOUT_FRAME_PREFIX):
        return None

    try:
        return int(frame_id[len(ROLLOUT_FRAME_PREFIX) :])
    except ValueError:
        return None


def build_rollout_control_message(rollout_id: int, flat_pose_array: list[float]) -> PoseArray:
    if rollout_id < 0:
        raise ValueError(f"rollout_id must be >= 0, got {rollout_id}.")
    if flat_pose_array and len(flat_pose_array) % 7 != 0:
        raise ValueError(
            "Rollout control pose arrays must contain 7 values per robot "
            f"(x, y, z, qx, qy, qz, qw). Got {len(flat_pose_array)} values."
        )

    msg = PoseArray()
    msg.header.frame_id = f"{ROLLOUT_FRAME_PREFIX}{rollout_id}"
    msg.poses = []

    for offset in range(0, len(flat_pose_array), 7):
        pose = Pose()
        pose.position.x = float(flat_pose_array[offset + 0])
        pose.position.y = float(flat_pose_array[offset + 1])
        pose.position.z = float(flat_pose_array[offset + 2])
        pose.orientation.x = float(flat_pose_array[offset + 3])
        pose.orientation.y = float(flat_pose_array[offset + 4])
        pose.orientation.z = float(flat_pose_array[offset + 5])
        pose.orientation.w = float(flat_pose_array[offset + 6])
        msg.poses.append(pose)

    return msg


def pose_array_to_flat_list(msg: PoseArray) -> list[float]:
    flat_pose_array: list[float] = []
    for pose in msg.poses:
        flat_pose_array.extend(
            [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ]
        )
    return flat_pose_array
