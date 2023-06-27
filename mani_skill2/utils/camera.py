from collections import OrderedDict
import numpy as np
import cv2


def resize_obs_images(obs: OrderedDict, new_shape,
                      interpolation=cv2.INTER_NEAREST_EXACT) -> OrderedDict:
    """Resize observation images into shape=(width, height)"""
    new_width, new_height = new_shape

    for cam_name in obs["camera_param"]:
        # Update intrinsics
        K = obs["camera_param"][cam_name]["intrinsic_cv"]

        camera_capture = obs["image"][cam_name]
        if "Color" in camera_capture:
            old_height, old_width = camera_capture["Color"].shape[:2]
        else:
            old_height, old_width = camera_capture["rgb"].shape[:2]

        resize_trans = np.array([[new_width / old_width, 0, 0],
                                 [0, new_height / old_height, 0],
                                 [0, 0, 1]], dtype=K.dtype)
        obs["camera_param"][cam_name]["intrinsic_cv"] = resize_trans @ K

        # Resize images
        for key in camera_capture:
            if key in ["Color", "Position", "rgb", "depth"]:
                camera_capture[key] = cv2.resize(
                    camera_capture[key],
                    new_shape, interpolation=interpolation
                )
            elif key == "Segmentation":
                camera_capture[key] = cv2.resize(
                    camera_capture[key].astype(np.uint16),
                    new_shape, interpolation=interpolation
                ).astype(np.uint32)
            else:
                raise ValueError(f"Unknown key {key}")
    return obs
