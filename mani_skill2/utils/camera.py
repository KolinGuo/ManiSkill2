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
        old_height, old_width = obs["image"][cam_name]["Color"].shape[:2]
        resize_trans = np.array([[new_width / old_width, 0, 0],
                                 [0, new_height / old_height, 0],
                                 [0, 0, 1]], dtype=K.dtype)
        obs["camera_param"][cam_name]["intrinsic_cv"] = resize_trans @ K

        # Resize images
        for key in obs["image"][cam_name]:
            if key in ["Color", "Position"]:
                obs["image"][cam_name][key] = cv2.resize(
                    obs["image"][cam_name][key],
                    new_shape, interpolation=interpolation
                )
            elif key == "Segmentation":
                obs["image"][cam_name][key] = cv2.resize(
                    obs["image"][cam_name][key].astype(np.uint16),
                    new_shape, interpolation=interpolation
                ).astype(np.uint32)
            else:
                raise ValueError(f"Unknown key {key}")
    return obs
