from collections import OrderedDict
from typing import Tuple
import numpy as np
import cv2


def resize_obs_images(obs: OrderedDict, new_size: Tuple[int],
                      interpolation=cv2.INTER_NEAREST_EXACT) -> OrderedDict:
    """Resize observation images into size=(width, height)"""
    new_width, new_height = new_size

    image_obs = obs["image"]
    camera_param_obs = obs.get("camera_param")

    for cam_name, images in image_obs.items():
        # Resize images
        for key, image in images.items():
            old_shape = image.shape
            if key in ["Color", "Position", "rgb", "depth"]:
                resized_image = cv2.resize(
                    image,
                    new_size, interpolation=interpolation
                )
            elif key == "Segmentation":
                resized_image = cv2.resize(
                    image.astype(np.uint16),
                    new_size, interpolation=interpolation
                ).astype(np.uint32)
            else:
                raise ValueError(f"Unknown key {key}")
            # Keep the same number of channels as input
            images[key] = resized_image.reshape(new_size[::-1] + old_shape[2:])

        # Update intrinsics
        if camera_param_obs is not None:
            K = camera_param_obs[cam_name]["intrinsic_cv"]
            old_height, old_width = old_shape[:2]
            resize_trans = np.array([[new_width / old_width, 0, 0],
                                    [0, new_height / old_height, 0],
                                    [0, 0, 1]], dtype=K.dtype)
            camera_param_obs[cam_name]["intrinsic_cv"] = resize_trans @ K

    return obs
