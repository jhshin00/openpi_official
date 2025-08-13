import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # Handle case where images are in [..., batch, channel, height, width] format
    # Only transpose if it's actually (C, H, W) format, not (H, W, C)
    if (images.shape[-1] in [height, width] and 
        images.shape[-2] in [height, width] and 
        images.shape[-3] == 3 and
        images.shape[-1] != images.shape[-2]):  # Make sure it's not already (H, W, C)
        # This looks like [..., batch, channel, height, width] format
        # Transpose to [..., batch, height, width, channel] format
        images = np.transpose(images, list(range(images.ndim - 4)) + [-3, -1, -2])
    
    # Handle chunk dimension: if we have (1, H, W, C) from chunk processing
    if (images.ndim == 4 and 
        images.shape[-4] == 1 and 
        images.shape[-3] in [height, width] and 
        images.shape[-2] in [height, width] and 
        images.shape[-1] == 3):
        # This is (1, H, W, C) from chunk processing, squeeze the first dimension
        images = images.squeeze(0)  # Remove the chunk dimension
    
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image
