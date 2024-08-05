# image_utils.py

from skimage import io as skio, img_as_ubyte, measure
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity, equalize_adapthist

def color_separate(ihc_rgb):
    ihc_hed = rgb2hed(ihc_rgb)
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = img_as_ubyte(hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1)))
    ihc_e = img_as_ubyte(hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1)))
    ihc_d = img_as_ubyte(hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1)))
    h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1), in_range=(0, np.percentile(ihc_hed[:, :, 0], 99)))
    d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1), in_range=(0, np.percentile(ihc_hed[:, :, 2], 99)))
    zdh = img_as_ubyte(np.dstack((null, d, h)))
    return ihc_h, ihc_e, ihc_d, zdh

def create_patches(image, patch_size):
    patches = []
    img_height, img_width = image.shape[:2]
    for i in range(0, img_height, patch_size):
        for j in range(0, img_width, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append((patch, i, j))
    return patches
