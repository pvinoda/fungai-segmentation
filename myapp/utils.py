import base64

import numpy as np
import colorsys
import cv2
from scipy.ndimage import find_objects, gaussian_filter, generate_binary_structure, label, maximum_filter1d, \
    binary_fill_holes
import numpy as np
from skimage.morphology import disk, binary_erosion, binary_dilation
import torch
from skimage.measure import label
import cv2
import skimage
import matplotlib.pyplot as plt
import cv2
from cv2 import resize
import logging
import pathlib
import sys
import os
from PIL import Image
import io


def colorize_mask(mask, rgba_color):
    """Apply an RGBA color to non-zero pixels in a binary mask.

    Parameters:
    - mask: 2D numpy array representing a binary mask.
    - rgba_color: Tuple of 4 values representing the desired RGBA color.

    Returns:
    - 3D numpy array with shape (height, width, 4) representing an RGBA image.
    """
    # Initialize an empty RGBA image of the same size as the mask
    rgba_image = np.zeros((*mask.shape, 4), dtype=np.uint8)

    # Assign the specified RGBA color to non-zero pixels in the mask
    rgba_image[mask > 0] = rgba_color

    return rgba_image


def overlay_masks_on_image(phase_img, masks, is_contour, alpha=100):
    # Predefined colors with transparency
    colors = [
        [255, 0, 0, alpha],
        [0, 0, 255, alpha],
        [0, 255, 0, alpha],
        [255, 255, 0, alpha],
        [0, 255, 255, alpha],
        [255, 218, 0, alpha]
    ]

    # Convert the 2D numpy array to a PIL Image
    phase_image_pil = Image.fromarray(convertGreyToRGB(phase_img)).convert("RGBA")

    for color_idx, mask in enumerate(masks):
        color = colors[color_idx].copy()

        if is_contour[color_idx]:
            color[-1] = 255
        # Convert binary mask to a PIL Image
        mask_binary = binarize(mask) * 255
        colored_mask = Image.fromarray(colorize_mask(mask_binary, color))
        phase_image_pil.paste(colored_mask, (0, 0), Image.fromarray(mask_binary))

    result_array = np.asarray(phase_image_pil, dtype=np.uint8)
    return result_array


def write_images_to_dir(path, ims, extension=".tif"):
    dir = os.path.dirname(path)
    if os.path.exists(path):
        return
    os.mkdir(path)
    if "." not in extension:
        extension = "." + extension
    for i, im in enumerate(ims):
        name = f"im_{i}{extension}"
        fname = os.path.join(path, name)
        skimage.io.imsave(fname, im)


def save_image(path, im, extension=".tif"):
    pass


def overlay_images(phase_img, mask_img, colorized_mask):
    # Convert numpy arrays to PIL images
    mask = mask_img.copy()
    mask[mask > 0] = 255
    microscopy_pil = Image.fromarray(convertGreyToRGB(phase_img))
    mask_pil = Image.fromarray(mask)
    colorized_pil = Image.fromarray(colorized_mask)

    # Convert images to 'RGBA' mode
    microscopy_pil = microscopy_pil.convert("RGBA")
    colorized_pil = colorized_pil.convert("RGBA")
    mask_pil = mask_pil.convert("L")

    # Overlay images
    microscopy_pil.paste(colorized_pil, (0, 0), mask_pil)

    # Convert PIL image back to numpy array
    result = np.array(microscopy_pil)

    return result


def get_frames_with_no_cells(masks):
    no_cell_frames = []
    for i, mask in enumerate(masks):
        if np.all(mask == 0):
            no_cell_frames.append(i)
    return no_cell_frames


def get_end_of_cells(masks):
    for i in reversed(range(masks.shape[0])):
        if np.any(masks[i] != 0):
            return i


def get_filename(path):
    _, name_with_extension = os.path.split(path)
    return os.path.splitext(path)[0]


def logger_setup():
    cp_dir = pathlib.Path.home().joinpath('.yeastvision')
    cp_dir.mkdir(exist_ok=True)
    log_file = cp_dir.joinpath('run.log')
    try:
        log_file.unlink()
    except:
        print('creating new log file')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f'WRITING LOG OUTPUT TO {log_file}')
    return logger, log_file


def check_gpu(do_torch=True):
    if do_torch:
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        d = {"total": t,
             "reserved": r,
             "allocated": a}
        print(f"Pytorch\n_________\n{d}")
        # free inside reserved


def showCellNums(mask):
    "annotates the current plt figure with cell numbers"
    # cell_vals = np.unique(mask[mask!=0]).astype(int)
    cell_vals = np.unique(mask).astype(int)
    cell_vals = np.delete(cell_vals, np.where(cell_vals == 0))

    for val in cell_vals:
        # print("cell val: " + str(val)) #test
        x, y = getCenter(mask, val)
        plt.annotate(str(val), xy=(x, y), ha='center', va='center')


def getCenter(mask, cell_val):
    '''
    takes random points within a cell to estimate its center. Used to find the x,y coordinates where the cell number
    annotation text should be displated
    '''
    y, x = (mask == cell_val).nonzero()
    sample = np.random.choice(len(x), size=20, replace=True)
    meanx = np.mean(x[sample])
    meany = np.mean(y[sample])

    return int(round(meanx)), int(round(meany))


def rescaleByFactor(factor, ims):
    row, col = ims[0].shape
    newrow, newcol = int(row * factor), int(col * factor)
    print(newrow, newcol)
    return rescaleBySize((newrow, newcol), ims)


def rescaleBySize(newshape, ims):
    row, col = newshape
    return [resize(im, (col, row), interpolation=cv2.INTER_CUBIC) for im in ims]


def binarize_relabel(mask):
    return label(binarize(mask))


def binarize(mask):
    return (mask > 0).astype(np.uint8)


def check_torch_gpu(gpu_number=0):
    try:
        device = torch.device('cuda:' + str(gpu_number))
        _ = torch.zeros([1, 2, 3]).to(device)
        print('** TORCH CUDA version installed and working. **')
        return True
    except:
        print('TORCH CUDA version not installed/working.')
        return False


def count_objects(labeledMask):
    return len(np.unique(labeledMask)) - 1


def capitalize(string):
    first_letter = string[0]
    rest_of_string = string[1:]
    return first_letter.capitalize() + rest_of_string


def get_mask_contour(mask):
    out_mask = np.zeros_like(mask)
    for cellnum in np.unique(mask):
        if cellnum > 0:
            cell_contour = get_cell_contour(mask, cellnum)
            out_mask[cell_contour == 1] = cellnum
    return out_mask


def get_cell_contour(mask, cellnum):
    out_mask = np.zeros_like(mask)
    mask_binary = (mask == cellnum).astype(np.uint8)
    in_mask = shrink_bud(mask_binary, kernel_size=2)
    mask_binary[in_mask] = 0
    return mask_binary


def shrink_bud(bud_mask, footprint=None, kernel_size=2):
    if not footprint:
        footprint = disk(kernel_size)
    return binary_erosion(bud_mask, footprint)


def enlarge_bud(bud_mask, footprint=None, kernel_size=4):
    if not footprint:
        footprint = disk(kernel_size)
    return binary_dilation(bud_mask, footprint)


def normalize_im(im_o, clip=True):
    """
    Normalizes a given image such that the values range between 0 and 1.     
    
    Parameters
    ---------- 
    im : 2d-array
        Image to be normalized.
    clip: boolean
        Whether or not to set im values <0 to 0
        
    Returns
    -------
    im_norm: 2d-array
        Normalized image ranging from 0.0 to 1.0. Note that this is now
        a floating point image and not an unsigned integer array. 
    """
    im = np.nan_to_num(im_o)
    if clip:
        im[im < 0] = 0
    if im.max() == 0:
        return im.astype(np.float32)
    im_norm = (im - im.min()) / (im.max() - im.min())
    im_norm[np.isnan(im_o)] = np.nan
    return im_norm.astype(np.float32)


def convertGreyToRGB(im):
    image = skimage.util.img_as_ubyte(normalize_im(im))
    image_3D = cv2.merge((image, image, image))
    # return cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    return image_3D
    # rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
    # rgba[:, :, 3] = 1
    # return rgba


def get_img_from_fig(fig):
    '''
    input: matplotlib pyplot figure object
    output: a 3D numpy array object corresponding to the image shown by that figure
    '''
    import io as i
    buf = i.BytesIO()
    fig.savefig(buf, format="tif", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return (np.array(img)).astype("uint8")


def overlay(im, mask, color=[255, 0, 0], get_fig=False, image_only=True):
    '''
    Overlays an image and with its binary mask for easily visualizing
    
    
    Parameters
    --------
    image: grayscale ndarray of shape mxn
    mask: binary ndaraay of shape mxn
    get_fig (bool): return the matplotlib figure as a np array
    color (list): color given to mask in [R,G,B] form
    image_only (bool): return only the merged images as a RGB ndarray 
    '''
    image = skimage.util.img_as_ubyte(normalize_im(im))
    image_3D = cv2.merge((image, image, image))
    mask_plot = image_3D.copy()
    mask_plot[mask == 1] = color  # set thresholded pixels to red

    if image_only:
        return mask_plot

    fig = plt.figure()
    plt.axis('off')
    plt.imshow(image_3D)
    plt.imshow(mask_plot)
    plt.show()

    if get_fig:
        return get_img_from_fig(fig)
    else:
        return mask_plot


def process_tiff_image(image_array):
    if image_array.ndim > 4:
        image_array = image_array.squeeze()
        if image_array.ndim > 4:
            raise ValueError("Cannot load 4D stack, reduce dimensions")
    elif image_array.ndim == 1:
        raise ValueError("Cannot load 1D stack, increase dimensions")

    if image_array.ndim == 4:
        # Handle 3D stack (modify based on your needs)
        # You can raise an error, process each slice separately, etc.
        raise NotImplementedError("3D stack processing not implemented yet")
    elif image_array.ndim == 3:
        # Assume smallest dimension is channels and put last
        c = np.array(image_array.shape).argmin()
        image_array = image_array.transpose(((c + 1) % 3, (c + 2) % 3, c))
    elif image_array.ndim == 2:
        # Add a channel if necessary
        image_array = image_array[..., np.newaxis]

    if image_array.shape[-1] > 3:
        print("WARNING: image has more than 3 channels, keeping only first 3")
        image_array = image_array[..., :3]
    elif image_array.shape[-1] == 2:
        # Fill in with blank channels to make 3 channels
        shape = image_array.shape
        image_array = np.concatenate(
            (image_array, np.zeros((*shape[:-1], 3 - shape[-1]), dtype=np.uint8)), axis=-1)
    elif image_array.shape[-1] == 1:
        pass  # Handle single channel image if needed

    # Convert to float32 and normalize (optional)
    image_array = image_array.astype(np.float32)
    image_min = image_array.min()
    image_max = image_array.max()
    if image_max > image_min + 1e-3:
        image_array = (image_array - image_min) / (image_max - image_min)
    image_array *= 255

    # You can now use the processed image_array for further calculations
    #  or save it to a new file (consider security)

    # Example: Calculate basic statistics
    num_channels = image_array.shape[-1]
    min_value = image_array.min()
    max_value = image_array.max()

    print(num_channels, min_value, max_value, image_array.shape)

    return num_channels, min_value, max_value, image_array


def normalize99(Y, lower=1, upper=99, copy=True):
    """
    Normalize the image so that 0.0 corresponds to the 1st percentile and 1.0 corresponds to the 99th percentile.

    Args:
        Y (ndarray): The input image.
        lower (int, optional): The lower percentile. Defaults to 1.
        upper (int, optional): The upper percentile. Defaults to 99.
        copy (bool, optional): Whether to create a copy of the input image. Defaults to True.

    Returns:
        ndarray: The normalized image.
    """
    X = Y.copy() if copy else Y
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    if x99 - x01 > 1e-3:
        X = (X - x01) / (x99 - x01)
    else:
        X[:] = 0
    return X


def hsv_to_rgb(arr):
    hsv_to_rgb_channels = np.vectorize(colorsys.hsv_to_rgb)
    h, s, v = np.rollaxis(arr, axis=-1)
    r, g, b = hsv_to_rgb_channels(h, s, v)
    rgb = np.stack((r, g, b), axis=-1)
    return rgb


def rgb_to_hsv(arr):
    rgb_to_hsv_channels = np.vectorize(colorsys.rgb_to_hsv)
    r, g, b = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv_channels(r, g, b)
    hsv = np.stack((h, s, v), axis=-1)
    return hsv


def mask_overlay(img, masks, colors=None):
    """Overlay masks on image (set image to grayscale).

    Args:
        img (int or float, 2D or 3D array): Image of size [Ly x Lx (x nchan)].
        masks (int, 2D array): Masks where 0=NO masks; 1,2,...=mask labels.
        colors (int, 2D array, optional): Size [nmasks x 3], each entry is a color in 0-255 range.

    Returns:
        RGB (uint8, 3D array): Array of masks overlaid on grayscale image.
    """
    if colors is not None:
        if colors.max() > 1:
            colors = np.float32(colors)
            colors /= 255
        colors = rgb_to_hsv(colors)
    if img.ndim > 2:
        img = img.astype(np.float32).mean(axis=-1)
    else:
        img = img.astype(np.float32)

    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:, :, 2] = np.clip((img / 255. if img.max() > 1 else img) * 1.5, 0, 1)
    hues = np.linspace(0, 1, masks.max() + 1)[np.random.permutation(masks.max())]
    for n in range(int(masks.max())):
        ipix = (masks == n + 1).nonzero()
        if colors is None:
            HSV[ipix[0], ipix[1], 0] = hues[n]
        else:
            HSV[ipix[0], ipix[1], 0] = colors[n, 0]
        HSV[ipix[0], ipix[1], 1] = 1.0
    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB


def image_to_rgb(img0, channels=[0, 0]):
    """Converts image from 2 x Ly x Lx or Ly x Lx x 2 to RGB Ly x Lx x 3.

    Args:
        img0 (ndarray): Input image of shape 2 x Ly x Lx or Ly x Lx x 2.

    Returns:
        ndarray: RGB image of shape Ly x Lx x 3.

    """
    img = img0.copy()
    img = img.astype(np.float32)
    if img.ndim < 3:
        img = img[:, :, np.newaxis]
    if img.shape[0] < 5:
        img = np.transpose(img, (1, 2, 0))
    if channels[0] == 0:
        img = img.mean(axis=-1)[:, :, np.newaxis]
    for i in range(img.shape[-1]):
        if np.ptp(img[:, :, i]) > 0:
            img[:, :, i] = np.clip(normalize99(img[:, :, i]), 0, 1)
            img[:, :, i] = np.clip(img[:, :, i], 0, 1)
    img *= 255
    img = np.uint8(img)
    RGB = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    if img.shape[-1] == 1:
        RGB = np.tile(img, (1, 1, 3))
    else:
        RGB[:, :, channels[0] - 1] = img[:, :, 0]
        if channels[1] > 0:
            RGB[:, :, channels[1] - 1] = img[:, :, 1]
    return RGB


def masks_to_outlines(masks):
    """Get outlines of masks as a 0-1 array.

    Args:
        masks (int, 2D or 3D array): Size [Ly x Lx] or [Lz x Ly x Lx], where 0=NO masks and 1,2,...=mask labels.

    Returns:
        outlines (2D or 3D array): Size [Ly x Lx] or [Lz x Ly x Lx], where True pixels are outlines.
    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError("masks_to_outlines takes 2D or 3D array, not %dD array" % masks.ndim)
    outlines = np.zeros(masks.shape, bool)

    if masks.ndim == 3:
        for i in range(masks.shape[0]):
            outlines[i] = masks_to_outlines(masks[i])
        return outlines
    else:
        slices = find_objects(masks.astype(int))
        for i, si in enumerate(slices):
            if si is not None:
                sr, sc = si
                mask = (masks[sr, sc] == (i + 1)).astype(np.uint8)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T
                vr, vc = pvr + sr.start, pvc + sc.start
                outlines[vr, vc] = 1
        return outlines


def show_segmentation(fig, img, maski, flowi, channels=[0, 0], file_name=None):
    # use existing

    img0 = img.copy()

    if img0.shape[0] < 4:
        img0 = np.transpose(img0, (1, 2, 0))
    if img0.shape[-1] < 3 or img0.ndim < 3:
        img0 = image_to_rgb(img0, channels=channels)
    else:
        if img0.max() <= 50.0:
            img0 = np.uint8(np.clip(img0, 0, 1) * 255)

    outlines = masks_to_outlines(maski)

    overlay = mask_overlay(img0, maski)

    outX, outY = np.nonzero(outlines)
    imgout = img0.copy()
    imgout[outX, outY] = np.array([255, 0, 0])  # pure red

    return overlay, imgout, flowi


def show_seg_yeast(img, maski, channels=[0, 0]):
    img0 = img.copy()

    if img0.shape[0] < 4:
        img0 = np.transpose(img0, (1, 2, 0))
    if img0.shape[-1] < 3 or img0.ndim < 3:
        img0 = image_to_rgb(img0, channels=channels)
    else:
        if img0.max() <= 50.0:
            img0 = np.uint8(np.clip(img0, 0, 1) * 255)

    # outlines = masks_to_outlines(maski)

    overlay = mask_overlay(img0, maski)

    # outX, outY = np.nonzero(outlines)
    # imgout = img0.copy()
    # imgout[outX, outY] = np.array([255, 0, 0])  # pure red

    return overlay


def convert_array_into_blob(image_array: np.array):
    image_PIL = Image.fromarray(image_array)
    # Create an in-memory byte stream buffer
    byte_buffer = io.BytesIO()

    # Save the PIL image to the byte buffer
    image_PIL.save(byte_buffer, format='PNG')

    # Get the value of the byte buffer
    byte_buffer.seek(0)
    byte_buffer_value = byte_buffer.getvalue()

    return byte_buffer_value


def convert_array_to_base64(image_array: np.array):
    image_PIL = Image.fromarray(image_array)
    buffer = io.BytesIO()
    image_PIL.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
