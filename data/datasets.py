import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
import os
import datetime


ImageFile.LOAD_TRUNCATED_IMAGES = True

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


def binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))

    dset = datasets.ImageFolder(
            root,
            transforms.Compose([
                rz_func,
                # transforms.Lambda(lambda img: frequency_domain(img, opt)),
                transforms.Lambda(lambda img: data_augment(img, opt)),
                transforms.Lambda(lambda img: frequency_domain(img, opt)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]))
    return dset


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])


def frequency_domain(img, opt):
    """
    Apply frequency domain transformations to an image based on options.

    Args:
    img (PIL.Image): The input image.
    opt: Options containing flags for different transformations.

    Returns:
    PIL.Image: The transformed image.
    """
    # Convert PIL image to numpy array
    img_np = np.array(img)

    def generate_filename(prefix):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{prefix}_{timestamp}.png"

    # Image.fromarray(img_np).save(generate_filename("basic"))

    # Apply transformations based on options
    if opt.high_pass:
        img_np = apply_high_pass(img_np, opt.thresh)
        # Image.fromarray(img_np).save(generate_filename("high_pass"))
    if opt.low_pass:
        img_np = apply_low_pass(img_np)
        # Image.fromarray(img_np).save(generate_filename("low_pass"))
    if opt.edge:
        img_np = apply_edge_detection(img_np)
        # Image.fromarray(img_np).save(generate_filename("edge_detection"))
    if opt.sharp_edge:
        img_np = apply_edge_detection(img_np)
        # Image.fromarray(img_np).save(generate_filename("sharp_edge"))



    # Apply frequency transformations
    if opt.fft:
        img_np = apply_fft(img_np)
        # Image.fromarray(img_np).save(generate_filename("fft"))
    elif opt.dct:
        img_np = apply_dct(img_np)
        # Image.fromarray(img_np).save(generate_filename("dct"))

    # Convert numpy array back to PIL Image and save final result
    final_image = Image.fromarray(img_np)
    # final_image.save(generate_filename("final"))

    return final_image

def apply_high_pass(img, threshold): # Maybe threshold the high pass
    """
    Apply high-pass filtering to an image.

    Args:
    img (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The high-pass filtered image.
    """
    # Apply median blur
    median_blurred = cv2.medianBlur(img, 5)

    # Perform high pass filtering
    hp_img = cv2.subtract(img, median_blurred)

    if threshold > 0:
        # Thresholded approach
        _, thresh_img = cv2.threshold(hp_img, threshold, 255, cv2.THRESH_BINARY)
        # non_zero_positions = (binary_image > 0).astype(int)
    else:
        thresh_img = hp_img

    return thresh_img

def apply_low_pass(img):
    """
    Apply low-pass (median blur) filtering to an image.

    Args:
    img (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The low-pass filtered image.
    """
    # Apply median blur
    return cv2.medianBlur(img, 5)

def apply_edge_detection(img):
    """
    Apply edge detection to an image.

    Args:
    img (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The edge-detected image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection (e.g., using Canny)
    edges = cv2.Canny(gray, 100, 200)

    # Replicate the edges across three channels if needed
    edges_3channel = cv2.merge([edges, edges, edges])

    return edges_3channel

def apply_sharp_edge_detection(image):
    """
    Apply sharpening to an image using a kernel that enhances edges.
    """

    # Create a sharpening kernel, other kernels can be used for different effects
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)

    
    return apply_edge_detection(sharpen_image)


def apply_fft(img):
    """
    Apply Fast Fourier Transform to an image.

    Args:
    img (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The FFT transformed image.
    """
    # Split the image into its Red, Green and Blue channels
    red_channel, green_channel, blue_channel = cv2.split(img)

    # Apply FFT to each channel
    red_fft = fft_on_channel(red_channel)
    green_fft = fft_on_channel(green_channel)
    blue_fft = fft_on_channel(blue_channel)

    # Merge the channels back
    img_fft = cv2.merge([red_fft, green_fft, blue_fft])
    return img_fft

def fft_on_channel(channel):
    """
    Apply FFT on a single channel of an image.

    Args:
    channel (numpy.ndarray): Single channel of an image.

    Returns:
    numpy.ndarray: FFT transformed channel.
    """
    # Apply FFT
    dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Get magnitude spectrum
    magnitude = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1) # Adding 1 to avoid log(0)

    # Normalize the magnitude
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Handle any potential NaNs or Infs
    magnitude = np.nan_to_num(magnitude)

    # Ensure all values are within the 0-255 range and cast to uint8
    magnitude = np.clip(magnitude, 0, 255).astype('uint8')
    
    return magnitude

def apply_dct(img):
    """
    Apply Discrete Cosine Transform to an image's three channels.

    Args:
    img (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The DCT transformed image.
    """
    # Split the image into its Red, Green, and Blue channels
    red_channel, green_channel, blue_channel = cv2.split(img)

    # Apply DCT to each channel
    red_dct = dct_on_channel(red_channel)
    green_dct = dct_on_channel(green_channel)
    blue_dct = dct_on_channel(blue_channel)

    # Merge the channels back
    img_dct = cv2.merge([red_dct, green_dct, blue_dct])
    return img_dct

def dct_on_channel(channel):
    """
    Apply DCT on a single channel of an image.

    Args:
    channel (numpy.ndarray): Single channel of an image.

    Returns:
    numpy.ndarray: DCT transformed channel.
    """
    # Perform DCT on channel
    dct_transformed = cv2.dct(np.float32(channel)/255.0) * 255.0

    # Normalize and cast to uint8
    dct_normalized = cv2.normalize(dct_transformed, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(dct_normalized)
