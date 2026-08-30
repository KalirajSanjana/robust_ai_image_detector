from PIL import Image, ImageFilter, ImageEnhance
import io
import random
import numpy as np

def jpeg_compression(image, quality):
    ## quality = 90, 70, 50, 30
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    return compressed_image.convert("RGB")

def gaussian_blur(image, sigma):
    ## sigma = 0.5, 1.0 or 2.0
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))

def resize_image(image, scale):
    ## scale = 0.5 or 0.25
    original_width, original_height = image.size
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    small_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    resized_image = small_image.resize((original_width, original_height), Image.Resampling.LANCZOS)
    return resized_image

def gaussian_noise(image, sigma):
    ##sigma = 0.02, 0.05, 0.10
    image_array = np.asarray(image).astype(np.float32) / 255.0
    noise = np.random.normal(loc=0.0, scale=sigma, size=image_array.shape)
    noisy_image = image_array + noise
    noisy_image = np.clip(noisy_image, 0.0, 1.0)
    noisy_image = (noisy_image * 255).astype(np.uint8)
    return Image.fromarray(noisy_image).convert("RGB")

def colour_jitter(image):
    ##random modification of brightness, contrast, saturation by +- 20%
    brightness_factor = random.uniform(0.8, 1.2)
    contrast_factor = random.uniform(0.8, 1.2)
    saturation_factor = random.uniform(0.8, 1.2)
    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    image = ImageEnhance.Color(image).enhance(saturation_factor)
    return image

def center_crop(image, crop_ratio=0.8):
    ##crop to 80%
    width, height = image.size
    new_width = int(width * crop_ratio)
    new_height = int(height * crop_ratio)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    return image.crop((left, top, right, bottom))

def apply_random_transform(image):

    if random.random() < 0.5:
        return image

    transform_type = random.choice([
        "jpeg",
        "blur",
        "resize",
        "noise",
        "colour",
        "crop"
    ])

    if transform_type == "jpeg":
        quality = random.choice([90, 70, 50, 30])
        image = jpeg_compression(image, quality)

    elif transform_type == "blur":
        sigma = random.choice([0.5, 1.0, 2.0])
        image = gaussian_blur(image, sigma)

    elif transform_type == "resize":
        scale = random.choice([0.5, 0.25])
        image = resize_image(image, scale)

    elif transform_type == "noise":
        sigma = random.choice([0.02, 0.05, 0.10])
        image = gaussian_noise(image, sigma)

    elif transform_type == "colour":
        image = colour_jitter(image)

    elif transform_type == "crop":
        image = center_crop(image, crop_ratio=0.8)

    return image