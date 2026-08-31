from PIL import Image

from transform_images import (
    jpeg_compression,
    gaussian_blur,
    resize_image,
    gaussian_noise,
    colour_jitter,
    center_crop
)


# Load any image from your computer
image = Image.open("image.jpg").convert("RGB")

print("Original size:", image.size)


# Apply transformations
jpeg_image = jpeg_compression(image, quality=90)
blur_image = gaussian_blur(image, sigma=2.0)
resize_result = resize_image(image, scale=0.25)
noise_image = gaussian_noise(image, sigma=0.10)
color_image = colour_jitter(image)
crop_image = center_crop(image, crop_ratio=0.8)


# Save results
jpeg_image.save("output_jpeg.jpg")
blur_image.save("output_blur.jpg")
resize_result.save("output_resize.jpg")
noise_image.save("output_noise.jpg")
color_image.save("output_color.jpg")
crop_image.save("output_crop.jpg")


print("All transformations completed!")