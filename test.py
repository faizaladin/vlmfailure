from PIL import Image, UnidentifiedImageError

try:
    img = Image.open('paired_frames/pos_0.9999999999999998_head_42/frame_00002.png')
    print("Image is valid.")
    print(img.size)  # (width, height)
except UnidentifiedImageError:
    print("Image is not valid or cannot be opened.")