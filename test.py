from PIL import Image
import matplotlib.pyplot as plt
import json
from vlm_train import LlavaSequenceClassificationDataset

def concatenate_images(image_paths, resize=(128, 128)):
    images = [Image.open(p).convert("RGB").resize(resize) for p in image_paths]
    total_width = sum(img.size[0] for img in images)
    max_height = max(img.size[1] for img in images)
    new_img = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    return new_img


# Just show a single image sequence from llava_input.json
with open("llava_input.json", "r") as f:
    all_data = json.load(f)
first_entry = all_data[0]
image_paths = first_entry["images"][:50]
concat_img = concatenate_images(image_paths, resize=(128, 128))

plt.figure(figsize=(20, 5))
plt.imshow(concat_img)
plt.axis('off')
plt.show()

with open("llava_input.json", "r") as f:
    data = json.load(f)

image_paths = sample['image_paths'][:50]
concat_img = concatenate_images(image_paths, resize=(128, 128))

plt.figure(figsize=(20, 5))
plt.imshow(concat_img)
plt.axis('off')
plt.show()

# Show first data point details
print("Prompt:", data[0]["prompt"])
print("Expected label:", data[0]["expected"])
print("Collision object:", data[0]["collision_object"])
print("Image paths:", data[0]["images"][:20])