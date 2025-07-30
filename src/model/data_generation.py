import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import random

# Configuration
OUTPUT_DIR = "boggle_data_binary"
FONT_PATH = "font.ttf"  # Replace if needed
LETTERS = [chr(i) for i in range(65, 91) if chr(i) != 'Q'] + ['QU']
FULL_IMG_SIZE = 128
CROP_SIZE = 64
IMAGES_PER_CLASS = 400
ROTATIONS = [0, 90, 180, 270]
TEXT_SIZE_SINGLE = 60
TEXT_SIZE_QU = 45
TRANSLATION_RANGE = 10  # px (in full image)


def make_dirs():
    for letter in LETTERS:
        os.makedirs(os.path.join(OUTPUT_DIR, letter), exist_ok=True)


def render_die(letter):
    img = Image.new('L', (FULL_IMG_SIZE, FULL_IMG_SIZE), 255)
    draw = ImageDraw.Draw(img)

    # Use smaller font for "Qu"
    font_size = TEXT_SIZE_QU if letter == "Qu" else TEXT_SIZE_SINGLE
    font = ImageFont.truetype(FONT_PATH, size=font_size)

    bbox = font.getbbox(letter)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pos = ((FULL_IMG_SIZE - w) // 2 - bbox[0], (FULL_IMG_SIZE - h) // 2 - bbox[1])

    # Draw bold letter
    offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for ox, oy in offsets:
        draw.text((pos[0] + ox, pos[1] + oy), letter, fill=0, font=font)

    # Add disambiguation mark for M/W/Z
    if letter.upper() in ['M', 'W', 'Z']:
        line_width = int(w * 0.85)
        underline_y = pos[1] + h + 21
        line_x1 = pos[0] + (w - line_width) // 2
        line_x2 = line_x1 + line_width
        draw.line([(line_x1, underline_y), (line_x2, underline_y)], fill=0, width=3)

    return img


def augment_and_crop(img_pil, base_angle):
    # Rotate
    jitter_angle = np.clip(np.random.normal(0, 2.5), -5, 5)
    angle = base_angle + jitter_angle
    rotated = img_pil.rotate(angle, resample=Image.BICUBIC, expand=False)
    img_np = np.array(rotated, dtype=np.uint8)

    # Translate
    dx = int(np.clip(np.random.normal(0, 1.5), -TRANSLATION_RANGE, TRANSLATION_RANGE))
    dy = int(np.clip(np.random.normal(0, 1.5), -TRANSLATION_RANGE, TRANSLATION_RANGE))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img_np = cv2.warpAffine(img_np, M, (FULL_IMG_SIZE, FULL_IMG_SIZE), borderValue=255)

    # Crop
    start = (FULL_IMG_SIZE - CROP_SIZE) // 2
    cropped = img_np[start:start + CROP_SIZE, start:start + CROP_SIZE]

    # Binary threshold so letter = white (255), background = black (0)
    _, binary = cv2.threshold(cropped, 127, 255, cv2.THRESH_BINARY_INV)

    return binary


def main():
    make_dirs()
    for letter in tqdm(LETTERS, desc="Generating clean binary"):
        for i in range(IMAGES_PER_CLASS):
            base_letter = "Qu" if letter == "QU" else letter
            img = render_die(base_letter)
            angle = random.choice(ROTATIONS)
            final_img = augment_and_crop(img, angle)
            out_path = os.path.join(OUTPUT_DIR, letter, f"{i:04d}.png")
            cv2.imwrite(out_path, final_img)


if __name__ == "__main__":
    main()
