import os
import cv2
import numpy as np
import string
from glob import glob

# ----- CONFIG -----
INPUT_DIR = "test_images"  # folder with full board images
ROWS, COLS = 4, 4
MARGIN = 0.16
IMG_SIZE = 64

SAVE_DIR = "labeled_tiles"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---- Globals ----
current_tile_index = 0
labeled_data = []
tile_images = []
tile_metadata = []

# ---- Helper functions ----
def crop_center_margin(img):
    h, w = img.shape[:2]
    y1 = int(h * MARGIN)
    y2 = int(h * (1 - MARGIN))
    x1 = int(w * MARGIN)
    x2 = int(w * (1 - MARGIN))
    return img[y1:y2, x1:x2]

def threshold_letter_area(cell_bgr):
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    lower_letter = np.array([0, 100, 0])
    upper_letter = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_letter, upper_letter)
    return mask

def split_image_into_grid(image, rows=4, cols=4):
    h, w = image.shape[:2]
    cell_h = h / rows
    cell_w = w / cols
    grid = []
    for i in range(rows):
        for j in range(cols):
            y1 = int(round(i * cell_h))
            y2 = int(round((i + 1) * cell_h))
            x1 = int(round(j * cell_w))
            x2 = int(round((j + 1) * cell_w))
            cell = image[y1:y2, x1:x2]
            grid.append((cell, i, j))
    return grid

def load_all_tiles():
    global tile_images, tile_metadata
    all_images = sorted(glob(os.path.join(INPUT_DIR, "*.jpg")) + glob(os.path.join(INPUT_DIR, "*.png")))
    for image_path in all_images:
        img = cv2.imread(image_path)
        if img is None:
            continue
        grid = split_image_into_grid(img, ROWS, COLS)
        for cell, i, j in grid:
            tile_images.append(cell)
            tile_metadata.append((image_path, i, j))

def show_current_tile():
    global current_tile_index
    if current_tile_index < 0 or current_tile_index >= len(tile_images):
        return
    cell = tile_images[current_tile_index]
    cropped = crop_center_margin(cell)
    binary = threshold_letter_area(cropped)
    resized = cv2.resize(binary, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    cv2.imshow("Label Tile", resized)

def save_label(label):
    global current_tile_index
    cell = tile_images[current_tile_index]
    cropped = crop_center_margin(cell)
    binary = threshold_letter_area(cropped)
    resized_binary = cv2.resize(binary, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    label_dir = os.path.join(SAVE_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    for k, angle in enumerate([0, 90, 180, 270]):
        rot = cv2.rotate(resized_binary, {0: cv2.ROTATE_90_CLOCKWISE, 90: cv2.ROTATE_180, 180: cv2.ROTATE_90_COUNTERCLOCKWISE, 270: None}[angle]) if angle != 270 else resized_binary.copy()
        out_path = os.path.join(label_dir, f"{label}_{current_tile_index:04d}_{k}.png")
        cv2.imwrite(out_path, rot)

    labeled_data.append((current_tile_index, label))
    current_tile_index += 1
    show_current_tile()

def skip_tile():
    global current_tile_index
    print(f"Skipped tile {current_tile_index}")
    current_tile_index += 1
    show_current_tile()

# ---- Main loop ----
if __name__ == "__main__":
    load_all_tiles()
    show_current_tile()

    print("\n--- Controls ---")
    print("Press A-Z or Q for 'QU' to label tile.")
    print("Backspace to go back. ESC to exit.")
    print("Delete to skip tile.")

    while True:
        key = cv2.waitKey(0)

        # ESC
        if key == 27:
            break

        # Backspace
        elif key == 8:
            current_tile_index = max(0, current_tile_index - 1)
            show_current_tile()

        # Delete (skip tile)
        elif key == 3014656 or key == 127:  # handle both common delete keycodes
            skip_tile()

        # Letter keys
        elif 65 <= key <= 90 or 97 <= key <= 122:
            letter = chr(key).upper()
            if letter == 'Q':
                letter = 'QU'
            save_label(letter)

        # Out of tiles
        if current_tile_index >= len(tile_images):
            print("\nLabeling complete.")
            break

    cv2.destroyAllWindows()
