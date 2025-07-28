import cv2
import json
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import pickle
from make_trie import Letter_Node

# ----- CONFIG -----
MODEL_PATH = "boggle_model.pth"
CLASS_MAP_PATH = "boggle_classes.json"
IMG_SIZE = 64
MARGIN = 0.16

MIN_WORD_LEN = 3

# ----- Load model (1-channel version) -----
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256), torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Load class map
with open(CLASS_MAP_PATH, 'r') as f:
    idx_to_class = json.load(f)
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}

# Define transform for binary image input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # single-channel
])

# Load model
model = SimpleCNN(num_classes=len(idx_to_class))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# -------------------- Helper Functions --------------------

def crop_center_margin(img):
    h, w = img.shape[:2]
    y1 = int(h * MARGIN)
    y2 = int(h * (1 - MARGIN))
    x1 = int(w * MARGIN)
    x2 = int(w * (1 - MARGIN))
    return img[y1:y2, x1:x2]

def threshold_letter_area(cell_bgr):
    """
    Returns a binary mask where white = letter, black = background.
    Works on Boggle tiles with black/blue/green text on light backgrounds.
    """
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)

    # Mask for die ink by filtering out low saturation
    lower_letter = np.array([0, 100, 0])
    upper_letter = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower_letter, upper_letter)

    # Clean up
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.dilate(mask, kernel, iterations=2)

    return mask

def classify_tile(cell_img_bgr):
    """Classify a single tile (BGR image)."""
    cropped = crop_center_margin(cell_img_bgr)
    binary_mask = threshold_letter_area(cropped)  # shape: [H, W]

    resized_mask = cv2.resize(binary_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    input_img = resized_mask
    input_tensor = torch.tensor(input_img / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    input_tensor = (input_tensor - 0.5) / 0.5  # Normalize

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        return idx_to_class[pred_idx], binary_mask

def split_image_into_grid(image, rows=4, cols=4):
    h, w = image.shape[:2]
    cell_h = h / rows
    cell_w = w / cols
    grid = []
    for i in range(rows):
        row = []
        for j in range(cols):
            y1 = int(round(i * cell_h))
            y2 = int(round((i + 1) * cell_h))
            x1 = int(round(j * cell_w))
            x2 = int(round((j + 1) * cell_w))
            cell = image[y1:y2, x1:x2]
            row.append(cell)
        grid.append(row)
    return grid

# ---------------- Board Calculation -----------

def neighbor_indices(i, j):
    return [
        (i, j + 1),
        (i, j - 1),
        (i + 1, j),
        (i + 1, j + 1),
        (i + 1, j - 1),
        (i - 1, j),
        (i - 1, j + 1),
        (i - 1, j - 1)
    ]

def is_safe(i, j, board_size):
    return i >= 0 and i < board_size and j >= 0 and j < board_size

def search_recursive(board, node: Letter_Node, vis, i, j, cur_word: str, words):
    vis[i][j] = True
    cur_word += board[i][j]

    if len(cur_word) >= MIN_WORD_LEN and node.is_terminal:
        words.add(cur_word)

    children = node.children

    if len(children) == 0:
        vis[i][j] = False
        return

    for m, n in neighbor_indices(i, j):
        if is_safe(m, n, len(board)) and not vis[m][n]:
            tile = board[m][n]
            if tile == "qu":
                print("tile is qu")
                if "q" in children and "u" in children["q"].children:
                    search_recursive(
                        board, children["q"].children["u"], vis, m, n, cur_word, words
                    )
            elif tile in children:
                search_recursive(
                    board, children[tile], vis, m, n, cur_word, words
                )

    vis[i][j] = False
    return

def find_all_words(board, trie_root):
    words = set()
    board_size = len(board)
    vis = [[False for _ in range(board_size)] for _ in range(board_size)]

    for i in range(board_size):
        for j in range(board_size):
            tile = board[i][j]
            if tile == "qu":
                q_node = trie_root.get_child("q")
                if q_node:
                    u_node = q_node.get_child("u")
                    if u_node:
                        search_recursive(board, u_node, vis, i, j, "", words)
            else:
                child = trie_root.get_child(tile)
                if child:
                    search_recursive(board, child, vis, i, j, "", words)

    return list(words)

def score(words):
    score = 0
    for w in words:
        size = len(w)
        match size:
            case _ if size < 3:
                continue
            case 3 | 4:
                score += 1
            case 5:
                score += 2
            case 6:
                score += 3
            case 7:
                score += 5
            case _:
                score += 11

    return score

# -------------------- Main --------------------

# if __name__ == "__main__":
#     board = [
#         ['qu', 'e', 'w', 'j'],
#         ['y', 's', 'e', 'i'],
#         ['t', 'e', 'n', 'w'],
#         ['l', 'e', 'f', 'i']
#     ]
#     board_size = 4

#     trie_root = None
#     with open("trie.pkl", 'rb') as f:
#         trie_root = pickle.load(f)

#     # image = cv2.imread(r"test_images\tb1.jpg")
#     # if image is None:
#     #     print("Could not load image.")
#     #     exit()

#     # grid = split_image_into_grid(image)
    
#     # board = [[None for _ in range(board_size)] for _ in range(board_size)]

#     # for i, row in enumerate(grid):
#     #     for j, cell in enumerate(row):
#     #         letter, _ = classify_tile(cell)
#     #         board[i][j] = letter.lower()

#     print("Board:", board)
    
#     words = find_all_words(board, trie_root)
#     words.sort(key=len, reverse=True)
#     full_score = score(words)

#     print("Words sorted by length:", words, "\n\nFor a total score of", full_score)
