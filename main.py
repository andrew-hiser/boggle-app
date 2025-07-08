import numpy as np
import easyocr
import cv2

def find_words():
    return None

def find_largest_square_contour(gray_img):
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort by area, largest first
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
        
    raise ValueError("No square board found.")

def process_board_image(image):
    # Grayscale the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the largest square contour
    try:
        square_contour = find_largest_square_contour(gray)
        
        # Draw the detected square on the original image
        result_img = image.copy()
        cv2.drawContours(result_img, [square_contour], -1, (0, 255, 0), 3)
        
        # Draw corner points
        for point in square_contour:
            cv2.circle(result_img, tuple(point), 5, (255, 0, 0), -1)
        
        # Resize image for display if it's too large
        height, width = result_img.shape[:2]
        max_size = 800  # Maximum width or height
        
        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            result_img = cv2.resize(result_img, (new_width, new_height))
        
        cv2.imshow('Detected Board', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return square_contour
        
    except ValueError as e:
        print(e)
        return None
    
def debug_all_contours(image):
    """Visualize all contours found in the image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Total contours found: {len(contours)}")
    
    # Create debug image
    debug_img = image.copy()
    
    # Draw all contours in different colors and analyze them
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Approximate to polygon
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Color code by number of sides
        if len(approx) == 4:
            color = (0, 255, 0)  # Green for 4-sided (squares/rectangles)
            thickness = 3
        elif len(approx) == 3:
            color = (255, 0, 0)  # Blue for triangles
            thickness = 2
        else:
            color = (0, 0, 255)  # Red for everything else
            thickness = 1
            
        cv2.drawContours(debug_img, [cnt], -1, color, thickness)
        
        # Print info for larger contours
        if area > 1000:  # Only show significant contours
            print(f"Contour {i}: {len(approx)} sides, area={area:.0f}, perimeter={perimeter:.0f}")
    
    # Resize for display
    height, width = debug_img.shape[:2]
    max_size = 800
    if width > max_size or height > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        debug_img = cv2.resize(debug_img, (new_width, new_height))
        thresh = cv2.resize(thresh, (new_width, new_height))
    
    # Save images instead of showing (to avoid the display error)
    cv2.imwrite('debug_all_contours.jpg', debug_img)
    cv2.imwrite('debug_threshold.jpg', thresh)
    
    print("Debug images saved:")
    print("- debug_all_contours.jpg (Green=4-sided, Blue=triangles, Red=other)")
    print("- debug_threshold.jpg (binary threshold image)")
    
    return contours

def find_dice_adaptive(image):
    """Find dice using adaptive methods - no shape assumptions"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Multiple preprocessing approaches
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try adaptive threshold instead of OTSU
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 10)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dice_candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Just filter by area, not shape
        if 200 < area < 8000:  # Adjust based on your image
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Very loose aspect ratio - dice can be quite distorted
            aspect_ratio = w / h
            if 0.3 < aspect_ratio < 3.0:  # Much more permissive
                dice_candidates.append((x, y, w, h))
    
    return dice_candidates

def visualize_dice_candidates(image):
    """Show all potential dice regions"""
    candidates = find_dice_adaptive(image)
    result = image.copy()
    
    print(f"Found {len(candidates)} potential dice regions")
    
    for i, (x, y, w, h) in enumerate(candidates):
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        print(f"Region {i}: size {w}x{h}, area {w*h}")
    
    cv2.imwrite('dice_candidates.jpg', result)
    return candidates

if __name__ == "__main__":
    image = cv2.imread(r"C:\CodeProjects\Boggle App\test_images\board1.png")
    if image is not None:
        # Visualize dice candidates
        candidates = visualize_dice_candidates(image)
        
        # Debug all contours first
        contours = debug_all_contours(image)
        
        # Then try normal processing
        process_board_image(image)
    else:
        print("Could not load image")