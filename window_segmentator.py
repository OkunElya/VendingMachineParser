import cv2
import numpy as np
from ultralytics import YOLO
from shared import MODEL_PATHES 


def process_mask_matrix(mask_matrix):
    """
    Applies morphological transforms and edge detection on a binary mask matrix.
    """
    
    binary_mask = (mask_matrix * 255).astype(np.uint8)
    
    # Define structural elements for morphology
    # 5x5 rectangular kernel works well for closing window frame gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    # 1. Close internal holes and gaps in the mask
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 2. Smooth out noisy/jagged outer boundaries
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 3. Detect edges using Canny
    edges = cv2.Canny(cleaned_mask, 100, 200)
    
    # Find contours from the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.array([])
    
    # Return the largest contour found on the edge map
    return max(contours, key=cv2.contourArea)
    
def iterate_for_4_points(mask_contour, max_iterations=100, tolerance=1e-5):
    perimeter = cv2.arcLength(mask_contour, True)
    if perimeter == 0:
        return None

    low = 0.0
    high = 1.0
    
    best_approx = None
    best_distance_to_four = float('inf')

    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        epsilon = mid * perimeter
        approx = cv2.approxPolyDP(mask_contour, epsilon, True)
        num_points = len(approx)

        if num_points == 4:
            return approx.reshape(4, 2)
        
        distance = abs(num_points - 4)
        if distance < best_distance_to_four:
            best_distance_to_four = distance
            best_approx = approx

        if num_points > 4:
            low = mid
        else:
            high = mid

        if (high - low) < tolerance:
            break

    # Fallback: If 4 points couldn't be strictly resolved, return the closest match
    if best_approx is not None:
        return best_approx.reshape(-1, 2)
    
    return None

def reduce_to_4_points_by_area(points):
    """
    If an approximation yields > 4 points, greedily removes the point 
    that minimizes the loss of polygon area until 4 points remain.
    """
    pts = list(points)
    while len(pts) > 4:
        best_idx = -1
        max_area = -1
        
        # Test the remaining area if we drop point `i`
        for i in range(len(pts)):
            test_pts = pts[:i] + pts[i+1:]
            area = cv2.contourArea(np.array(test_pts, dtype=np.int32))
            if area > max_area:
                max_area = area
                best_idx = i
                
        pts.pop(best_idx)
        
    return np.array(pts, dtype=np.int32)

def to4PointPoly(mask):
    contour = np.int32([mask])

    points = iterate_for_4_points(contour)

    if points is None:
        return np.array([])
    if len(points) > 4:
        points = reduce_to_4_points_by_area(points)
    elif len(points) < 4:
        hull = cv2.convexHull(contour)
        points = reduce_to_4_points_by_area(hull.reshape(-1, 2))

    return points

def matrix_to_4_point_poly(masks_matrix):
    if masks_matrix is None or len(masks_matrix) == 0:
        return np.array([])

    # 2. Convert to standard CV_8UC1 format (0 or 255)
    binary_mask = (masks_matrix * 255).astype(np.uint8)
    
    # 3. Morphological Transforms
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    edges = cv2.Canny(cleaned_mask, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.array([])
    
    edge_contour = max(contours, key=cv2.contourArea)
    
    points = iterate_for_4_points(edge_contour)

    if points is None:
        hull = cv2.convexHull(edge_contour)
        return reduce_to_4_points_by_area(hull.reshape(-1, 2))
    if len(points) > 4:
        points = reduce_to_4_points_by_area(points)
    elif len(points) < 4 or points.size == 0:
        hull = cv2.convexHull(edge_contour)
        points = reduce_to_4_points_by_area(hull.reshape(-1, 2))

    return points


class WindowSegmentator(YOLO):
    def __init__(self):
        super().__init__(MODEL_PATHES["window_segmentator"])
    
    def getPoly(self,image,conf=0.25):
        results = self.predict(source=image, conf=conf,retina_masks=True,verbose = False)
        result = results[0]
        if result.masks is not None:
            masks_matrices = result.masks.data.cpu().numpy()
            largest_mask = max(masks_matrices, key=lambda m: m.astype(np.float32).mean())
                
            return matrix_to_4_point_poly(largest_mask)
    

                    


if __name__ == "__main__":
    image = cv2.imread("./datasets/window_segmentation/train/images/complex_02_0.jpg")
    segmentator = WindowSegmentator()
    poly = segmentator.getPoly(image)
    
    frame = image
    cv2.polylines(frame,[poly],True,(0,255,0))
    for point in poly:
        cv2.circle(frame,tuple(point.astype(np.int32)), 4,(0,255,0),-1)

    
    cv2.imshow("frame",frame)
    cv2.waitKey(-1)
    
    
    