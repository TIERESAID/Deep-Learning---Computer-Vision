import cv2
import numpy as np
import time
import functools

def time_it(func):
    """
    Simple decorator to measure function execution time.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.4f} seconds")
        
        return result
    return wrapper

@time_it
def find_way_from_maze(image: np.ndarray) -> tuple:
    """
    Find path through the maze.
    :param image: maze image
    :return: path coordinates from the maze as (x, y), where x and y are coordinate arrays
    """
    a = 0  # side of the traversal square
    for i in range(image.shape[1]):
        if (image[0, i] == np.array([255, 255, 255])).all():
            a += 1
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
    
    contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw = np.zeros(image.shape)
    cv2.drawContours(draw, contours, 0, (0, 0, 255), 0)
    
    kernel = np.ones((a, a))
    img_dilation = cv2.dilate(draw, kernel, iterations=2, borderType=cv2.BORDER_CONSTANT)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=2, borderType=cv2.BORDER_CONSTANT)
    diff = cv2.absdiff(img_dilation, img_erosion)
    
    x = np.where(np.all(diff == np.array([0, 0, 255]), axis=-1))[0]
    y = np.where(np.all(diff == np.array([0, 0, 255]), axis=-1))[1]
    coords = (x, y)
    
    return coords

