import nbimporter
import cv2, math, os
import numpy as np
import matplotlib.pyplot as plt
from utils import get_parameters, Gauss2D, filter_image_vec
from detect_edges import edge_detection_nms

def visualize_houghlines(image_name, constants):
    image_rgb = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
    print("-" * 50 + "\n" + "Original Image:")
    plt.imshow(image_rgb); plt.show() # Displaying the sample image
    
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    image_m, image_o, image_x, image_y = edge_detection_nms(image, constants.sigma)
    
    print("-" * 50 + "\n" + "Edge Image:")
    plt.imshow(image_m, cmap="gray"); plt.show() # Displaying the sample image
    
    image_m_thres = 1.0*(image_m > constants.thres) # threshold the edge magnitude image
    print("-" * 50 + "\n" + "Thresholded Edge Image:")
    plt.imshow(image_m_thres, cmap="gray"); plt.show() # Displaying the sample image
    
    #--------------hough transform----------------
    H, rho_arr, theta_arr = hough_transform(image_m, constants.thres, constants.rho_res, constants.theta_res)   
    peak_rho_arr, peak_theta_arr = peak_hough_lines(H, rho_arr, theta_arr, constants.num_lines)
    
    #--------------vis using infinitely long lines----------------------------
    vis_line_len = 1000 # len of line in pixels, big enough to span the image
    vis_image_rgb = np.copy(image_rgb)
    for (rho, theta) in zip(peak_rho_arr, peak_theta_arr):
        x0 = rho*np.cos(theta); y0 = rho*np.sin(theta)
        x1 = int(x0 - vis_line_len*np.sin(theta)); y1 = int(y0 + vis_line_len*np.cos(theta))
        x2 = int(x0 + vis_line_len*np.sin(theta)); y2 = int(y0 - vis_line_len*np.cos(theta)); 
        cv2.line(vis_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    #---------------------------------------------
    print("-" * 50 + "\n" + "Edge Image:")
    plt.imshow(vis_image_rgb); plt.show() # Displaying the sample image
    
    return

def hough_transform(image_m, thres, rho_res, theta_res):
    """Compute Hough Transform for the image

    Args:
        image_m: np.array, HxW, edge magnitude image.
        thres: float, scalar to threshold image_m
        rho_res: integer, resolution of rho
        theta_res: integer, resolution of theta in degrees
        
    Returns:
        H: np.array, (num of rhos x num of thetas), hough transform accumulator (rho x theta), NOT theta x rho!
        rho_arr: np.array, dim=num of rhos, quantized rhos
        theta_arr: np.array, dim=num of thetas, quantized thetas
    """
    
    image_m_thres = 1.0*(image_m > thres) # threshold the edge magnitude image
    height, width = image_m_thres.shape # image height and width 
    diagonal_len = np.ceil(np.sqrt(height**2 + width**2)) # image diagonal = rho_max
      
    rho_max = diagonal_len
    # compute rho_arr, we go from [-rho_max to rho_max] in rho_res steps
    # rho_arr = ?
    # YOUR CODE HERE
    # print(rho_res)
    rho_arr = np.linspace(-rho_max, rho_max, num=math.floor(2*rho_max/rho_res)+1)
    # raise NotImplementedError()
    
    # compute theta_arr, we go from [0, pi] in theta_res steps, NOT [-pi/2, pi/2]!
    # Note theta_res is in degrees but theta_scale should be in radians [0, pi]
    # theta_arr = ?
    # YOUR CODE HERE
    theta_arr = np.linspace(0, 180, num=math.floor(180/theta_res)+1)
    theta_arr = theta_arr*(math.pi/180)
    # raise NotImplementedError()
    ## H is accumulator
    H = np.zeros((len(rho_arr), len(theta_arr)), dtype=np.int32)
    
    # find all edge (nonzero) pixel indexes
    y_idxs, x_idxs = image_m_thres.nonzero() 
    # print(image_m_thres.shape)
    # exit()
    # Putting the edge points into bins in parameter space
    for x, y in zip(x_idxs, y_idxs):
        for theta_idx, theta in enumerate(theta_arr):
            # compute rho_idx, note, theta is in radians!
            # Hint: compute rho first from theta, round it to nearest rho_prime in rho_arr
            # and then find rho_prime's rho_idx (index of rho_prime in rho_arr, NOT index of rho!)
            # rho_idx = ?
            # YOUR CODE HERE
            ## basically plotting the sinusiodal curves for every edge pixel
            rho = x*math.cos(theta) + y*math.sin(theta)
            rho_idx = np.digitize(rho, rho_arr)-1
            ## checking which half of the bin rho is in and assigning index accordingly
            if rho>=rho_arr[rho_idx] and rho<rho_arr[rho_idx+1]:
                if abs(rho-rho_arr[rho_idx])<=abs(rho-rho_arr[rho_idx+1]):
                    rho_idx = rho_idx
                else:
                    rho_idx = rho_idx+1
                    
            # raise NotImplementedError()

            
            H[rho_idx, theta_idx] += 1
    
    return H, rho_arr, theta_arr


def peak_hough_lines(H, rho_arr, theta_arr, num_lines):
    """Returns the rhos and thetas corresponding to top local maximas in the accumulator H

    Args:
        H: np.array, (num of rhos x num of thetas), hough transform accumulator
        rho_arr: np.array, dim=num of rhos, quantized rhos
        theta_arr: np.array, dim=num of thetas, quantized thetas
        num_lines: integer, number of lines we wish to detect in the image
        
    Returns:
        peak_rho_arr: np.array, dim=num_lines, top num_lines rhos by votes in the H
        peak_theta_arr: np.array, dim=num_lines, top num_lines thetas by votes in the H
    """
    
    # compute peak_rho_arr and peak_theta_arr
    # sort H using np.argsort, pick the top num_lines lines
    # peak_rho_arr = ?
    # peak_theta_arr = ?
    
    # YOUR CODE HERE
    top_line_idxs = np.argsort(H.flatten())[-num_lines:]
    org_idxs = np.unravel_index(top_line_idxs, (len(rho_arr), len(theta_arr)))
    peak_rho_arr = np.array([rho_arr[i] for i in org_idxs[0]])
    peak_theta_arr = np.array([theta_arr[i] for i in org_idxs[1]])

    # raise NotImplementedError()
    
    assert(len(peak_rho_arr) == num_lines)
    assert(len(peak_theta_arr) == num_lines)
    return peak_rho_arr, peak_theta_arr


def main():
  # Code goes here
  image_list, constants = get_parameters()
  image_idx = np.random.randint(0, len(image_list))
  visualize_houghlines(image_list[image_idx], constants)


if __name__ == "__main__":
  main()

    
    