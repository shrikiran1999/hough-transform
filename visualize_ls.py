## DO NOT MODIFY! 
## Import from previous notebook
import nbimporter
from utils import cv2, np, plt, math, SimpleNamespace
from utils import get_parameters, Gauss2D, filter_image_vec
from detect_edges import edge_detection_nms
from detect_houghlines import hough_transform, peak_hough_lines

image_list, constants = get_parameters()

#----------------------------------------------------------------------
# Different from visualize in p3, calls hough_accumulator_nms()
def visualize_ls(image_name, constants):
    image_rgb = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
    print("-" * 50 + "\n" + "Original Image:")
    plt.imshow(image_rgb); plt.show() # Displaying the sample image
    
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    image_m, image_o, image_x, image_y = edge_detection_nms(image, constants.sigma)
    print(np.any(image_m==0))
    
    print("-" * 50 + "\n" + "Edge Image:")
    plt.imshow(image_m, cmap="gray"); plt.show() # Displaying the sample image
    
    image_m_thres = 1.0*(image_m > constants.thres) # threshold the edge magnitude image
    print("-" * 50 + "\n" + "Thresholded Edge Image:")
    plt.imshow(image_m_thres, cmap="gray"); plt.show() # Displaying the sample image
    y_idxs, x_idxs = image_m_thres.nonzero() 
    print(x_idxs, y_idxs)
    
    #--------------hough transform----------------
    H, rho_arr, theta_arr = hough_transform(image_m, constants.thres, constants.rho_res, constants.theta_res)
    H = hough_accumulator_nms(H) # nms on H
    peak_rho_arr, peak_theta_arr = peak_hough_lines(H, rho_arr, theta_arr, constants.num_lines)
    vis_image_rgb = np.copy(image_rgb)
    vis_line_len = 3
    # 0.1 for pot
    for (x0, y0) in zip(x_idxs, y_idxs):
        for (rho, theta) in zip(peak_rho_arr, peak_theta_arr):
            if (abs(rho - (x0*np.cos(theta)+y0*np.sin(theta)))<0.1):
                x1 = int(x0 - vis_line_len*np.sin(theta)); y1 = int(y0 + vis_line_len*np.cos(theta))
                x2 = int(x0 + vis_line_len*np.sin(theta)); y2 = int(y0 - vis_line_len*np.cos(theta)); 
                cv2.line(vis_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    plt.imshow(vis_image_rgb); plt.show() # Displaying the sample image

    return 


def hough_accumulator_nms(H):
    """Compute Hough Transform for the image

    Args:
        image_m: np.array, HxW, edge magnitude image.
        
    Returns:
        image_m_prime: np.array, HxW, suppressed edge magnitude image.
    """
    H_prime = np.copy(H) 
    H_pad = np.pad(H, 1)
    neighbor_offsets = [(dy, dx) for dy in range(-1, 2) for dx in range(-1, 2) if (dy != 0 or dx != 0)]
    # print(len(neighbor_offsets))
    ## directions are [-1,-1], [-1,1], [1,-1], [1,1]
    
    # compute supression mask per neighbour, 1 to suppress, 0 to keep
    # compare H and a part of H_pad, the part of H_pad can be obtained by moving H_pad using the neighbor_offsets
    # suppress_masks_per_neighbor = [? for (dy, dx) in neighbor_offsets]
    # YOUR CODE HERE
    # raise NotImplementedError()
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            suppress_masks_per_neighbor = [1 if H_pad[i+1+dx, j+1+dy]>H[i,j] else 0 for (dy, dx) in neighbor_offsets]
            if np.amax(suppress_masks_per_neighbor)==1:
                H_prime[i,j] = 0


    # suppress_mask = np.logical_or.reduce(suppress_masks_per_neighbor) # 1 to suppress, 0 to keep
    # H_prime[suppress_mask] = 0
    
    return H_prime


def hough_egde_match(image_m, thres, rho_res, theta_res, peak_rho_arr, peak_theta_arr, rho_arr, theta_arr):
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
    
    # find all edge (nonzero) pixel indexes
    y_idxs, x_idxs = image_m_thres.nonzero() 
    # Putting the edge points into bins in parameter space
    for x, y in zip(x_idxs, y_idxs):
        for theta_idx, theta in enumerate(theta_arr):
            rho = x*math.cos(theta) + y*math.sin(theta)
            rho_idx = np.digitize(rho, rho_arr)-1
            ## checking which half of the bin rho is in and assigning index accordingly
            if rho>=rho_arr[rho_idx] and rho<rho_arr[rho_idx+1]:
                if abs(rho-rho_arr[rho_idx])<=abs(rho-rho_arr[rho_idx+1]):
                    rho_idx = rho_idx
                else:
                    rho_idx = rho_idx+1

            if (rho_idx, theta_idx) not in zip(peak_rho_arr, peak_theta_arr):
                image_m_thres[y, x]=0

    
    return image_m_thres


# from skimage.draw import line

def visualize_line_segments(image_name, constants, vis_image_rgb, rho_arr, theta_arr, peak_rho_arr, peak_theta_arr):
    
    image_rgb = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
    print("-" * 50 + "\n" + "Original Image:")
    plt.imshow(image_rgb); plt.show() # Displaying the sample image
    
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    image_m, image_o, image_x, image_y = edge_detection_nms(image, constants.sigma)
    print(np.any(image_m==0))
    
    print("-" * 50 + "\n" + "Edge Image:")
    plt.imshow(image_m, cmap="gray"); plt.show() # Displaying the sample image
    
    image_m_thres = 1.0*(image_m > constants.thres) # threshold the edge magnitude image
    print("-" * 50 + "\n" + "Thresholded Edge Image:")
    plt.imshow(image_m_thres, cmap="gray"); plt.show() # Displaying the sample image
    y_idxs, x_idxs = image_m_thres.nonzero() 
    # print(x_idxs, y_idxs)
    
    #--------------hough transform----------------
    H, rho_arr, theta_arr = hough_transform(image_m, constants.thres, constants.rho_res, constants.theta_res)
    H = hough_accumulator_nms(H) # nms on H
    peak_rho_arr, peak_theta_arr = peak_hough_lines(H, rho_arr, theta_arr, constants.num_lines)
    vis_image_rgb = np.copy(image_rgb)
    vis_line_len = 0.01
    for (x0, y0) in zip(x_idxs, y_idxs):
        for (rho, theta) in zip(peak_rho_arr, peak_theta_arr):
            if (rho - (x0*np.cos(theta)+y0*np.sin(theta))<0.00001):
                x1 = int(x0 - vis_line_len*np.sin(theta)); y1 = int(y0 + vis_line_len*np.cos(theta))
                x2 = int(x0 + vis_line_len*np.sin(theta)); y2 = int(y0 - vis_line_len*np.cos(theta)); 
                cv2.line(vis_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    plt.imshow(vis_image_rgb); plt.show() # Displaying the sample image


def main():
    image_idx = np.random.randint(0, len(image_list))
    # image_idx = 0
    visualize_ls(image_list[image_idx], constants)

if __name__=="__main__":
    main()