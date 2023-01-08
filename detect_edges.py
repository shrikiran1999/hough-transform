import nbimporter
import cv2, math, os
import numpy as np
import matplotlib.pyplot as plt

from utils import get_parameters, Gauss2D, filter_image_vec

image_list, constants = get_parameters()

#----------------------------------------------------------------------
def visualize(function, image_name, sigma):
    image_rgb = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)

    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    image_m, image_o, image_x, image_y = function(image, sigma)
    
    print("-" * 50 + "\n" + "Original Image:")
    plt.imshow(image_rgb); plt.show() # Displaying the sample image
    
    print("-" * 50 + "\n" + "Edge Magnitude:")
    plt.imshow(image_m, cmap="gray"); plt.show()
    
    print("-" * 50 + "\n" + "Edge Orientation:")
    plt.imshow(image_o, cmap="gray"); plt.show()
    
    print("-" * 50 + "\n" + "Gradient x:")
    plt.imshow(image_x, cmap="gray"); plt.show()
    
    print("-" * 50 + "\n" + "Gradient y:")
    plt.imshow(image_y, cmap="gray"); plt.show()
    
    return


def edge_detection(image, sigma):
    """Detect edges in the image

    Args:
        image: np.array, HxW, the input grayscale image. 
        sigma: float, std dev of the Gauss2D filter used as in creation of h_filter. 

    Returns:
        image_m: np.array, HxW, contains the edge magnitudes, each value in [-rho_max ,rho_max]
        image_o: np.array, HxW, edge orientations in degrees [0, 360]
        image_x: np.array, HxW, image gradient in x
        image_y: np.array, HxW, image gradient in y
    """
    h_size = 2*math.ceil(3*sigma) + 1
    gaussian_kernel = Gauss2D(kernel=(h_size, h_size), sigma=sigma)
    
    #--------------------------------------
    # smooth image using gaussian kernel, we overwrite variable image with the smoothed image!
    # image = ?
    # YOUR CODE HERE
    image = filter_image_vec(image, gaussian_kernel)
    # raise NotImplementedError()
    
    #--------------------------------------
    # define sobel_filter_x and sobel_filter_y, you can ignore the 1/8 normalization weight!
    # sobel_filter_x = ?
    # sobel_filter_y = ?
    # YOUR CODE HERE
    sobel_filter_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    sobel_filter_x = sobel_filter_y.T
    # raise NotImplementedError()
    
    image_x = filter_image_vec(image, sobel_filter_x)
    image_y = filter_image_vec(image, sobel_filter_y)
    
    # YOUR CODE HERE
    image_m = np.sqrt(image_x**2 + image_y**2)
    image_o = np.arctan2(image_y,image_x)*(180/np.pi) # [-180, 180)

    # converting to [0,360] range
    for i in range(image_o.shape[0]):
        for j in range(image_o.shape[1]):
            if (image_o[i,j]>=-180.0 and image_o[i,j]<-90):
                image_o[i,j] = 360.0+image_o[i,j]
            if (image_o[i,j]>=-90.0 and image_o[i,j]<0):
                image_o[i,j] = 360.0+image_o[i,j]


    # raise NotImplementedError()
    
    return image_m, image_o, image_x, image_y


def edge_detection_nms(image, sigma):
    """Detect edges in the image with nms preprocessing

    Args:
        image: np.array, HxW, the input grayscale image. 
        sigma: float, std dev of the Gauss2D filter used as in creation of h_filter. 

    Returns:
        image_m: np.array, HxW, contains the edge magnitudes
        image_o: np.array, HxW, edge orientations in degrees [0, 360]
        image_x: np.array, HxW, image gradient in x
        image_y: np.array, HxW, image gradient in y
    """
    h_size = 2*math.ceil(3*sigma) + 1
    gaussian_kernel = Gauss2D(kernel=(h_size, h_size), sigma=sigma)
    
    #--------------------------------------
    # smooth image using gaussian kernel, we overwrite variable image with the smoothed image!
    # image = ?
    # YOUR CODE HERE
    image = filter_image_vec(image, gaussian_kernel)
    # raise NotImplementedError()
    
    #--------------------------------------
    # define sobel_filter_x and sobel_filter_y, you can ignore the 1/8 normalization weight!
    # sobel_filter_x = ?
    # sobel_filter_y = ?
    # YOUR CODE HERE
    sobel_filter_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    sobel_filter_x = sobel_filter_y.T
    # raise NotImplementedError()
    
    image_x = filter_image_vec(image, sobel_filter_x)
    image_y = filter_image_vec(image, sobel_filter_y)
    
    # image_m = ?
    # image_o = ?
    # YOUR CODE HERE
    image_m = np.sqrt(image_x**2 + image_y**2)
    image_o = np.arctan2(image_y,image_x)*(180/np.pi) # [-180, 180)
    # raise NotImplementedError()

    # converting to [0,360] range
    for i in range(image_o.shape[0]):
        for j in range(image_o.shape[1]):
            if (image_o[i,j]>=-180.0 and image_o[i,j]<-90):
                image_o[i,j] = 360.0+image_o[i,j]
            if (image_o[i,j]>=-90.0 and image_o[i,j]<0):
                image_o[i,j] = 360.0+image_o[i,j]
    
    # apply nms
    image_m = edge_nms(image_m, image_o)
    
    return image_m, image_o, image_x, image_y

#----------------------------------------------------------------------
def edge_nms(image_m, image_o):
    """Performs edge nms on image_m
    Args:
        image_m: np.array, HxW, edge magnitude image
        image_o: np.array, HxW, edge orientations image

    Returns:
        image_m_prime: np.array, suppressed image_m after NMS
    """
    mask = np.ones_like(image_m) ## per pixel boolean mask, 1 = keep, 0 = suppress
    
    # loop per pixel
    for i in range(1, image_m.shape[0]-1):
        for j in range(1, image_m.shape[1]-1):
            
            # round of the pixel gradient to one of the 4 cases in degrees. Reminder, image_o is [0, 360]
            # pixel_gradient = ?
            # YOUR CODE HERE
            t = image_o[i,j]
            if (t<22.5 and t>=0) or (t>=337.5 and t<360) or (t>=157.5 and t<202.5):
                pixel_gradient = 0.0
            elif (t>=22.5 and t<67.5) or (t>=202.5 and t<247.5):
                pixel_gradient = 45.0
            elif (t>=67.5 and t<112.5) or (t>=247.5 and t<292.5):
                pixel_gradient = 90.0
            else:
                pixel_gradient = 135.0
            # raise NotImplementedError()
            
            mask[i, j] = keep_pixel(image_m, i, j, pixel_gradient)
    
    image_m_prime = mask*image_m
    
    return image_m_prime

#----------------------------------------------------------------------
def keep_pixel(image_m, i, j, gradient):
    """Performs edge nms on image_m
    Args:
        image_m: np.array, HxW, edge magnitude image 
        i: integer, row index of pixel
        j: integer, col index of pixel
        gradient: integer, rounded gradient in degrees, one of the values in [0, 45, 90, 135].

    Returns:
        output: boolean integer (1 or 0). 1 to keep pixel, 0 to suppress pixel
    """
    
    # Compare the magnitude at image_m[i, j] with its neighbours
    # angle decides which neighbours to check
    
    # YOUR CODE HERE
    # raise NotImplementedError()
    ############# remember right hand rule while determining x and y axes
    ## when eliminating lower gradients, say a point is on an edge along 0deg, you need to check with pixel perpendicular to the direction
    ## of the edge (thickness of the edge has be reduced perpendicular to the edge direction)
    if gradient==0.0:
        if image_m[i,j+1]>image_m[i,j] or image_m[i,j-1]>image_m[i,j]:
            output = 0
        else:
            output = 1
    elif gradient==45.0:
        if image_m[i+1,j+1]>image_m[i,j] or image_m[i-1,j-1]>image_m[i,j]:
            output = 0
        else:
            output = 1
    elif gradient==90.0:
        if image_m[i+1,j]>image_m[i,j] or image_m[i-1,j]>image_m[i,j]:
            output = 0
        else:
            output = 1
    elif gradient==135.0:
        if image_m[i-1,j+1]>image_m[i,j] or image_m[i+1,j-1]>image_m[i,j]:
            output = 0
        else:
            output = 1
    
    
    return output

def main():
    image_list, constants = get_parameters()
    image_idx = np.random.randint(0, len(image_list))
    visualize(edge_detection, image_list[image_idx], constants.sigma)
    
if __name__=="__main__":
    main()