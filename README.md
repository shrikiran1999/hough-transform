# Hough Transform for Edge Detection
<img align="center" src="data/handout/ht.gif" width="600">

###### *GIF source: 16720B course materials, Prof. Kris Kitani (CMU)*

\
This repo contains a from-scratch implementation of hough transform. (Implemented as part of the Computer Vision course 16720B at CMU)

## Concept:

If two edge points lay on the same line, their corresponding cosine curves will intersect each other on a specific $(ρ, θ)$ pair. Thus, the Hough Transform algorithm detects lines by finding the $(ρ, θ)$ pairs that have a number of intersections larger than a certain threshold.


## Repo structure:

1. Run ``` detect_edges.py ``` for **Edge detection with Non-Maximal Suppression(NMS)** (NMS is done to get sharper edges) 

2. Run ``` detect_houghlines.py ``` for applying **Hough transform with Non-Maximal Supression**(to supress neighbouring hough lines) on the edge magnitude image to identify the hough lines

3. Run ```visualize_ls.py ``` to visualize intersection of hough lines and edges



## Sample output:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Original image:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="data/img01.jpg" alt="img01" width="350"/>

1. Edge detection with NMS

<img src="edgenms_results/01.png" alt="img01" width="400"/>

2. Detected Hough lines using **Hough Transform**

<img src="houglines_results/01.png" alt="img01" width="400"/> 

3. Interesection of Hough lines with edges

<img src="final_results/0.png" alt="img01" width="400"/>




## ***Few more final results:***

<img src="final_results/1.png" alt="img01" width="400"/>

<img src="final_results/2.png" alt="img01" width="400"/>

<img src="final_results/3.png" alt="img01" width="400"/>

<img src="final_results/4.png" alt="img01" width="400"/>

<img src="final_results/5.png" alt="img01" width="400"/>

<img src="final_results/6.png" alt="img01" width="400"/>

<img src="final_results/7.png" alt="img01" width="400"/>


