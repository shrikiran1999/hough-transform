# Hough Transform
<img align="center" src="data/handout/ht.gif" width="500">

This repo contains a from-scratch implementation of hough transform. (Implemented as part of the Computer Vision course 16720B at CMU)
Steps in the imlementation:
1. **Edge detection with Non-Maximal Suppression(NMS)** (NMS is done to get sharper edges) 
2. **Hough transform with Non-Maximal Supression**(to supress neighbouring hough lines) is applied on the edge magnitude image to identify the hough lines.
3. Visualization of intersection of hough lines and edges



## Sample output:
1. Edge detection with NMS
<img src="data/img01.jpg" alt="img01" width="200"/> <img src="edgenms_results/01.png" alt="img01" width="200"/>

