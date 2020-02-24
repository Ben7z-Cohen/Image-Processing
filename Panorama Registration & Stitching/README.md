**Overview**
In this exercise, I was guided through the steps discussed in class to perform automatic ”Stereo
Mosaicking”. The input of such an algorithm is a sequence of images scanning a scene from left to right
(due to camera rotation and/or translation -  assuming rigid transform between images), with significant
overlap in the field of view of consecutive frames. This exercise covers the following steps:

• Registration:<br/> The geometric transformation between each consecutive image pair is found by de-
tecting Harris feature points, extracting their MOPS-like descriptors, matching these descriptors
between the pair and fitting a rigid transformation that agrees with a large set of inlier matches
using the RANSAC algorithm.

• Stitching:<br/> Combining strips from aligned images into a sequence of panoramas. Global motion will
be compensated, and the residual parallax, as well as other motions will become visible.
