# metrics

**metrics** is a script designed to automate DIBCO measurement.</br>
It requires DIBCO weights and metrics evaluation tools and data folder
with pairs of binarized and ground-truth images with following names format:</br>
*\d+_(gt|out).png* (*1_gt.png*, *2_out.png*, *33_gt.png*, etc).</br>
The result of script are text files in data folder with following names format:</br>
*\d+_res.png* (*1_res.txt*, *33_res.txt*) for every image and *total_res.png* for all folder.</br>
Result files contain four measures: F-Measure, pseudo F-measure, PSNR and DRD.
