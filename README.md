This is a work in progress.

### **Generate hand-picked ISDs**
Use the command below to generate handpicked ISDs for a dataset. Change the `PATH_LOG` and `PATH_RGB` in the script to the correct log/pseudo log and rgb image paths respectively.

**Run the command:**
```bash
python generate_handpickedISD.py
```
This script generates grayscale ISD vector projections and the data needed to generate the ISDs in a text file, of the log images and saves the results in the folder `results_generatedISD`.

Sample generated image (RGB Image with lit and shadow points selected // Grayscale version of RGB // Reprojected log img in Grayscale ):

![100-2_compare](https://github.com/user-attachments/assets/fb230a81-55a1-43d9-8d48-a1b830256091)



### **Generate Color Reprojected Log Image & Pointcloud for debugging** 
Use the command below to generate color reprojected log images from ISD vectors. Set either the --handpickedISD or --networkISD to True, to set the desired ISD vector to be used during reprojection. 

**Run the command:**
```bash
python generate_color_reproj_from_ISD.py --handpickedISD T --saveimg T
```

This script saves the generated color reprojected log image and the corresponding pointcloud in the `results_colorreproj` folder.

Sample generated image (RGB Image // Log Chromaticity Image // Intensity map created using ISD // Color Reprojected Image):

![87-19_comparison](https://github.com/user-attachments/assets/37992256-298a-487c-be29-8bb351591451)

The corresponding pointcloud with the 3D red line as the ISD vector, the projection plane, the 3D log image points before and after projection.

![image](https://github.com/user-attachments/assets/bc69ec20-a1dd-4f70-8ad9-7bb5c94e8372)


### **Naive Reprojection method**

`sandbox.py` is a script that I used for testing out a bunch of different things. A very naive and **VERY** inefficient way of reprojection is implemented in this script along with a lot of other things.
Sample generated image (RGB image // Color Reprojected Image // Grayscale log projection)

![image](https://github.com/user-attachments/assets/f7c582e6-92dc-41ec-8b3f-a5eb3f8443fb)

