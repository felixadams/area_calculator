# Cell Colony Area Calculator

This project provides a method to estimate the area of cell colonies in bright field images captured in 6 well plates. The process involves processing the original bright field images along with the corresponding h5 files generated after the modeling step.

## How to Run:

### Jupyter Notebook Version:
1. Open the notebook file in a Jupyter environment.
2. Ensure that the input folder contains the original bright field images and their corresponding h5 files.
3. Run the notebook cells step by step to process the images and estimate the colony areas.

### Python File Version:
1. Ensure you have Python installed on your system.
2. Run the following command in the terminal:
    ```
    python photo_processing.py "path/to/input_folder" "path/to/output_folder"
    ```
    Replace `"path/to/input_folder"` with the path to the folder containing the original bright field images and their corresponding h5 files. Replace `"path/to/output_folder"` with the desired path where you want to save the output results.

   this example worked for me
       ```
   PS C:\Users\felix\Documents\Python Scripts> python photo_processing.py "C:\Users\felix\Documents\confluence_measure\input_photos" "C:\Users\felix\Documents\confluence_measure\output_confluence_calculator"
       ```
## Description:
- The input folder should contain the original bright field images and their corresponding h5 files obtained after the modeling step.
- The `photo_processing.py` script processes the images and estimates the area of cell colonies.
- The output results are saved in the specified output folder.

## Dependencies:
- Python
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- h5py
