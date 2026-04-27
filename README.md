# Cluste-YOLO
Cluste-YOLO: an object detection framework for rapeseed inflorescence cluster
Shuchen Liu<sup>1,2</sup>, Yunpeng Cui<sup>1,2*</sup>

The main files are as follows:

(1) samples/ - Sample prediction data folder in the current package.

    input/ - Input image folder.

    output/ - Output result folder.

(2) ultralytics/ - Minimal runtime source code kept for the current model structure.

(3) weight/ - Model weight folder.

    Cluste-YOLO.pt - Current detection weight file used by the script.

(4) detect.py - Main detection script for image prediction.

Default folder behavior:

- The script loads weights from: weight/Cluste-YOLO.pt
- The script reads images from: samples/input
- The script saves results to: samples/output


• Read the beginner’s guide to Python if you are new to the language: https://wiki.python.org/moin/BeginnersGuide

• For Windows users, Python 3 release can be downloaded via: https://www.python.org/downloads/windows/

• To install Anaconda Python distribution:

1) Read the install instruction using the URL: https://docs.continuum.io/anaconda/install

2) For Windows users, a detailed step-by-step installation guide can be found via: https://docs.continuum.io/anaconda/install/windows

3) An Anaconda Graphical installer can be found via: https://www.continuum.io/downloads

Some dependencies of the Jupyter notebooks:

- Python 3.8 or later
- torch
- numpy
- opencv-python
