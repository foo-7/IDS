# IDS
Intrusion Detection System. Using ML models to create IDS. Research under Professor Torre. T CSS 499
Please create a virtual environment. 

# To create a virtual environment
python -m venv venv

# For virtual environment reminder
Please run: .\venv\Scripts\Activate.ps1 to activate venv.
Or run: .\venv\Scripts\activate

# RF-SVC GPU-accelerator reminder
Please login to Docker

To use ML models in GPU, please run:
docker run -it --rm --gpus all --memory=16g -v "C:\Users\noahd\OneDrive\Desktop\CleanIDS:/workspace" rapidsai/base:24.06-cuda11.8-py3.11 bash

