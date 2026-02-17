# UAV Intrusion Detection System (IDS)

This repository contains implementations of a hybrid **Random Forest → Support Vector Classifier (RF--SVC)** and a **1D Convolutional Neural Network (1D-CNN)** for multi-class UAV network intrusion detection. The project includes:

- Data preprocessing
- Model training and evaluations

## Setup
Requires Python 3.10+

It is recommended to use a virtual environment:

```bash
python -m venv <env_name>
source <env_name>/bin/activate  # Linux/macOS
<env_name>\Scripts\activate     # Windows
```

Afterwards, please install the environment:

```bash
pip install -r requirements.txt
```


## DELETE THIS LATER
## IF WORKING ON WINDOWS
1. Install WSL2 onto your machine
2. Please install the WSL extension on Virtual Studio Code
3. Find the >< icon within the bottom left of Virtual Studio Code
4. Click it and select "Connect to WSL"
5. VS Code will restart. Now, your, terminal, file explorer, and Python interpreter are all running "inside" Linux