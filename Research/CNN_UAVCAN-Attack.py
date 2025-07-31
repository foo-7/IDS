import torch
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from CNN_Model import CNN_Model as CNN
from DataPreprocess import DataPreprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_bin_1 = pd.read_csv()