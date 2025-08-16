import glob
import os
import pandas as pd

"""
    UAV Attack
"""
df_jamming = pd.read_csv('jamming-merged-gps-only.csv')
df_spoofing = pd.read_csv('spoofing-merged-gps-only.csv')
print(f'[UAV ATTACK JAMMING] Total jamming: {(df_jamming['label'] == 'malicious').sum()}')
print(f'[UAV ATTACK SPOOFING] Total spoofing: {(df_spoofing['label'] == 'malicious').sum()}')

"""
    T-ITS 
"""
df_T_ITS = pd.read_csv('Dataset_T-ITS.csv')
print(f'[T-ITS] Checking whether if the dataset size is 54784 (index starts at 1, so its 54783) (T or F): {df_T_ITS.shape[0] == 54783}')
print(f'[T-ITS] The size of the dataset: {df_T_ITS.shape[0]}')

"""
    UAVCAN Attack
"""
bin_files = glob.glob('UAVCAN-Attack/*.bin')
file_size = {}
column_names = [
    'Status', 'Time', 'Interface', 'ID', 'Length', 'Data1', 'Data2', 
    'Data3', 'Data4', 'Data5', 'Data6', 'Data7','Data8'
]

df_files = []
for file in bin_files:
    df = pd.read_csv(file, sep=r'\s+', names=column_names)
    df['Time'] = df['Time'].str.replace('[()]', '', regex=True)  # Remove parentheses
    df['Time'] = df['Time'].astype(float)
    file_size[file] = df.shape[0]
    df_files.append(df)

print(len(df_files) == 10)

malicious_type1 = (df_files[1]['Status'] == 'Attack').sum() # Flooding
malicious_type2 = (df_files[2]['Status'] == 'Attack').sum() # Flooding
malicious_type3 = (df_files[3]['Status'] == 'Attack').sum() # Fuzzy
malicious_type4 = (df_files[4]['Status'] == 'Attack').sum() # Fuzzy
malicious_type5 = (df_files[5]['Status'] == 'Attack').sum() # Replay
malicious_type6 = (df_files[6]['Status'] == 'Attack').sum() # Replay

malicious_type7_Flooding = 0
malicious_type7_Fuzzy = 0
found_attack = 0
for index, row in df_files[7].iterrows():
    if row['Status'] == 'Attack':
        found_attack += 1

    time_rounded = round(row['Time'])
    # Flooding
    if time_rounded >= 48 and \
       time_rounded <= 92 and \
       row['Status'] == 'Attack':
        malicious_type7_Flooding += 1
    
    # Fuzzy
    elif time_rounded >= 98 and \
         time_rounded <= 132 and \
         row['Status'] == 'Attack':
        malicious_type7_Fuzzy += 1

    # Flooding
    elif time_rounded >= 138 and \
         time_rounded <= 182 and \
         row['Status'] == 'Attack':
        malicious_type7_Flooding += 1

    # Fuzzy
    elif time_rounded >= 188 and \
         time_rounded <= 222 and \
         row['Status'] == 'Attack':
        malicious_type7_Fuzzy += 1

print(bin_files)
print(df_files[6])
print(f'[TYPE 7 ATTACK INFO] {found_attack}')

malicious_type8_Replay = 0
malicious_type8_Fuzzy = 0
for index, row in df_files[8].iterrows():
    time_rounded = round(row['Time'])

    # Fuzzy
    if time_rounded >= 60 and \
       time_rounded <= 100 and \
       row['Status'] == 'Attack':
        malicious_type8_Fuzzy += 1
    
    # Replay
    elif time_rounded >= 110 and \
         time_rounded <= 140 and \
         row['Status'] == 'Attack':
        malicious_type8_Replay += 1

    # Fuzzy
    elif time_rounded >= 150 and \
         time_rounded <= 190 and \
         row['Status'] == 'Attack':
        malicious_type8_Fuzzy += 1

    # Replay
    elif time_rounded >= 200 and \
         time_rounded <= 230 and \
         row['Status'] == 'Attack':
        malicious_type8_Replay += 1

malicious_type9_Flooding = 0
malicious_type9_Replay = 0
for index, row in df_files[9].iterrows():
    time_rounded = round(row['Time'])

    # Flooding
    if time_rounded >= 55 and \
       time_rounded <= 114 and \
       row['Status'] == 'Attack':
        malicious_type9_Flooding += 1
    
    # Replay
    elif time_rounded >= 115 and \
         time_rounded <= 154 and \
         row['Status'] == 'Attack':
        malicious_type9_Replay += 1

    # Flooding
    elif time_rounded >= 155 and \
         time_rounded <= 204 and \
         row['Status'] == 'Attack':
        malicious_type9_Flooding += 1

    # Replay
    elif time_rounded >= 205 and \
         time_rounded <= 270 and \
         row['Status'] == 'Attack':
        malicious_type9_Replay += 1

malicious_type10_Flooding = 0
malicious_type10_Fuzzy = 0
malicious_type10_Replay = 0
for index, row in df_files[0].iterrows():
    time_rounded = round(row['Time'])

    # Flooding
    if time_rounded >= 60 and \
       time_rounded <= 110 and \
       row['Status'] == 'Attack':
        malicious_type10_Flooding += 1
    
    # Replay
    elif time_rounded >= 120 and \
         time_rounded <= 160 and \
         row['Status'] == 'Attack':
        malicious_type10_Fuzzy += 1

    # Flooding
    elif time_rounded >= 170 and \
         time_rounded <= 200 and \
         row['Status'] == 'Attack':
        malicious_type10_Replay += 1

print(f'[TYPE 1]: Flooding attacks amount: {malicious_type1} | Total attacks: {malicious_type1}')
print(f'[TYPE 2]: Flooding attacks amount: {malicious_type2} | Total attacks: {malicious_type2}')
print(f'[TYPE 3]: Fuzzy attacks amount: {malicious_type3} | Total attacks: {malicious_type3}')
print(f'[TYPE 4]: Fuzzy attacks amount: {malicious_type4} | Total attacks: {malicious_type4}')
print(f'[TYPE 5]: Replay attacks amount: {malicious_type5} | Total attacks: {malicious_type5}')
print(f'[TYPE 6]: Replay attacks amount: {malicious_type6} | Total attacks: {malicious_type6}')
print(f'[TYPE 7]: Flooding attacks amount: {malicious_type7_Flooding}, Fuzzy attacks amount: {malicious_type7_Fuzzy} | Total attacks: {malicious_type7_Flooding + malicious_type7_Fuzzy}')
print(f'[TYPE 8]: Fuzzy attacks amount: {malicious_type8_Fuzzy}, Replay attacks amount: {malicious_type8_Replay} | Total attacks: {malicious_type8_Fuzzy + malicious_type8_Replay}')
print(f'[TYPE 9]: Flooding attacks amount: {malicious_type9_Flooding}, Replay attacks amount: {malicious_type9_Replay} | Total attacks: {malicious_type9_Flooding + malicious_type9_Replay}')
print(f'[TYPE 10]: Flooding attacks amount: {malicious_type10_Flooding}, Fuzzy attacks amount: {malicious_type10_Fuzzy}, Replay attacks amount: {malicious_type10_Replay} | Total attacks: {malicious_type10_Flooding+malicious_type10_Replay+malicious_type10_Fuzzy}')

total_flooding = malicious_type1 + malicious_type2 + malicious_type7_Flooding + malicious_type9_Flooding + malicious_type10_Flooding
total_fuzzy = malicious_type3 + malicious_type4 + malicious_type8_Fuzzy + malicious_type7_Fuzzy + malicious_type10_Fuzzy
total_replay = malicious_type5 + malicious_type6 + malicious_type8_Replay + malicious_type9_Replay + malicious_type10_Replay
print(f'[ATTACK INFO] Total flooding: {total_flooding} | Total fuzzy: {total_fuzzy} | Total replay: {total_replay}')
print(
    f'[DOUBLE CHECK] Type 1: {(df_files[1]['Status'] == 'Attack').sum()} == {malicious_type1} | Type 2: {(df_files[2]['Status'] == 'Attack').sum()} == {malicious_type2} | Type 3: {(df_files[3]['Status'] == 'Attack').sum()}  == {malicious_type3}' + \
    f'| Type 4: {(df_files[4]['Status'] == 'Attack').sum()} == {malicious_type4} | Type 5: {(df_files[5]['Status'] == 'Attack').sum()} == {malicious_type5} | Type 6: {(df_files[6]['Status'] == 'Attack').sum()} == {malicious_type6} ' + \
    f'| Type 7: {(df_files[7]['Status'] == 'Attack').sum()} == {malicious_type7_Flooding + malicious_type7_Fuzzy} | Type 8: {(df_files[8]['Status'] == 'Attack').sum()} == {malicious_type8_Fuzzy + malicious_type8_Replay} | Type 9: {(df_files[9]['Status'] == 'Attack').sum()} == {malicious_type9_Flooding + malicious_type9_Replay} ' + \
    f'| Type 10: {(df_files[0]['Status'] == 'Attack').sum()} == {malicious_type10_Flooding + malicious_type10_Fuzzy + malicious_type10_Replay}'
)

"""
    ISOT-Drone
"""
directory_path = 'ISOT-Drone/'
attack_amount = {}

for subfolder in os.listdir(directory_path):
    subfolder_path = os.path.join(directory_path, subfolder)
    if os.path.isdir(subfolder_path):
        if subfolder != 'Regular':
            attack_amount[subfolder] = 0

            for file in os.listdir(subfolder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(subfolder_path, file)
                    df = pd.read_csv(file_path)
                    attack_amount[subfolder] += df.shape[0]

print(f'[ISOT-DRONE] All the amount of malicious attacks:\n{attack_amount}')