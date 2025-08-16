import matplotlib.pyplot as plt
import pandas as pd

# Table data
data = {
    "Model": ["RF-SVC", "CNN", "RF-SVC", "CNN", "RF-SVC", "CNN", "RF-SVC", "CNN"],
    "Dataset": ["UAV Attack", "UAV Attack", "T-ITS", "T-ITS", "UAVCAN Attack", "UAVCAN Attack", "ISOT-Drone", "ISOT-Drone"],
    "Precision (%)": [100, 100, 100, 100, 83.32, 91.24, 100, 100],
    "Recall (%)": [100, 100, 100, 100, 83.24, 80.62, 100, 100],
    "Accuracy (%)": [100, 100, 100, 100, 83.24, 93.02, 100, 100],
    "F1-Score (%)": [100, 100, 100, 100, 82.11, 83.88, 100, 100]
}

df = pd.DataFrame(data)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

# Create table
table = ax.table(cellText=df.values,
                 colLabels=df.columns,
                 cellLoc='center',
                 loc='center')

# Set font size
table.auto_set_font_size(False)
table.set_fontsize(10)

# Set column widths
table.auto_set_column_width(col=list(range(len(df.columns))))

# Add horizontal lines to separate datasets
row_separators = [2, 4, 6]  # after which row to draw a thicker line
for i in range(len(df)):
    for j in range(len(df.columns)):
        cell = table[i+1, j]  # +1 because row 0 is header
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)
        # Thicker line for dataset separation
        if i in row_separators:
            cell.set_linewidth(2)

# Add vertical lines for all columns
for j in range(len(df.columns)):
    for i in range(len(df)+1):  # include header row
        cell = table[i, j]
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)

# Save as PNG with transparent background
plt.savefig("multi_class_table_separated.png", bbox_inches='tight', dpi=300, transparent=True)
plt.close()
