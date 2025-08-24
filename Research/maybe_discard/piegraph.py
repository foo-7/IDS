import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    'Type': ['DoS', 'Injection', 'Spoofing', 'Manipulation', 'MITM', 'Password', 'Replay', 'UDP Hijacking', 'Video Interception', 'Evil Twin', 'Jamming'],
    'T-ITS': [11672, 4282, 0, 0, 0, 0, 12007, 0, 0, 5684, 0],
    'UAVCAN Attack': [227080, 317652, 498, 0, 0, 0, 211784, 0, 0, 0, 1460],
    'ISOT-Drone': [1220131, 4743, 9733, 6099, 105141, 231218, 5860, 5012, 5860, 0, 0]
}

df = pd.DataFrame(data)
df.set_index('Type', inplace=True)

# Total attacks per type
total_attacks = df.sum(axis=1)
total_sum = total_attacks.sum()

# --- Sort descending ---
total_attacks_sorted = total_attacks.sort_values(ascending=False)

# Legend labels with percentages
labels_with_pct = [f"{atype} ({attacks/total_sum*100:.1f}%)" 
                   for atype, attacks in zip(total_attacks_sorted.index, total_attacks_sorted)]

# ---------------- Pie chart figure ----------------
plt.figure(figsize=(8,8))
plt.pie(total_attacks, labels=None, startangle=140)  # keep original pie order
plt.axis('equal')
plt.tight_layout()
plt.savefig('attack_types_pie_only.png', transparent=True, dpi=300, bbox_inches='tight')
plt.close()

# ---------------- Legend figure ----------------
fig_legend = plt.figure(figsize=(4,8))
# Create dummy handles for legend (match color cycle to index order!)
handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=f"C{list(total_attacks.index).index(atype)}", markersize=15)
    for atype in total_attacks_sorted.index
]

plt.legend(handles, labels_with_pct, title='Attack Types', loc='center')
plt.axis('off')
plt.tight_layout()
plt.savefig('attack_types_legend_only.png', transparent=True, dpi=300, bbox_inches='tight')
plt.close()
