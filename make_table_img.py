import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_excel(r'f:\NIDM-SRM Drone Project Docs\ref_for_doc\GLOF Index.xlsx')
print("Columns:", list(df.columns))
cols_wanted = ['W_ID','ICI','SI','AAR','CCI','HI','CSI','Year']
cols_available = [c for c in cols_wanted if c in df.columns]
df5 = df[cols_available].head(5).round(4)

fig, ax = plt.subplots(figsize=(14, 3))
ax.axis('off')

table = ax.table(
    cellText=df5.values,
    colLabels=df5.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.2)

for j in range(len(df5.columns)):
    table[0, j].set_facecolor('#1A5276')
    table[0, j].set_text_props(color='white', fontweight='bold')

for i in range(1, 6):
    color = '#D6EAF8' if i % 2 == 0 else '#FFFFFF'
    for j in range(len(df5.columns)):
        table[i, j].set_facecolor(color)

plt.title('GLOF_Index.xlsx — First 5 Rows (CCI / SI / AAR / ICI)', 
          fontsize=13, fontweight='bold', pad=20, color='#1A5276')
plt.tight_layout()
out = r'f:\NIDM-SRM Drone Project Docs\glof_index_table.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved to', out)
