import sys
import subprocess
import os

try:
    import pandas as pd
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas', 'openpyxl', 'tabulate', '--quiet'])
    import pandas as pd

files = [
    r'f:\NIDM-SRM Drone Project Docs\ref_for_doc\Glacial lake parameters.xlsx',
    r'f:\NIDM-SRM Drone Project Docs\ref_for_doc\Glacier parameters.xlsx',
    r'f:\NIDM-SRM Drone Project Docs\ref_for_doc\GLOF Index.xlsx'
]

with open(r'f:\NIDM-SRM Drone Project Docs\ref_for_doc\table_dump.txt', 'w') as out_file:
    for file in files:
        try:
            out_file.write(f'\n--- {os.path.basename(file)} ---\n')
            df = pd.read_excel(file).head(3)
            out_file.write(df.to_markdown(index=False) + '\n')
        except Exception as e:
            out_file.write(f'Error reading {file}: {e}\n')
