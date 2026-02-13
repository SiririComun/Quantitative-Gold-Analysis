import csv
import os

files_to_sample = [
    'data/raw/hipervinculos_wsj.csv',
    'data/processed/articulos_filtrados_ordenados.csv'
]

def create_sample(file_path, rows=500):
    if os.path.exists(file_path):
        sample_path = file_path.replace('.csv', '_sample.csv')
        print(f"Creating sample of: {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f_in:
            reader = csv.reader(f_in)
            with open(sample_path, 'w', encoding='utf-8', newline='') as f_out:
                writer = csv.writer(f_out)
                for i, row in enumerate(reader):
                    if i > rows:
                        break
                    writer.writerow(row)
        print(f"✅ Sample created in: {sample_path}")
    else:
        print(f"⚠️ File not found: {file_path}")

if __name__ == "__main__":
    for f in files_to_sample:
        create_sample(f)