import os

for filename in os.listdir():
    if filename.startswith("sharp_edge_2023"):
        os.remove(filename)
        # print(f"Deleted file: {filename}")