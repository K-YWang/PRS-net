import thingi10k

thingi10k.init(cache_dir="./Thingi10K")

count = 0
for entry in thingi10k.dataset():
    file_id = entry['file_id']
    vertices, faces = thingi10k.load_file(entry['file_path'])
    print(f"{file_id}: vertices={len(vertices)}, faces={len(faces)}")
    count += 1

print(f"✅ 总共加载了 {count} 个模型")
