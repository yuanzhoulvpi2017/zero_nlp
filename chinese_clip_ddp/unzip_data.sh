
#!/bin/bash

# 目标文件夹
TARGET_DIR="data/jackyhate/text-to-image-2M/data_512_2M"
# 解压目标文件夹
DEST_DIR="data/jackyhate/unzip2mdata"
mkdir -p $DEST_DIR
# 遍历并解压所有以 .tar 结尾的文件
for file in "$TARGET_DIR"/*.tar; do
  if [ -f "$file" ]; then
    echo "正在解压: $file"
    temp_name=$(basename $file .tar)
    if tar -tf "$file" > /dev/null 2>&1; then
      rm -rf "$DEST_DIR/$temp_name"
      mkdir -p "$DEST_DIR/$temp_name"
      tar -xvf "$file" -C "$DEST_DIR/$temp_name" || { echo "解压 $file 时出错，跳过该文件"; continue; }
    else
      echo "$file 不是有效的 tar 文件，跳过该文件"
    fi
  fi
done

# 查看文件夹数量
find data/jackyhate/unzip2mdata -type f | wc -l