import os
import glob

def remove_zone_identifier_files(dataset_path="Dataset"):
    """
    删除指定目录下所有的Zone.Identifier文件
    
    Args:
        dataset_path (str): Dataset目录的路径，默认为"Dataset"
    """
    # 确保路径存在
    if not os.path.exists(dataset_path):
        print(f"错误: 目录 '{dataset_path}' 不存在")
        return
    
    # 使用glob递归查找所有Zone.Identifier文件
    pattern = os.path.join(dataset_path, "**", "*Zone.Identifier")
    zone_files = glob.glob(pattern, recursive=True)
    
    if not zone_files:
        print(f"在 '{dataset_path}' 目录下未找到Zone.Identifier文件")
        return
    
    print(f"找到 {len(zone_files)} 个Zone.Identifier文件:")
    
    removed_count = 0
    for file_path in zone_files:
        try:
            print(f"删除: {file_path}")
            os.remove(file_path)
            removed_count += 1
        except OSError as e:
            print(f"删除失败 {file_path}: {e}")
    
    print(f"\n完成! 成功删除了 {removed_count} 个Zone.Identifier文件")

if __name__ == "__main__":
    # 执行删除操作
    remove_zone_identifier_files()
    
    # 如果需要指定其他路径，可以这样调用：
    # remove_zone_identifier_files("/path/to/your/dataset")