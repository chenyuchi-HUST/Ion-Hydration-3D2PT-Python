import sys
import numpy as np
import time

###################### 定义平均函数 ##############################
def average_npz_files(file_names, output_prefix):
    # 读取第一个文件作为基准
    print(f"读取文件: {file_names[0]}")
    with np.load(file_names[0]) as data:
        metadata = eval(str(data['metadata'][0]))
        data_name = metadata.get('data_name', 'data')
        print(f"数据名称: {data_name}")
        if data_name in ['density']:  
            total_data = data[data_name].copy()
            n_files = 1
        elif data_name in ['vacf_trans', 'vacf_ang']:
            # VACF数据需要加权平均
            total_data = data[data_name].copy()
            total_counts = data['counts'].copy()
        else: 
            # 其他数据类型（偶极角度、主轴、四面体序参数、势能）
            original_data = data[data_name].copy()
            # 创建两倍大小的数组，用nan填充
            new_shape = list(original_data.shape)
            new_shape[3] = new_shape[3] * 2  # 只将第4维（max_angles_per_grid）扩大到两倍
            total_data = np.full(new_shape, np.nan, dtype=original_data.dtype)
            if len(original_data.shape) == 4:  # 对于4维数据
                total_data[:,:,:,:original_data.shape[3]] = original_data
            elif len(original_data.shape) == 6:  # 对于6维数据（主轴）
                total_data[:,:,:,:original_data.shape[3],:,:] = original_data
            total_counts = data['counts'].copy()
    
    # 读取并处理其他文件的数据
    for fname in file_names[1:]:
        print(f"读取文件: {fname}")
        with np.load(fname) as data:
            curr_metadata = eval(str(data['metadata'][0]))
            curr_data = data[data_name]
            
            # 检查网格参数
            if (curr_metadata['grid_sizes'] != metadata['grid_sizes'] or 
                curr_metadata['grid_spacings'] != metadata['grid_spacings']):
                raise ValueError(f"文件 {fname} 中的网格参数不匹配")
            
            if data_name in ['density']:
                # 平均粒子数量直接累加
                total_data += curr_data
                n_files += 1
            
            elif data_name in ['vacf_trans', 'vacf_ang']:
                # VACF数据的加权平均
                curr_counts = data['counts']
                total_data += curr_data 
                total_counts += curr_counts
            
            else:
                # 检查数据维度（除了第4维外的其他维度都应该匹配）
                expected_shape = list(total_data.shape)
                expected_shape[3] = curr_data.shape[3]  # 第4维可以不同
                if curr_data.shape != tuple(expected_shape):
                    raise ValueError(f"文件 {fname} 中的数据维度不匹配: "
                                  f"期望形状 {expected_shape}, "
                                  f"实际形状 {curr_data.shape}")
                
                curr_counts = data['counts']
                double_max = total_data.shape[3]  # 两倍最大容量
                for i in range(total_data.shape[0]):
                    for j in range(total_data.shape[1]):
                        for k in range(total_data.shape[2]):
                            count = curr_counts[i,j,k]
                            if count > 0:
                                current_stored = total_counts[i,j,k]
                                if current_stored < double_max:
                                    space_left = double_max - current_stored
                                    data_to_add = min(count, space_left)
                                    # 根据数据维度正确处理数据复制
                                    if len(curr_data.shape) > 4:
                                        total_data[i,j,k,current_stored:current_stored+data_to_add,...] = \
                                            curr_data[i,j,k,:data_to_add,...]
                                    else:
                                        total_data[i,j,k,current_stored:current_stored+data_to_add] = \
                                            curr_data[i,j,k,:data_to_add]
                                    total_counts[i,j,k] += data_to_add
    
    # 处理最终结果
    if data_name in ['density']:
        # 平均粒子数量计算算术平均
        averaged_data = total_data / n_files
        metadata['n_files_averaged'] = n_files
    elif data_name in ['vacf_trans', 'vacf_ang']:
        # VACF数据：计算加权平均
        averaged_data = np.zeros_like(total_data)
        if data_name == 'vacf_trans':
            # vacf_trans: (nx, ny, nz, n_delay)
            for i in range(total_data.shape[0]):
                for j in range(total_data.shape[1]):
                    for k in range(total_data.shape[2]):
                        if total_counts[i,j,k] > 0:
                            averaged_data[i,j,k,:] = total_data[i,j,k,:] / total_counts[i,j,k]
        else:
            # vacf_ang: (nx, ny, nz, 3, n_delay)
            for i in range(total_data.shape[0]):
                for j in range(total_data.shape[1]):
                    for k in range(total_data.shape[2]):
                        if total_counts[i,j,k] > 0:
                            averaged_data[i,j,k,:,:] = total_data[i,j,k,:,:] / total_counts[i,j,k]
    else:
        # 保持堆积形式
        averaged_data = total_data
    
    # 更新并保存元数据
    metadata['averaged_from_files'] = True
    metadata['creation_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    # 保存结果
    print(f"保存结果到: {output_prefix}_averaged.npz")
    save_dict = {
        data_name: averaged_data,
        'metadata': np.array([str(metadata)])
    }
    if data_name not in ['density']: # 密度数据不保存计数
        save_dict['counts'] = total_counts
    
    np.savez_compressed(f'{output_prefix}_averaged.npz', **save_dict)
    
    nx, ny, nz = metadata['grid_sizes']
    print(f"处理完成。网格尺寸: {nx}x{ny}x{nz}")
    print(f"数据形状: {averaged_data.shape}")

###################### 定义主函数 ##############################
def main():
    if len(sys.argv) < 3:
        print("用法: python Average_npz.py <输出文件前缀> <输入NPZ文件1> <输入NPZ文件2> ...")
        sys.exit(1)
    
    output_prefix = sys.argv[1]
    file_names = sys.argv[2:]
    
    average_npz_files(file_names, output_prefix)

if __name__ == "__main__":
    main()