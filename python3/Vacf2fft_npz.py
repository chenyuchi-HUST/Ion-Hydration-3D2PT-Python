import numpy as np
from scipy.constants import Boltzmann, atomic_mass, angstrom
import argparse
import time

###################### 定义计算 FFT 函数 ##############################
def symFT_cop(data, dt=0.008):
    """
    对数据进行对称傅里叶变换。
    
    参数:
    data : numpy.ndarray : 输入数据数组，形状为(n_delay,) 或 (3, n_delay)
    dt : float : 时间步长，默认为0.008 ps
    
    返回:
    numpy.ndarray : 傅里叶变换的幅度谱
    float : 频率间隔 (cm^-1)
    """
    dt_s = dt * 1e-12   # 将时间步长从皮秒转换为秒
    n = len(data) if data.ndim == 1 else data.shape[-1]  # 获取时间点数量
    lightspeed = 2.99792458e10  # 光速，单位：cm/s
    
    if data.ndim == 1:  # 平动VACF
        # 创建对称数组以进行FFT
        array = np.zeros(2 * n - 1, dtype=float)
        array[n-1:] = data
        array[:n-1] = data[1:][::-1]
        
        # 执行快速傅里叶变换
        ft_result = np.fft.fft(array, norm=None)
        ft_abs = np.abs(ft_result)
        ft_abs_norm = ft_abs / (2*n-1)
        ft_abs_norm_half = ft_abs_norm[range(int(n))]
        
    else:  # 角度VACF (3, n_delay)
        ft_abs_norm_half = np.zeros((3, n), dtype=float)
        for dim in range(3):
            array = np.zeros(2 * n - 1, dtype=float)
            array[n-1:] = data[dim]
            array[:n-1] = data[dim,1:][::-1]
            
            ft_result = np.fft.fft(array, norm=None)
            ft_abs = np.abs(ft_result)
            ft_abs_norm = ft_abs / (2*n-1)
            ft_abs_norm_half[dim] = ft_abs_norm[range(int(n))]
    
    df = 1.0 / ((2*n-1) * dt_s) / lightspeed  # 频率间隔 cm-1
    return ft_abs_norm_half, df

###################### 定义处理 VACF 的函数 ##############################
def process_vacf_npz(vacf_filename, fft_filename, temperature=300):
    """
    处理VACF的npz文件，执行FFT并保存结果。
    """
    # 常数设置
    time_conversion = 1e-12
    kBT = Boltzmann * temperature
    trans_factor = (angstrom ** 2) / (time_conversion ** 2)
    water_mass = 18.01528
    principal_moments = np.array([0.59697187, 1.3437603, 1.9385073])
    
    print(f"读取文件: {vacf_filename}")
    with np.load(vacf_filename) as data:
        metadata = eval(str(data['metadata'][0]))
        data_name = metadata.get('data_name', 'data')
        vacf_data = data[data_name]
        counts = data['counts'] if 'counts' in data else None
        
    # 获取网格尺寸 - 支持不对等网格划分
    if 'grid_sizes' in metadata:
        # 新格式：使用三维网格尺寸数组
        grid_sizes = np.array(metadata['grid_sizes'])
        nx, ny, nz = grid_sizes
        print(f"检测到不对等网格划分: {nx}x{ny}x{nz}")
    elif 'grid_size' in metadata:
        # 旧格式：使用单一网格尺寸
        grid_size = metadata['grid_size']
        if isinstance(grid_size, list):
            grid_sizes = np.array(grid_size)
            nx, ny, nz = grid_sizes
        else:
            nx = ny = nz = grid_size
            grid_sizes = np.array([nx, ny, nz])
        print(f"使用等分网格: {nx}x{ny}x{nz}")
    else:
        raise ValueError("元数据中缺少网格尺寸信息")
    dt = metadata.get('dt', 0.008)
    
    # 初始化结果数组
    if vacf_data.ndim == 5:
        # vacf_ang_grid shape: (nx, ny, nz, 3, n_delay)
        n_delay = vacf_data.shape[-1]
        fft_data = np.zeros((nx, ny, nz, 3, n_delay), dtype=np.float32)
    else:
        # vacf_trans_grid shape: (nx, ny, nz, n_delay)
        n_delay = vacf_data.shape[-1]
        fft_data = np.zeros((nx, ny, nz, n_delay), dtype=np.float32)
    
    # 对每个网格点进行FFT
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if counts is None or counts[i,j,k] > 0:
                    if fft_data.ndim == 5:
                        # 角度VACF (3, n_delay)
                        vacf = vacf_data[i,j,k]  # shape: (3, n_delay)
                        fft_result, df = symFT_cop(vacf, dt=dt)  # shape: (3, n_delay)
                        # 对每个维度使用对应的主惯性矩
                        for axis in range(3):
                            moment_of_inertia = principal_moments[axis] * atomic_mass * angstrom**2
                            fft_data[i,j,k,axis] = fft_result[axis] * (moment_of_inertia / (time_conversion**2)) * (2 / kBT)
                    else:
                        # 平动VACF (n_delay,)
                        vacf = vacf_data[i,j,k]
                        fft_result, df = symFT_cop(vacf, dt=dt)
                        fft_data[i,j,k] = fft_result * (water_mass * atomic_mass * trans_factor) * (2 / kBT)
    
    # 更新元数据
    metadata.update({
        'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'df': df,
        'temperature': temperature,
        'data_name': 'vdos_ang' if fft_data.ndim == 5 else 'vdos_trans',
        'data_type': 'vdos_ang' if fft_data.ndim == 5 else 'vdos_trans',
    })
    
    # 使用更新后的metadata中的data_name
    data_name = metadata['data_name']
    print(f"保存FFT数据，使用数据名称: {data_name}")
    save_dict = {
        data_name: fft_data,  # 使用更新后的data_name
        'metadata': np.array([str(metadata)])
    }
    if counts is not None:
        save_dict['counts'] = counts

    # 保存结果
    print(f"保存结果到: {fft_filename}")
    np.savez_compressed(fft_filename, **save_dict)
    
    print(f"处理完成。网格尺寸: {nx}x{ny}x{nz}")
    print(f"频率间隔: {df:.5f} cm^-1")

###################### 定义主函数 ##############################
def main():
    parser = argparse.ArgumentParser(description="处理VACF的npz文件并执行FFT")
    parser.add_argument("vacf_file", help="输入VACF的npz文件")
    parser.add_argument("fft_file", help="输出FFT的npz文件")
    parser.add_argument("--temperature", type=float, default=300, help="温度（K）")

    args = parser.parse_args()

    print(f"开始处理文件：{args.vacf_file}")
    print(f"输出文件：{args.fft_file}")
    print(f"温度 = {args.temperature} K")

    process_vacf_npz(
        args.vacf_file,
        args.fft_file,
        temperature=args.temperature
    )

if __name__ == "__main__":
    main()