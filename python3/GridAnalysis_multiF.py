###################### 导入必要的库 ##############################
import re
import os
import time
import sys
import numpy as np
from tqdm import tqdm
import numba as nb
from scipy.constants import Boltzmann, atomic_mass, angstrom
import multiprocessing as mp
# import MDAnalysis as mda
#################### 超算核数设定 ############################
# 获取环境变量中设置的线程数
# num_threads = int(os.environ.get('NUMBA_NUM_THREADS', 1))

# 设置Numba的线程数
# nb.set_num_threads(num_threads)
###################### 定义读取MD数据函数 ##############################
def find_frame_start_and_columns(file_name, atom_count, frames):
    """
    找到每一帧在文件中的起始字节位置，并确定每个原子的信息列数。
    
    参数:
    file_name : str : LAMMPS dump 文件的路径
    atom_count : int : 每帧中的原子数量
    frames : int : 总帧数
    
    返回:
    tuple : (frame_starts, n_columns)
        frame_starts : list : 每一帧的起始字节位置
        n_columns : int : 每个原子的信息列数
    """
    frame_starts = []
    n_columns = None
    
    with open(file_name, 'r') as f:
        for frame in range(frames):
            # 跳过每帧的前 9 行头信息
            for _ in range(9):
                line = f.readline()
                if not line:
                    break
            frame_start = f.tell()
            frame_starts.append(frame_start)
            
            # 读取第一行原子数据来确定列数
            if n_columns is None:
                first_line = f.readline()
                n_columns = len(first_line.split())
                # 回退一行
                f.seek(frame_start)
            
            # 跳过当前帧的原子数据
            for _ in range(atom_count):
                line = f.readline()
                if not line:
                    break              
    return frame_starts, n_columns
def read_single_frame(args):
    """
    读取单个帧的数据。
    
    参数:
    args : tuple : (file_name, frame_start, atom_count, n_columns)
    
    返回:
    np.ndarray : 形状为 (atom_count, n_columns) 的数组
    """
    file_name, frame_start, atom_count, n_columns = args
    frame_data = np.empty((atom_count, n_columns), dtype=float)
    
    with open(file_name, 'r') as f:
        f.seek(frame_start)
        for atom_idx in range(atom_count):
            line = f.readline()
            if not line:
                frame_data[atom_idx, :] = np.nan
                continue
            try:
                atom_info = [float(x) for x in line.split()]
                if len(atom_info) == n_columns:
                    frame_data[atom_idx, :] = atom_info
                else:
                    frame_data[atom_idx, :] = np.nan
            except ValueError:
                frame_data[atom_idx, :] = np.nan
    return frame_data
def read_lammps_dump_parallel(file_name, atom_count, frames, num_processes=None):
    """
    并行读取 LAMMPS dump 文件并返回包含原子信息的 NumPy 数组。
    
    参数:
    file_name : str : LAMMPS dump 文件的路径
    atom_count : int : 每帧中的原子数量
    frames : int : 总帧数
    num_processes : int, optional : 使用的进程数，默认使用 CPU 核心数
    
    返回:
    np.ndarray : 一个形状为 (atom_count, n_columns, frames) 的 NumPy 数组
    """
    # Step 1: 找到每一帧的起始位置和列数
    print("正在寻找每一帧的起始位置和列数...")
    frame_starts, n_columns = find_frame_start_and_columns(file_name, atom_count, frames)
    print(f"完成寻找帧的起始位置。每个原子有 {n_columns} 列数据。")
    
    # Step 2: 准备多进程读取参数
    pool_args = [(file_name, frame_start, atom_count, n_columns) 
                for frame_start in frame_starts]
    
    # Step 3: 创建多进程池并读取数据
    print("开始并行读取帧数据...")
    
    start_time = time.time()
    with mp.Pool(processes=num_processes) as pool:
        all_frames = list(tqdm(pool.imap(read_single_frame, pool_args), total=frames))
    end_time = time.time()
    
    print(f"完成所有帧的读取，耗时 {end_time - start_time:.2f} 秒。")
    
    # Step 4: 将数据转换为 NumPy 数组并转置
    data = np.array(all_frames)  # 形状为 (frames, atom_count, n_columns)
    data = data.transpose(1, 2, 0)  # 转置为 (atom_count, n_columns, frames)
    
    return data

def extract_box_lengths(data_file):
    """
    从 LAMMPS data 文件中提取 box 信息

    参数:
    data_file : str : LAMMPS data 文件的路径

    返回:
    tuple : 一个 (lx, ly, lz) 元组, 代表 box 的长度
    """ 
    xlo, xhi = None, None
    ylo, yhi = None, None
    zlo, zhi = None, None

    with open(data_file, 'r') as file:
        for line in file:
            if "xlo xhi" in line:
                xlo, xhi = map(float, line.split()[:2])
            elif "ylo yhi" in line:
                ylo, yhi = map(float, line.split()[:2])
            elif "zlo zhi" in line:
                zlo, zhi = map(float, line.split()[:2])
            if xlo is not None and ylo is not None and zlo is not None:
                break

    if None in [xlo, xhi, ylo, yhi, zlo, zhi]:
        raise ValueError("Could not find box dimensions in the data file.")
    
    # Calculate the lengths
    lx = xhi - xlo
    ly = yhi - ylo
    lz = zhi - zlo

    return lx, ly, lz
#################################################

###################### 定义计算 VACF 函数 ##############################
@nb.njit(fastmath=True)
def calculate_vacf(velocities):
    """
    计算单个分子的平均速度自相关函数 (VACF)
    
    参数:
    velocities : numpy.ndarray : 粒子的速度矩阵 (T, 3) T 是时间步数
    
    返回:
    numpy.ndarray : VACF 数组 (unit: A^2/ps^2)
    """
    if velocities.ndim != 2 or velocities.shape[0] <= 1:
        raise ValueError(f"Expected velocities to be 2D, but got shape {velocities.shape}")
    # 时间长度:
    n_times = velocities.shape[0]
    # 初始化 VACF 数组:
    vacf = np.zeros(n_times, dtype=np.float32)
    # 遍历每个时间：
    for j in range(n_times):
        vacf[j] += np.dot(velocities[0, :], velocities[j, :])
    return vacf

@nb.njit(fastmath=True)
def calculate_vacf_angular(velocities):
    """
    计算单个分子所有主轴方向角速度相关函数 (VACF)

    参数:
    velocities : numpy.ndarray : 粒子的速度矩阵 (T, 3), T 是时间步数
    
    返回:    
    numpy.ndarray : VACF 数组
    """
    if velocities.ndim != 2 or velocities.shape[0] <= 1:
        raise ValueError(f"Expected velocities to be 2D, but got shape {velocities.shape}")
    # 时间长度:
    n_times = velocities.shape[0]
    # 初始化三个主轴方向的 VACF 数组:
    vacf_1 = np.zeros(n_times, dtype=np.float32)
    vacf_2 = np.zeros(n_times, dtype=np.float32)
    vacf_3 = np.zeros(n_times, dtype=np.float32)
    # 遍历每个时间
    for j in range(n_times):
        vacf_1[j] += velocities[0, 0] * velocities[j, 0]
        vacf_2[j] += velocities[0, 1] * velocities[j, 1]
        vacf_3[j] += velocities[0, 2] * velocities[j, 2]
    # 返回三个方向的 VACF   
    return vacf_1, vacf_2, vacf_3
#################################################

###################### 定义一些辅助函数 ##############################
def get_grid_center(grid_x, grid_y, grid_z, grid_spacings, box_lengths):
    # 计算网格中心坐标
    x_center = (grid_x + 0.5) * grid_spacings[0] - box_lengths[0]
    y_center = (grid_y + 0.5) * grid_spacings[1] - box_lengths[1]
    z_center = (grid_z + 0.5) * grid_spacings[2] - box_lengths[2]
    return np.array([x_center, y_center, z_center])

@nb.njit(fastmath=True)
def apply_minimum_image_convention(water_positions, MD_box_length):
    """
    根据最小镜像法修正单个水分子的原子位置

    参数:
    water_positions : numpy.ndarray : 单个水分子中所有原子的坐标 (3, 3)（顺序为氧原子、氢原子、氢原子）
    MD_box_length : float : MD模拟盒子的长度

    返回:
    new_positions : numpy.ndarray : 修正后的原子坐标 (3, 3)
    """
    # 创建一个新的数组来存储修正后的位置
    new_positions = np.copy(water_positions)

    # 计算氢原子和氧原子的距离
    delta1 = new_positions[1, :] - new_positions[0, :]
    delta2 = new_positions[2, :] - new_positions[0, :]

    # 应用最小镜像约定
    for i in range(3):
        if delta1[i] > MD_box_length * 0.5:
            new_positions[1, i] -= MD_box_length
        elif delta1[i] < -MD_box_length * 0.5:
            new_positions[1, i] += MD_box_length
        
        if delta2[i] > MD_box_length * 0.5:
            new_positions[2, i] -= MD_box_length
        elif delta2[i] < -MD_box_length * 0.5:
            new_positions[2, i] += MD_box_length

    return new_positions
#################################################

###################### 定义计算水分子性质函数 ##############################
@nb.njit(fastmath=True)
def compute_water_properties(atom_matrix, solute_coord=None):
    """
    计算水分子的性质
    
    参数:
    atom_matrix: numpy 数组, 3*7 的矩阵, 其中每行代表一个原子:
                 前三列为 x, y, z 坐标，接下来的三列为 vx, vy, vz 速度，
                 最后一列为质量 mass。顺序为 O-H-H。
    solute_coord: numpy 数组, 溶质坐标 (x, y, z)
    
    返回:
    com: numpy 数组, 质心坐标 (x, y, z)
    vcom: numpy 数组, 质心速度 (x, y, z)
    principal_moments: numpy 数组, 三个主转动惯量
    angular_velocity: numpy 数组, 三个主轴方向上的角速度
    cos_dipole_angle: numpy 数组, 水分子与溶质连线的夹角(cos值)
    principal_axes: numpy 3*3数组, 水分子三个主轴
    """
    # 提取坐标、速度和质量
    coordinates = atom_matrix[:, :3].astype(np.float32)
    velocities = atom_matrix[:, 3:6].astype(np.float32)
    masses = atom_matrix[:, 6].astype(np.float32)
    
    # 假设三个原子分别为 O, H1, H2
    m_O, m_H = masses[0], masses[1]
    m_total = m_O + 2 * m_H
    
    # 计算质心和质心速度
    COM = (m_O * coordinates[0] + m_H * (coordinates[1] + coordinates[2])) / m_total
    vCOM = (m_O * velocities[0] + m_H * (velocities[1] + velocities[2])) / m_total

    # 初始化转动惯量张量和角动量
    inertia_tensor = np.zeros((3, 3), dtype=np.float32)
    angMom = np.zeros(3, dtype=np.float32)

    # 同时计算转动惯量张量和角动量
    for i in range(3):
        r_com = coordinates[i] - COM
        v_com = velocities[i] - vCOM
        
        # 计算转动惯量张量的贡献
        r_squared = np.dot(r_com, r_com)
        inertia_tensor += masses[i] * (r_squared * np.eye(3) - np.outer(r_com, r_com))
        
        # 计算角动量的贡献
        angMom += masses[i] * np.cross(r_com, v_com)

    # 计算主转动惯量和对应的主轴
    principal_moments, principal_axes = np.linalg.eigh(inertia_tensor)

    # 计算角动量在主轴上的投影
    angMomProj = np.dot(angMom, principal_axes)

    # 计算角速度
    angular_velocity = np.zeros(3, dtype=np.float32)
    for i in range(3):
        if principal_moments[i] > 1e-10:  # 避免除以零
            angular_velocity[i] = angMomProj[i] / principal_moments[i]

    # 如果提供了溶质坐标，计算偶极矩与氧原子-溶质连线的夹角
    if solute_coord is not None:
        # 计算水分子偶极矩（从氧原子指向H原子的中点）
        h_midpoint = (coordinates[1] + coordinates[2]) / 2
        dipole_vector = h_midpoint - coordinates[0]
        
        # 计算溶质到氧原子的向量
        o_solute_vector = coordinates[0] - solute_coord

        # 确保两个向量的数据类型一致
        dipole_vector = dipole_vector.astype(np.float32)
        o_solute_vector = o_solute_vector.astype(np.float32)
        
        # 计算两个向量的夹角
        dot_product = np.dot(dipole_vector, o_solute_vector)
        dipole_norm = np.dot(dipole_vector, dipole_vector)
        o_solute_norm = np.dot(o_solute_vector, o_solute_vector)
        cos_angle = dot_product / (np.sqrt(dipole_norm) * np.sqrt(o_solute_norm))

    return COM.astype(np.float32), vCOM.astype(np.float32), principal_moments.astype(np.float32), angular_velocity.astype(np.float32), cos_angle, principal_axes.astype(np.float32)

@nb.njit(fastmath=True, nogil=True)
def compute_tetrahedral_order(coords, box_length):
    """
    计算单个水分子的四面体序参数
    
    参数:
    coords: np.ndarray, shape=(N,3), 所有氧原子的坐标
    box_length: float, 模拟盒子的长度
    
    返回:
    q: float32, 四面体序参数
    neighbor_indices: np.array, shape=(4,), 最近的4个邻居的索引
    """
    # 确保数据类型
    coords = coords.astype(np.float32)
    box_length = np.float32(box_length)
    n_atoms = len(coords)
    distances = np.zeros(n_atoms, dtype=np.float32)
    
    # 计算与其他所有氧原子的距离
    for j in range(n_atoms):
        dx = np.float32(coords[0, 0] - coords[j, 0])
        dy = np.float32(coords[0, 1] - coords[j, 1])
        dz = np.float32(coords[0, 2] - coords[j, 2])
        
        # 周期性边界条件
        half_box = box_length / np.float32(2.0)
        if dx > half_box:
            dx -= box_length
        elif dx < -half_box:
            dx += box_length
            
        if dy > half_box:
            dy -= box_length
        elif dy < -half_box:
            dy += box_length
            
        if dz > half_box:
            dz -= box_length
        elif dz < -half_box:
            dz += box_length
            
        # 防止除零错误
        dist_sq = dx*dx + dy*dy + dz*dz
        if dist_sq > 1e-10:  # 添加一个小的阈值
            distances[j] = np.sqrt(dist_sq)
        else:
            distances[j] = np.float32(1e5)  # 给自身一个很大的距离
    
    # 找到最近的4个邻居
    neighbor_indices = np.argsort(distances)[1:5]  # 跳过自身
    
    # 计算四面体序参数
    q = np.float32(0.0)
    for k in range(3):
        for l in range(k+1, 4):
            # 计算两个邻居之间的夹角
            vec1 = coords[neighbor_indices[k]] - coords[0]
            vec2 = coords[neighbor_indices[l]] - coords[0]
            
            # 周期性边界条件
            for vec in [vec1, vec2]:
                for m in range(3):
                    if vec[m] > half_box:
                        vec[m] -= box_length
                    elif vec[m] < -half_box:
                        vec[m] += box_length
            
            # 计算夹角的余弦值，防止除零
            norm1 = np.sqrt(np.dot(vec1, vec1))
            norm2 = np.sqrt(np.dot(vec2, vec2))
            
            if norm1 > 1e-10 and norm2 > 1e-10:  # 防止除零
                cos_phi = np.dot(vec1, vec2) / (norm1 * norm2)
                cos_phi = min(max(cos_phi, np.float32(-1.0)), np.float32(1.0))
                q += np.float32((cos_phi + 1.0/3.0)**2)
            else:
                return np.nan, neighbor_indices  # 如果发现无效距离，返回NaN
    
    q = np.float32(1.0 - 3.0/8.0 * q)
    return q, neighbor_indices
#################################################

###################### 输出文件函数 ##############################
def write_cube_file(filename, title, grid_spacings, grid_sizes, data_array, avg_grid=True):
    """
    将数据写入 cube 文件格式。
    
    参数:
    filename : str : 输出文件的名称
    title : str : 文件头部的标题
    grid_spacings : numpy.ndarray : 每个方向的网格间隔 [dx, dy, dz]
    grid_sizes : numpy.ndarray : 网格的尺寸 [nx, ny, nz]
    data_array : numpy.ndarray : 需要输出的数据
    avg_grid : bool : 是否对每个网格的体积进行平均
    """
    # 获取盒子的半边长
    box_lengths = grid_sizes * grid_spacings / 2
    # 获取网格的原点坐标 （角落处）
    grid_origin = np.array([
        -box_lengths[0],
        -box_lengths[1],
        -box_lengths[2]
    ])
    
    with open(filename, 'w') as f:
        # 输出文件标题
        f.write(f"{title}\n")
        f.write("Generated by Yuchi Chen\n")
        # 输出原点坐标，默认为 1 个原子 (可根据实际情况修改)
        f.write(f"    1{grid_origin[0]/0.5292:12.6f}{grid_origin[1]/0.5292:12.6f}{grid_origin[2]/0.5292:12.6f}\n")
        # 输出网格尺寸和每个方向的步长
        f.write(f"{grid_sizes[0]:5d}{grid_spacings[0]/0.5292:12.6f}{0.0:12.6f}{0.0:12.6f}\n")
        f.write(f"{grid_sizes[1]:5d}{0.0:12.6f}{grid_spacings[1]/0.5292:12.6f}{0.0:12.6f}\n")
        f.write(f"{grid_sizes[2]:5d}{0.0:12.6f}{0.0:12.6f}{grid_spacings[2]/0.5292:12.6f}\n")

        # 输出虚拟的原子信息（这里假设没有原子，手动输出一个虚拟原子）
        f.write(f"8{0.0:12.6f}{0.0:12.6f}{0.0:12.6f}{0.0:12.6f}\n")
        
        # 输出网格数据
        nx, ny, nz = grid_sizes
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # 将数据值写入文件
                    if avg_grid:
                        value = data_array[i, j, k]/(0.001*np.prod(grid_spacings))
                    else:
                        value = data_array[i, j, k]
                    
                    if k % 6 == 0:
                        f.write(f"{value:.6e}")
                    else:
                        f.write(f" {value:.6e}")
                    # 每行最多写 6 个数据
                    if (k+1) % 6 == 0:
                        f.write("\n")
                if (nz % 6) != 0:
                    f.write("\n")

def save_grid_data_npz(data_array, grid_sizes, grid_spacings, output_prefix, counts_array=None, data_name="data", dt=0.008, **additional_metadata):
    """
    通用的网格数据存储函数，将数据保存为压缩的npz文件
    
    参数:
    data_array: np.ndarray, 要储存的主要数据数组
    grid_sizes: numpy.ndarray, 三个方向的网格数量 [nx, ny, nz]
    grid_spacings: numpy.ndarray, 三个方向的网格间距 [dx, dy, dz]
    output_prefix: str, 输出文件前缀
    counts_array: np.ndarray, 可选, 计数数组，形状为(grid_sizes[0], grid_sizes[1], grid_sizes[2])
    data_name: str, 可选, 数据在npz文件中的键名，默认为"data"
    dt: float, 可选, 时间步长，默认为0.008
    **additional_metadata: 其他要包含在元数据中的键值对
    """
    # 创建基础元数据字典
    metadata = {
        'grid_sizes': grid_sizes.tolist() if isinstance(grid_sizes, np.ndarray) else grid_sizes,
        'grid_spacings': grid_spacings.tolist() if isinstance(grid_spacings, np.ndarray) else grid_spacings,
        'dt': dt,
        'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'array_shape': data_array.shape,
        'data_name': data_name
    }
    
    # 添加额外的元数据
    metadata.update(additional_metadata)
    
    # 准备保存的数据字典
    save_dict = {
        data_name: data_array,
        'metadata': np.array([str(metadata)])
    }
    
    # 如果提供了计数数组，则添加到保存字典中
    if counts_array is not None:
        save_dict['counts'] = counts_array
    
    # 保存数据和元数据
    np.savez_compressed(f'{output_prefix}.npz', **save_dict)
##############################################################

########################## 分析单帧数函数 ###########################
@nb.njit(fastmath=True, nogil=True)
def process_frames(all_frames_data, frame_idx, grid_sizes, box_lengths, grid_spacings, o_atom_type, MD_box_length, n_delay):
    """
    处理单个帧及其后续帧的数据，计算水分子的属性。

    输入参数:
    all_frames_data: numpy.ndarray, 所有帧的原子数据
    frame_idx: int, 当前帧的索引
    grid_sizes: numpy.ndarray, 三个方向的网格数量 [nx, ny, nz]
    box_lengths: numpy.ndarray, 三个方向的网格盒子半长度 [lx/2, ly/2, lz/2]
    grid_spacings: numpy.ndarray, 三个方向的网格间距 [dx, dy, dz]
    o_atom_type: int, 氧原子的类型标识
    MD_box_length: float, 模拟盒子的长度
    n_delay: int, 需要处理的后续帧数

    输出:
    grid_indices: numpy.ndarray (n_molecules, 3), 水分子所在的网格索引
    vCOMs_all: numpy.ndarray (n_molecules, n_delay, 3), 所有水分子在当前帧及后续帧的质心速度
    angular_velocities_all: numpy.ndarray (n_molecules, n_delay, 3), 所有水分子在当前帧及后续帧的角速度
    principal_moments: numpy.ndarray (3), 当前帧第一个水分子的主转动惯量
    dipole_angle: numpy.ndarray (n_molecules), 当前帧所有水分子与溶质连线的夹角(cos值)
    principal_axes: numpy.ndarray (n_molecules, 3, 3), 当前帧所有水分子主轴
    tetrahedral_order: numpy.ndarray (n_molecules), 当前帧所有水分子四面体序参数
    mol_energy: numpy.ndarray (n_molecules), 当前帧所有水分子势能
    """
    # 获取当前帧数据
    frame_data = all_frames_data[:, :, frame_idx]
    # 原子属性数量
    n_columns = frame_data.shape[1]
    # 提取氧原子、氢原子和溶质的数据
    o_atoms = frame_data[frame_data[:, 2] == o_atom_type]
    h_atoms = frame_data[frame_data[:, 2] == o_atom_type + 1]
    #solute_atoms = frame_data[frame_data[:, 2] == o_atom_type + 2]
    #solute_coord = solute_atoms[:, 3:6].astype(np.float32) # 溶质的坐标
    solute_coord = np.array([0.0, 0.0, 0.0], dtype=np.float32) 
    # 预分配数组
    n_o_atoms = len(o_atoms)
    water_molecules = np.empty((n_o_atoms, 3, n_columns), dtype=np.float32)
    valid_molecule = np.zeros(n_o_atoms, dtype=np.bool_)

    # 处理每个氧原子和氢原子对
    for i in range(n_o_atoms):
        o_atom = o_atoms[i]
        h_atoms_of_molecule = h_atoms[h_atoms[:, 1] == o_atom[1]]
        if len(h_atoms_of_molecule) == 2:
            molecule = np.empty((3, n_columns), dtype=np.float32)
            molecule[0, :] = o_atom.astype(np.float32)
            molecule[1, :] = h_atoms_of_molecule[0].astype(np.float32)
            molecule[2, :] = h_atoms_of_molecule[1].astype(np.float32)
            molecule[:, 3:6] = apply_minimum_image_convention(molecule[:, 3:6], MD_box_length)
            water_molecules[i, :, :] = molecule
            valid_molecule[i] = True

    # 提取有效的水分子
    water_molecules = water_molecules[valid_molecule]

    # 提取所有氧原子坐标
    o_coords = water_molecules[:, 0, 3:6]  # 只取氧原子的坐标
    
    # 计算当前帧的水分子属性
    n_molecules = len(water_molecules)
    COMs = np.empty((n_molecules, 3), dtype=np.float32)
    vCOMs_current = np.empty((n_molecules, 3), dtype=np.float32)
    angular_velocities_current = np.empty((n_molecules, 3), dtype=np.float32)
    mol_ids = np.empty(n_molecules, dtype=np.int32)
    grid_indices = np.empty((n_molecules, 3), dtype=np.int32)
    valid_indices = np.ones(n_molecules, dtype=np.bool_)
    dipole_angle = np.empty(n_molecules, dtype=np.float32)
    principal_axes = np.empty((n_molecules, 3, 3), dtype=np.float32)
    # 只计算一次主转动惯量
    principal_moments = compute_water_properties(water_molecules[0][:, 3:10], solute_coord=solute_coord)[2]
    # 四面体序参数
    tetrahedral_order = np.empty(n_molecules, dtype=np.float32)
    # 原子势能
    mol_energy = np.empty(n_molecules, dtype=np.float32)
    
    # 循环每个水分子
    for i in range(n_molecules): 
        prop = compute_water_properties(water_molecules[i][:, 3:10], solute_coord=solute_coord)
        COMs[i, :] = prop[0]
        vCOMs_current[i, :] = prop[1]
        angular_velocities_current[i, :] = prop[3]
        dipole_angle[i] = prop[4]
        principal_axes[i, :, :] = prop[5]
        mol_ids[i] = water_molecules[i][0, 1]
        # 四面体序参数计算：
        coords = np.empty((n_molecules, 3), dtype=np.float32)
        if i > 0:  # 如果不是第一个分子
            coords[1:i+1] = o_coords[:i]  # 前面的分子
        if i < n_molecules - 1:  # 如果不是最后一个分子
            coords[i+1:] = o_coords[i+1:]  # 后面的分子
        coords[0] = o_coords[i]  # 当前水分子放在第一位
        q, _ = compute_tetrahedral_order(coords, MD_box_length)
        tetrahedral_order[i] = q
        # 输出每个水分子势能
        mol_energy[i] = np.float32(np.sum(water_molecules[i][:, -1]))
        # 确定每个水分子所在的网格索引
        grid_indices[i, 0] = int((COMs[i, 0] + box_lengths[0]) // grid_spacings[0])
        grid_indices[i, 1] = int((COMs[i, 1] + box_lengths[1]) // grid_spacings[1])
        grid_indices[i, 2] = int((COMs[i, 2] + box_lengths[2]) // grid_spacings[2])
        # 检查网格索引是否有效
        if (grid_indices[i, 0] < 0 or grid_indices[i, 0] >= grid_sizes[0] or
            grid_indices[i, 1] < 0 or grid_indices[i, 1] >= grid_sizes[1] or
            grid_indices[i, 2] < 0 or grid_indices[i, 2] >= grid_sizes[2]):
            valid_indices[i] = False

    # 只保留在网格划分内的有效水分子数据
    grid_indices = grid_indices[valid_indices]
    vCOMs = vCOMs_current[valid_indices]
    angular_velocities = angular_velocities_current[valid_indices]
    mol_ids = mol_ids[valid_indices]
    dipole_angle = dipole_angle[valid_indices]
    principal_axes = principal_axes[valid_indices]
    tetrahedral_order = tetrahedral_order[valid_indices]
    mol_energy = mol_energy[valid_indices]

    # 预分配数组，获取后续n_delay-1帧的数据，主要是速度和角速度
    n_molecules = len(mol_ids) # Grid内数量
    vCOMs_all = np.full((n_molecules, n_delay, 3), np.nan, dtype=np.float32)
    angular_velocities_all = np.full((n_molecules, n_delay, 3), np.nan, dtype=np.float32)
    vCOMs_all[:, 0, :] = vCOMs
    angular_velocities_all[:, 0, :] = angular_velocities
    for delay in range(1, n_delay):
        if frame_idx + delay < all_frames_data.shape[2]:
            next_frame_data = all_frames_data[:, :, frame_idx + delay]
            for i, mol_id in enumerate(mol_ids):
                mol_atoms = next_frame_data[next_frame_data[:, 1] == mol_id]
                if len(mol_atoms) == 3:
                    mol_atoms = mol_atoms[np.argsort(mol_atoms[:, 2])]  # 氧原子排在第一个
                    mol_atoms[:, 3:6] = apply_minimum_image_convention(mol_atoms[:, 3:6], MD_box_length)
                    prop = compute_water_properties(mol_atoms[:, 3:10], solute_coord=solute_coord)
                    vCOMs_all[i, delay, :] = prop[1]
                    angular_velocities_all[i, delay, :] = prop[3]

    return grid_indices, vCOMs_all, angular_velocities_all, principal_moments, dipole_angle, principal_axes, tetrahedral_order, mol_energy

########################## 分析多帧数函数（并行） ###########################
@nb.njit(parallel=True, fastmath=True)
def process_frames_batch(frames_to_analyze, start_frame, end_frame, grid_sizes, box_lengths, grid_spacings, o_atom_type, MD_box_length, n_delay, max_angles_per_grid):
    """批量处理多个帧的数据
    
    参数:
    frames_to_analyze: numpy.ndarray, 所有帧的原子数据
    start_frame: int, 起始帧索引
    end_frame: int, 结束帧索引
    grid_sizes: numpy.ndarray, 三个方向的网格数量 [nx, ny, nz]
    box_lengths: numpy.ndarray, 三个方向的网格盒子半长度 [lx/2, ly/2, lz/2]
    grid_spacings: numpy.ndarray, 三个方向的网格间距 [dx, dy, dz]
    o_atom_type: int, 氧原子的类型标识
    MD_box_length: float, 模拟盒子的长度
    n_delay: int, 延迟时间步数
    max_angles_per_grid: int, 每个网格存储的最大角度数
    """
    nx, ny, nz = grid_sizes
    vacf_trans_grid = np.zeros((nx, ny, nz, n_delay), dtype=np.float32)
    vacf_ang_grid = np.zeros((nx, ny, nz, 3, n_delay), dtype=np.float32)
    avg_particle_count = np.zeros((nx, ny, nz), dtype=np.int32)
    principal_moments_array = np.zeros((end_frame - start_frame, 3), dtype=np.float32) 
    # 预分配数组
    dipole_angles_grid = np.full((nx, ny, nz, max_angles_per_grid), np.nan, dtype=np.float32)
    principal_axes_grid = np.full((nx, ny, nz, max_angles_per_grid, 3, 3), np.nan, dtype=np.float32)
    tetrahedral_order_grid = np.full((nx, ny, nz, max_angles_per_grid), np.nan, dtype=np.float32)
    mol_energy_grid = np.full((nx, ny, nz, max_angles_per_grid), np.nan, dtype=np.float32)
    
    for frame_idx in nb.prange(start_frame, end_frame):
        grid_indices, vCOMs, angular_velocities, principal_moments, dipole_angles, principal_axes, tetrahedral_order, mol_energy = process_frames(
            frames_to_analyze, frame_idx, grid_sizes, box_lengths, grid_spacings,
            o_atom_type, MD_box_length, n_delay
        )
        
        # 计算每帧的 principal_moments
        principal_moments_array[frame_idx - start_frame] = principal_moments    
        
        # 计算每个网格点的数据  
        for i in range(len(grid_indices)):
            x, y, z = grid_indices[i]
            vacf_trans = calculate_vacf(vCOMs[i])
            vacf_ang1, vacf_ang2, vacf_ang3 = calculate_vacf_angular(angular_velocities[i])        
            vacf_trans_grid[x, y, z, :] += vacf_trans
            vacf_ang_grid[x, y, z, 0, :] += vacf_ang1
            vacf_ang_grid[x, y, z, 1, :] += vacf_ang2
            vacf_ang_grid[x, y, z, 2, :] += vacf_ang3      
            
            # 存储其他属性
            current_count = avg_particle_count[x, y, z]
            if current_count < max_angles_per_grid:
                dipole_angles_grid[x, y, z, current_count] = dipole_angles[i]
                principal_axes_grid[x, y, z, current_count, :, :] = principal_axes[i, :, :]
                tetrahedral_order_grid[x, y, z, current_count] = tetrahedral_order[i]
                mol_energy_grid[x, y, z, current_count] = mol_energy[i]
                avg_particle_count[x, y, z] += 1

    return vacf_trans_grid, vacf_ang_grid, avg_particle_count, principal_moments_array, dipole_angles_grid, principal_axes_grid, tetrahedral_order_grid, mol_energy_grid

########################## 分析网格函数 ###########################
def analyze_grid(matrix, frame_start, n_frames, n_delay, grid_sizes, grid_spacings, o_atom_type, MD_box_length, max_angles_per_grid, batch_size=100):
    """
    分析网格数据，计算速度自相关函数和角动量投影的自相关函数。

    参数:
    matrix: 包含所有帧数据的矩阵
    frame_start: 起始帧索引
    n_frames: 要分析的总帧数
    n_delay: 延迟时间步数
    grid_sizes: numpy.ndarray, 三个方向的网格数量 [nx, ny, nz]
    grid_spacings: numpy.ndarray, 三个方向的网格间距 [dx, dy, dz]
    o_atom_type: 氧原子类型
    MD_box_length: MD模拟盒子长度
    max_angles_per_grid: 每个网格存储的最大角度数
    batch_size: 批处理大小

    返回:
    vacf_trans_grid: 平动速度自相关函数网格
    vacf_ang_grid: 角速度自相关函数网格
    avg_particle_count: 平均粒子数网格
    all_principal_moments: 所有帧第一个水分子的主转动惯量
    all_dipole_angles: 所有偶极角度数组
    all_principal_axes: 所有主轴方向数组
    all_tetrahedral_order: 所有四面体序参数数组
    all_mol_energy: 所有势能数组
    """
    # 计算每个方向的盒子半长度
    nx, ny, nz = grid_sizes
    box_lengths = np.array([
        nx * grid_spacings[0] / 2,
        ny * grid_spacings[1] / 2,
        nz * grid_spacings[2] / 2
    ], dtype=np.float32)
    frames_to_analyze = matrix[:, :, frame_start:frame_start + n_frames]
    
    total_frames = n_frames - n_delay + 1
    vacf_trans_grid = np.zeros((nx, ny, nz, n_delay), dtype=np.float32)
    vacf_ang_grid = np.zeros((nx, ny, nz, 3, n_delay), dtype=np.float32)
    avg_particle_count = np.zeros((nx, ny, nz), dtype=np.int32)
    all_principal_moments = np.zeros((total_frames, 3), dtype=np.float32)
    # 初始化存储所有偶极角度的数组
    all_dipole_angles = np.full((nx, ny, nz, max_angles_per_grid), np.nan, dtype=np.float32)
    all_principal_axes = np.full((nx, ny, nz, max_angles_per_grid, 3, 3), np.nan, dtype=np.float32)
    all_tetrahedral_order = np.full((nx, ny, nz, max_angles_per_grid), np.nan, dtype=np.float32)
    all_mol_energy = np.full((nx, ny, nz, max_angles_per_grid), np.nan, dtype=np.float32)

    #计算总的batch数，其中不够整除的也考虑 
    num_batches = (total_frames + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="处理帧："):
        start_frame = batch_idx * batch_size
        end_frame = min(start_frame + batch_size, total_frames)
        
        batch_vacf_trans, batch_vacf_ang, batch_avg_particle, batch_principal_moments, batch_dipole_angles, batch_principal_axes, batch_tetrahedral_order, batch_mol_energy = process_frames_batch(
            frames_to_analyze, start_frame, end_frame, grid_sizes, box_lengths, grid_spacings,
            o_atom_type, MD_box_length, n_delay, max_angles_per_grid
        )
        
        # 更新累积结果
        vacf_trans_grid += batch_vacf_trans
        vacf_ang_grid += batch_vacf_ang
        avg_particle_count += batch_avg_particle
        all_principal_moments[start_frame:end_frame] = batch_principal_moments
        
        # 合并各种属性数据
        for i in range(grid_sizes[0]):
            for j in range(grid_sizes[1]):
                for k in range(grid_sizes[2]):
                    # 对其他属性的存储进行限制
                    current_count = np.sum(~np.isnan(all_dipole_angles[i,j,k]))  # 当前已存储的数量
                    new_count = batch_avg_particle[i,j,k]
                    if new_count > 0:
                        # 确保不超过最大容量
                        space_left = max_angles_per_grid - current_count
                        data_to_add = min(new_count, space_left)
                        if data_to_add > 0:
                            slice_range = slice(current_count, current_count + data_to_add)
                            # 存储各种属性
                            all_dipole_angles[i,j,k,slice_range] = batch_dipole_angles[i,j,k,:data_to_add]
                            all_principal_axes[i,j,k,slice_range, :, :] = batch_principal_axes[i,j,k,:data_to_add, :, :]
                            all_tetrahedral_order[i,j,k,slice_range] = batch_tetrahedral_order[i,j,k,:data_to_add]
                            all_mol_energy[i,j,k,slice_range] = batch_mol_energy[i,j,k,:data_to_add]

    return vacf_trans_grid, vacf_ang_grid, avg_particle_count, all_principal_moments, all_dipole_angles, all_principal_axes, all_tetrahedral_order, all_mol_energy
##############################################################

###################### 主函数定义！ #########################
def main(input_file):
    print("Start happy computing! Written by Yuchi Chen @ 2025.4.27")

    # 读取参数
    params = read_input_file(input_file)
    print_params(params)

    # MD轨迹读取
    print("Reading MD trajectory...")
    dump_data = read_lammps_dump_parallel(params['file_path'], atom_count=params['num_particles'], frames=params['total_frames'])
    
    # 速度单位转换 A/fs > A/ps
    dump_data[:, 6:9, :] *= 1000  # vx,vy,vz: A/fs > A/ps
    #np.save('dump_data.npy', dump_data)  # 保存为npy文件
    print(f"The MD trajectory has been read successfully! The shape of MD dump_data is {dump_data.shape}")
    
    # 从最后一帧的data文件中获取盒子信息
    MD_box_length = extract_box_lengths(params['data_file_path'])  # in angstroms
    print(f"The MD box length is {MD_box_length} angstroms")

    # 执行analyze_grid函数，分析dump文件，生成相关的统计量
    print("Start analyzing grid data...")
    vacf_trans_grid, vacf_ang_grid, avg_particle_count, all_principal_moments, dipole_angles_grid, all_principal_axes, all_tetrahedral_order, all_mol_energy = analyze_grid(
        matrix=dump_data,
        frame_start=params['frame_start'],
        n_frames=params['n_frames'],
        n_delay=params['n_delay'],
        grid_sizes=params['grid_sizes'],
        grid_spacings=params['grid_spacings'],
        o_atom_type=params['o_atom_type'],
        MD_box_length=MD_box_length[0],
        max_angles_per_grid=params['max_angles_per_grid'],
        batch_size=params['chunk_size']
    )
    print("Grid data analysis completed!")

    # 水分子的质量和主转动惯量
    #water_mass = 18.01528  # 水分子的质量，单位为原子质量单位(amu)
    #water_principal_moments = all_principal_moments[0] # 单位 (amu * A2)
   
    # 输出VACF文件
    save_grid_data_npz(
        data_array=vacf_trans_grid,
        grid_sizes=params['grid_sizes'],
        grid_spacings=params['grid_spacings'],
        output_prefix='vacf_trans_grid',
        counts_array=avg_particle_count,
        data_name='vacf_trans',
        data_type='vacf_trans',# 额外的元数据
        dt=params['dt']  
    )
    save_grid_data_npz(
        data_array=vacf_ang_grid,
        grid_sizes=params['grid_sizes'],
        grid_spacings=params['grid_spacings'],
        output_prefix='vacf_ang_grid',
        counts_array=avg_particle_count,
        data_name='vacf_ang',
        data_type='vacf_ang',
        dt=params['dt']
    )

    # 输出平均粒子数
    save_grid_data_npz(
        data_array=avg_particle_count/(params['n_frames']-params['n_delay']+1)/(0.001*np.prod(params['grid_spacings'])),
        grid_sizes=params['grid_sizes'],
        grid_spacings=params['grid_spacings'],
        output_prefix='density_grid',
        counts_array=None,
        data_name='density',
        data_type='density',
        dt=params['dt']
    )
    
    # 设置数量限制
    limited_counts = np.minimum(avg_particle_count, params['max_angles_per_grid'])
    # 保存偶极矩数据
    save_grid_data_npz(
        data_array=dipole_angles_grid,
        grid_sizes=params['grid_sizes'],
        grid_spacings=params['grid_spacings'],
        output_prefix='cos_dipole_angles_grid',
        counts_array=limited_counts,
        data_name='cos_dipole_angles',
        data_type='cos_dipole_angles',
        dt=params['dt']
    )

    # 保存主轴数据
    save_grid_data_npz(
        data_array=all_principal_axes,
        grid_sizes=params['grid_sizes'],
        grid_spacings=params['grid_spacings'],
        output_prefix='principal_axes_grid',
        counts_array=limited_counts,
        data_name='principal_axes',
        data_type='principal_axes',
        dt=params['dt']
    )

    # 保存四面体序参数
    save_grid_data_npz(
        data_array=all_tetrahedral_order,
        grid_sizes=params['grid_sizes'],
        grid_spacings=params['grid_spacings'],
        output_prefix='tetrahedral_order_grid',
        counts_array=limited_counts,
        data_name='tetrahedral_order',
        data_type='tetrahedral_order',
        dt=params['dt']
    )

    # 输出势能文件
    save_grid_data_npz(
        data_array=all_mol_energy,
        grid_sizes=params['grid_sizes'],
        grid_spacings=params['grid_spacings'],
        output_prefix='mol_energy_grid',
        counts_array=limited_counts,
        data_name='mol_energy',
        data_type='mol_energy',
        dt=params['dt']
    )
    
    print("Finished writing results to file!")

############################ 额外函数（可优化） #########################
def read_input_file(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    non_comment_lines = [line.strip() for line in lines if not line.strip().startswith('#')]

    # 解析网格大小和间距
    # 格式：nx,ny,nz 和 dx,dy,dz
    grid_sizes = [int(x) for x in non_comment_lines[7].split(',')]
    grid_spacings = [float(x) for x in non_comment_lines[8].split(',')]
    
    # 确保网格大小和间距都是三维的
    if len(grid_sizes) == 1:
        grid_sizes = [grid_sizes[0], grid_sizes[0], grid_sizes[0]]  # 如果只提供一个值，则应用于所有维度
    if len(grid_spacings) == 1:
        grid_spacings = [grid_spacings[0], grid_spacings[0], grid_spacings[0]]  # 如果只提供一个值，则应用于所有维度
    
    # 转换为NumPy数组
    grid_sizes = np.array(grid_sizes, dtype=np.int32)
    grid_spacings = np.array(grid_spacings, dtype=np.float32)

    return {
        'file_path': non_comment_lines[0],
        'total_frames': int(non_comment_lines[1]),
        'num_particles': int(non_comment_lines[2]),
        'data_file_path': non_comment_lines[3],
        'frame_start': int(non_comment_lines[4]),
        'n_frames': int(non_comment_lines[5]),
        'n_delay': int(non_comment_lines[6]),
        'grid_sizes': grid_sizes,
        'grid_spacings': grid_spacings,
        'o_atom_type': int(non_comment_lines[9]),
        'dt': float(non_comment_lines[10]),
        'chunk_size': int(non_comment_lines[11]),
        'max_angles_per_grid': int(non_comment_lines[12])
    }

def print_params(params):
    print("Reading the input file is completed:")
    print((f"The traj file is {params['file_path']}\n"
           f"The total frames is {params['total_frames']}\n"
           f"the number of particles is {params['num_particles']}\n"
           f"the data file path is {params['data_file_path']}\n"
           f"the start frame is {params['frame_start']}\n"
           f"the analyzed frame is {params['n_frames']}\n"
           f"the delay frame is {params['n_delay']}\n"
           f"the grid sizes is {params['grid_sizes']}\n"
           f"the grid spacings is {params['grid_spacings']}\n"
           f"the o atom type is {params['o_atom_type']}\n"
           f"the dt is {params['dt']}\n"
           f"the chunk size is {params['chunk_size']}\n"
           f"the max angles per grid is {params['max_angles_per_grid']}"))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python GridAnalysis.py <input_file>")
        sys.exit(1)
    main(sys.argv[1])
##################### end ##############################




