import math
import sys
import numpy as np
import time
from tqdm import tqdm
from scipy.optimize import root_scalar
from scipy.stats import gaussian_kde
from GridAnalysis_multiF import save_grid_data_npz

########################### 函数定义 #####################################
# 定义坐标计算函数
def get_grid_center(grid_x, grid_y, grid_z, grid_spacings, box_lengths):
    # 计算网格中心坐标
    x_center = (grid_x + 0.5) * grid_spacings[0] - box_lengths[0]
    y_center = (grid_y + 0.5) * grid_spacings[1] - box_lengths[1]
    z_center = (grid_z + 0.5) * grid_spacings[2] - box_lengths[2]
    return np.array([x_center, y_center, z_center])

# 计算f值
def func_zeta(Delta, x, zeta=1):
    return (2 / (Delta ** (9 / 2)) * x ** (3 + 9 * zeta / 2)
            - 6 / (Delta ** (3)) * x ** (2 + 3 * zeta)
            - 1 / (Delta ** (3 / 2)) * x ** (1 + 5 / 2 * zeta)
            + 6 / (Delta ** (3 / 2)) * x ** (1 + 3 / 2 * zeta)
            + 2 * x ** (zeta) - 2)
def find_f(Delta, zeta):
    try:
        return root_scalar(lambda x: func_zeta(Delta, x, zeta), bracket=[1e-10, 1], method='brentq').root
    except ValueError:
        return np.nan

# 定义取向熵修正函数
def Compute_S_rr(r, T=298.0, Q=1.0, r1=np.array([0., 0., 0.]), Q2=None, r2=None):
    """
    计算 S_O 随离子电荷 Q 和距离 r 的变化，考虑两个点电荷的矢量电场叠加。

    参数:
    - r (numpy.ndarray): 观察点的坐标数组 [x, y, z] (Å)
    - T (float): 温度 (K)，默认为 298 K
    - Q (float): 第一个离子的电荷量 (e)
    - r1 (numpy.ndarray): 第一个点电荷的坐标 [x, y, z] (Å)，默认在原点
    - Q2 (float): 第二个离子的电荷量 (e)
    - r2 (numpy.ndarray): 第二个点电荷的坐标 [x, y, z] (Å)

    返回:
    - S_O_r (numpy.ndarray): 对应 r 的 S_O 值 (J/mol)
    """
    # 定义常数
    k = 1.3806488e-23          # 玻尔兹曼常数 (J/K)
    epsilon0 = 8.854187817e-12 # 真空介电常数 (C²/(N·m²))
    e_charge = 1.602176634e-19 # 元电荷 (C)
    mu_debye = 1.854           # 水分子的偶极矩 (Debye)
    debye_to_CM = 3.33564e-30  # 1 Debye = 3.33564e-30 C·m
    mu = mu_debye * debye_to_CM # 偶极矩 (C·m)

    # 转换单位
    r_m = r * 1e-10  # 观察点坐标（米）
    r1_m = r1 * 1e-10  # 第一个电荷位置（米）
    # 计算第一个点电荷产生的电场（矢量）
    q1 = Q * e_charge
    r1_vec = r_m - r1_m  # 从电荷指向观察点的矢量
    r1_mag = np.linalg.norm(r1_vec)  # 距离大小
    r1_unit = r1_vec / r1_mag  # 单位方向向量
    # 计算电荷1电场
    E1_vec = (q1 / (4 * np.pi * epsilon0)) * r1_unit / (r1_mag**2)

    # 如果存在第二个点电荷，计算其产生的电场
    if Q2 is not None and r2 is not None:
        r2_m = r2 * 1e-10  # 第二个电荷位置（米）
        q2 = Q2 * e_charge
        # 计算第二个点电荷产生的电场（矢量）
        r2_vec = r_m - r2_m  # 从第二个电荷指向观察点的矢量
        r2_mag = np.linalg.norm(r2_vec)  # 距离大小
        r2_unit = r2_vec / r2_mag  # 单位方向向量
        # 计算电荷2介电常数
        E2_vec = (q2 / (4 * np.pi * epsilon0)) * r2_unit / (r2_mag**2)
        
        # 电场矢量叠加
        E_total_vec = E1_vec + E2_vec
        # 计算总电场大小
        E = np.linalg.norm(E_total_vec)
    else:
        # 只有一个电荷时的电场大小
        E = np.linalg.norm(E1_vec)

    # 计算 uE = μ · E
    uE = mu * E
    # 计算 y = uE / (k * T)
    y = uE / (3 * k * T)
    # 计算额外熵
    #f_y = np.log(np.sinh(y)) - np.log(y) - y * (np.cosh(y) / np.sinh(y)) + 1 
    #F_y = -5 * np.arctan(y / 27) * np.exp(-y ** 6 / 32)
    #entr_rr_E = k * f_y - k * F_y # J/K
    L_y = np.cosh(y) / np.sinh(y) - 1 / y  # 平均的取向计算
    F_y = -5 * np.arctan(y / 27) * np.exp(-y ** 6 / 32)  # 考虑水-水相互作用修正
    entr_rr_E = -k * np.log(np.pi/(2 * np.arccos(L_y))) - k * F_y

    return entr_rr_E

def compute_orientational_entropy_from_dipole(cos_dipole, Q=1.0):
    """
    从偶极取向角度计算熵修正
    
    参数:
    cos_dipole: np.ndarray
        偶极取向的cos值
    
    返回:
    float: 熵修正值 (J/K)
    """
    # 计算平均角度（弧度）
    angles = np.arccos(np.clip(cos_dipole, -1, 1))
    avg_angle = np.mean(angles)
    
    # 计算熵修正：S_cor = k*ln(a*2/pi)
    k = 1.3806488e-23  # 玻尔兹曼常数
    if Q >= 0:
      S_cor = k * np.log(avg_angle * 2 / np.pi)
    else:
      S_cor = k * np.log((np.pi - avg_angle) * 2 / np.pi)
    
    return S_cor

########################### 定义常量 ###########################
PI = np.pi
k = 1.3806488e-23  # 玻尔兹曼常数
Avogadro = 6.02214076e23  # 阿伏伽德罗常数
lightspeed = 2.99792458e10  # cm/s
mass = 2.98897006e-26  # 水分子的质量 (kg)
planck = 6.62606957e-34  # 普朗克常数
sym = 2  # 水分子的对称数 (rigid water)
mominert_1 = 0.99e-47  # 垂直水分子平面轴 
mominert_2 = 2.23e-47  # 偶极轴 
mominert_3 = 3.22e-47  # 转动惯量 (kg m^2) 
theta_1 = planck * planck / (8 * PI * PI * mominert_1 * k)
theta_2 = planck * planck / (8 * PI * PI * mominert_2 * k)
theta_3 = planck * planck / (8 * PI * PI * mominert_3 * k)
 
# 检查命令行参数
if len(sys.argv) < 5:
    print(f"SYNTAX: {sys.argv[0]} {{VDOS.npz}} {{NumDens.npz}} {{cos_dipole_file.npz}} [density] [temperature] [Qcharge]", file=sys.stderr)
    sys.exit(1)

# 读取命令行参数
vdos_file = sys.argv[1]
numdens_file = sys.argv[2]
cos_dipole_file = sys.argv[3]
one_dens = float(sys.argv[4])
temp = float(sys.argv[5])
Qc = float(sys.argv[6])
kT = k * temp

########################### 读取密度数据 #####################################
print(f"读取密度文件: {numdens_file}")
with np.load(numdens_file) as dens_data:
    dens_metadata = eval(str(dens_data['metadata'][0]))
    data_name = dens_metadata.get('data_name', 'data')
    dens_3d = dens_data[data_name]
    grid_spacings = np.array(dens_metadata['grid_spacings'])  # 单位应该是Angstrom
    grid_sizes = np.array(dens_metadata['grid_sizes'])
    nx, ny, nz = grid_sizes

########################### 读取VDOS数据 #####################################
print(f"读取VDOS文件: {vdos_file}")
with np.load(vdos_file) as vdos_data:
    vdos_metadata = eval(str(vdos_data['metadata'][0]))
    data_name = vdos_metadata.get('data_name', 'data')
    vdos_3d = vdos_data[data_name]  # shape: (grid_size, grid_size, grid_size, 3, n_delay)
    df = vdos_metadata['df'] * lightspeed  # 转换为Hz
    MAX_FREQS = vdos_3d.shape[-1]

########################### 读取cos_dipole_3d数据 #####################################
print(f"读取cos_dipole文件: {cos_dipole_file}")
with np.load(cos_dipole_file) as cos_dipole_data:
    cos_dipole_metadata = eval(str(cos_dipole_data['metadata'][0]))
    data_name = cos_dipole_metadata.get('data_name', 'data')
    cos_dipole_3d = cos_dipole_data[data_name]

########################### 初始化数组 #####################################
print(f"网格间距 = {grid_spacings} Angstrom")
print(f"网格大小 = {nx}x{ny}x{nz}")

# 计算每个方向的盒子半长度
box_lengths = grid_sizes * grid_spacings / 2

if one_dens >= 0.0:
    dens_3d.fill(one_dens * 1e27)  # 转换为 m^-3
else:
    dens_3d *= 1e27  # 转换为 m^-3

########################### 计算网格中心坐标和半径 #####################################
x = np.arange(nx)[:, np.newaxis, np.newaxis]
y = np.arange(ny)[np.newaxis, :, np.newaxis]
z = np.arange(nz)[np.newaxis, np.newaxis, :]

x_centers = (x + 0.5) * grid_spacings[0] - box_lengths[0]
y_centers = (y + 0.5) * grid_spacings[1] - box_lengths[1]
z_centers = (z + 0.5) * grid_spacings[2] - box_lengths[2]

r_centers = np.sqrt(x_centers**2 + y_centers**2 + z_centers**2)

########################### 初始化结果数组 #####################################
entr_3d = np.zeros((nx, ny, nz))  # 3个轴向的熵
entr_solid_3d = np.zeros((nx, ny, nz))
entr_gas_3d = np.zeros((nx, ny, nz))
f_3d = np.zeros((nx, ny, nz))
vdos_gas_3d = np.zeros((nx, ny, nz, MAX_FREQS))
vdos_solid_3d = np.zeros((nx, ny, nz, MAX_FREQS))

########################### 主要计算循环 #####################################
total_iterations = nx * ny * nz

with tqdm(total=total_iterations, desc="处理转动VDOS") as pbar:
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                # 获取当前网格点的密度
                dens_local = dens_3d[ix,iy,iz]
                if dens_local <= 0:
                    pbar.update(1)  # 更新进度条
                    continue
                # 获取当前网格点的偶极取向
                cos_dipole_local = cos_dipole_3d[ix,iy,iz]
                valid_cos_dipole_local = cos_dipole_local[~np.isnan(cos_dipole_local)]  # shape: (N,)
                # 处理当前网格点的三个主轴VDOS
                vdos_local = vdos_3d[ix,iy,iz].sum(axis=0)  # shape: (3, n_freqs) to (n_freqs,)
                norm = np.sum(vdos_local) * df  # 计算三个轴向的归一化因子
                if norm == 0:
                    continue
                # 归一化VDOS
                vdos_local = vdos_local / norm * 3.0
                g0 = vdos_local[0]  #零频率值
                if g0 <= 0:
                    continue
                # 计算三个轴向的Delta和f值
                Delta = 2.0 * g0 / 9.0 * math.sqrt(PI * kT / mass) * dens_local ** (1.0 / 3.0) * (6.0 / PI) ** (2.0 / 3.0)
                f = find_f(Delta, zeta=1)
                if f > 1.0:
                    print("f out of bounds", file=sys.stderr)
                    continue
                # 计算转动熵
                entr_rr = k * (1.5 + math.log(math.sqrt(PI) / sym * math.sqrt(temp ** 3 / (theta_1 * theta_2 * theta_3))))
                # 计算转动熵修正
                entr_rr_corr = Compute_S_rr(r = get_grid_center(ix,iy,iz, grid_spacings, box_lengths),
                                            T = temp,
                                            Q = Qc)
                #entr_rr_corr = compute_orientational_entropy_from_dipole(valid_cos_dipole_local, Q = Qc)
                # entr_rr_corr = 0.0
                w_gas = (entr_rr + entr_rr_corr) / 3.0 / k
                # 储存f值
                f_3d[ix,iy,iz] = f
                
                # 计算频率相关的熵贡献
                freqs = np.arange(MAX_FREQS)
                w_ho = np.zeros(MAX_FREQS)
                w_ho[1:] = planck * freqs[1:] * df / (kT * (np.exp(planck * freqs[1:] * df / kT) - 1.0)) - np.log(1.0 - np.exp(-planck * freqs[1:] * df / kT))
                
                # 计算气相和固相VDOS（向量化计算）
                vdos_gas = g0 / (1.0 + (g0 * freqs * df * PI / (6.0 * f)) ** 2)
                vdos_solid = vdos_local - vdos_gas
                
                vdos_gas_3d[ix,iy,iz] = vdos_gas
                vdos_solid_3d[ix,iy,iz] = vdos_solid
                
                # 计算熵贡献（向量化计算）
                entr_solid_3d[ix,iy,iz] = np.sum(w_ho * vdos_solid * df)
                entr_gas_3d[ix,iy,iz] = w_gas * np.sum(vdos_gas * df)
                
                # 计算总熵
                entr_3d[ix,iy,iz] = (k * Avogadro) * (entr_solid_3d[ix,iy,iz] + entr_gas_3d[ix,iy,iz])

                pbar.update(1)  # 更新进度条

########################### 保存结果 #####################################
# 基础元数据
base_metadata = {
    'temperature': temp,
    'Qcharge': Qc,
    'data_type': 'rotational'  # 标识这是转动数据
}

# 保存熵
save_grid_data_npz(
    data_array=entr_3d,
    grid_sizes=grid_sizes,
    grid_spacings=grid_spacings,
    output_prefix='Srot_grid',
    data_name='Srot',
    **base_metadata
)

save_grid_data_npz(
    data_array=entr_gas_3d,
    grid_sizes=grid_sizes,
    grid_spacings=grid_spacings,
    output_prefix='Srot_gas_grid',
    data_name='Srot_gas',
    **base_metadata
)

save_grid_data_npz(
    data_array=entr_solid_3d,
    grid_sizes=grid_sizes,
    grid_spacings=grid_spacings,
    output_prefix='Srot_solid_grid',
    data_name='Srot_solid',
    **base_metadata
)

# 保存f值
save_grid_data_npz(
    data_array=f_3d,
    grid_sizes=grid_sizes,
    grid_spacings=grid_spacings,
    output_prefix='frot_grid',
    data_name='frot',
    **base_metadata
)

# 分别保存气态和固态VDOS
save_grid_data_npz(
    data_array=vdos_gas_3d,
    grid_sizes=grid_sizes,
    grid_spacings=grid_spacings,
    output_prefix='vdos_gas_rot_grid',
    data_name='vdos_gas_rot',
    **base_metadata
)

save_grid_data_npz(
    data_array=vdos_solid_3d,
    grid_sizes=grid_sizes,
    grid_spacings=grid_spacings,
    output_prefix='vdos_solid_rot_grid',
    data_name='vdos_solid_rot',
    **base_metadata
)

print("计算完成，已保存所有结果")