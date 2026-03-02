import math
import sys
import time
import numpy as np
from tqdm import tqdm
from scipy.optimize import root_scalar
from GridAnalysis_multiF import save_grid_data_npz

########################### 函数定义 #####################################
def get_grid_center(grid_x, grid_y, grid_z, grid_spacings, box_lengths):
    # 计算网格中心坐标
    x_center = (grid_x + 0.5) * grid_spacings[0] - box_lengths[0]
    y_center = (grid_y + 0.5) * grid_spacings[1] - box_lengths[1]
    z_center = (grid_z + 0.5) * grid_spacings[2] - box_lengths[2]
    return np.array([x_center, y_center, z_center])

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

################################################################
# 定义常量
PI = math.pi
k = 1.3806488e-23  # 玻尔兹曼常数 （J/K）
lightspeed = 2.99792458e10  # 光速 (cm/s)
mass = 2.98897006e-26  # 水分子的质量 (kg)
planck = 6.62606957e-34  # 普朗克常数 (J/HZ)
Avogadro = 6.02214076e23  # 阿伏伽德罗常数 (mol-1)

# 检查命令行参数
if len(sys.argv) < 6:
    print(f"SYNTAX: {sys.argv[0]} {{VDOS npz file}} {{NumDens npz file}} [density] [temperature] [Qcharge]", file=sys.stderr)
    sys.exit(1)

vdos_file = sys.argv[1]
numdens_file = sys.argv[2]
one_dens = float(sys.argv[3])
temp = float(sys.argv[4])  # 从命令行参数读取温度
Qc = float(sys.argv[5]) # 从命令行参数读取离子电荷

kT = k * temp
debroglie = planck / math.sqrt(2 * PI * mass * kT)  # 德布罗伊波长 单位 m

########################### 读取VDOS数据 #####################################
print(f"读取VDOS文件: {vdos_file}")
with np.load(vdos_file) as vdos_data:
    vdos_metadata = eval(str(vdos_data['metadata'][0]))
    data_name = vdos_metadata.get('data_name', 'data')
    vdos_3d = vdos_data[data_name]
    df = vdos_metadata['df'] * lightspeed  # 转换为Hz
    MAX_FREQS = vdos_3d.shape[-1]

########################### 读取密度数据 #####################################
print(f"读取密度文件: {numdens_file}")
with np.load(numdens_file) as dens_data:
    dens_metadata = eval(str(dens_data['metadata'][0]))
    data_name = dens_metadata.get('data_name', 'data')
    dens_3d = dens_data[data_name]
    grid_spacings = np.array(dens_metadata['grid_spacings'])  # 单位应该是Angstrom
    grid_sizes = np.array(dens_metadata['grid_sizes'])
    nx, ny, nz = grid_sizes

########################### 初始化数组 #####################################
print(f"网格间距 = {grid_spacings} Angstrom")
print(f"网格大小 = {nx}x{ny}x{nz}")

# 计算每个方向的盒子半长度
box_lengths = grid_sizes * grid_spacings / 2
ngrid = nx * ny * nz

if one_dens >= 0.0:
    dens_3d.fill(one_dens * 1e27)  # 转换为 m^-3
else:
    dens_3d *= 1e27  # 转换为 m^-3

########################### 初始化结果数组 #####################################
entr_3d = np.zeros((nx, ny, nz))
entr_solid_3d = np.zeros((nx, ny, nz))
entr_gas_3d = np.zeros((nx, ny, nz))
f_3d = np.zeros((nx, ny, nz))
vdos_gas_3d = np.zeros_like(vdos_3d)
vdos_solid_3d = np.zeros_like(vdos_3d)

########################### 计算网格中心坐标和半径 #####################################
x = np.arange(nx)[:, np.newaxis, np.newaxis]
y = np.arange(ny)[np.newaxis, :, np.newaxis]
z = np.arange(nz)[np.newaxis, np.newaxis, :]

x_centers = (x + 0.5) * grid_spacings[0] - box_lengths[0]
y_centers = (y + 0.5) * grid_spacings[1] - box_lengths[1]
z_centers = (z + 0.5) * grid_spacings[2] - box_lengths[2]

r_centers = np.sqrt(x_centers**2 + y_centers**2 + z_centers**2)

########################### 主要计算循环 #####################################
total_iterations = nx * ny * nz

with tqdm(total=total_iterations, desc="处理平动VDOS") as pbar:
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                vdos_local = vdos_3d[ix,iy,iz]
                dens_local = dens_3d[ix,iy,iz]
                
                if dens_local <= 0:
                    pbar.update(1)  # 更新进度条
                    continue
                    
                norm = np.sum(vdos_local) * df
                if norm == 0:
                    continue
                    
                vdos_local = vdos_local / norm * 3.0
                
                g0 = vdos_local[0]
                if g0 <= 0:
                    continue
                    
                Delta = 2.0 * g0 / 9.0 * math.sqrt(PI * kT / mass) * dens_local ** (1.0 / 3.0) * (6.0 / PI) ** (2.0 / 3.0)
                
                zeta = 1.0
                f = find_f(Delta, zeta)
                if f > 1.0:
                    print("f out of bounds", file=sys.stderr)
                    continue
                # 储存f值
                f_3d[ix,iy,iz] = f

                # 计算平动熵
                y = (f / Delta) ** (1.5)
                zfy = (1.0 + f*y + (f*y)**2 - (f*y)**3) / ((1.0 - f*y)**3)
                #entr_cs = (f*y*(3.0*f*y - 4.0) / ((1.0 - f*y)**2))*k # 修订后的entr_cs (JCP 2017 SUN)
                entr_cs = (math.log(zfy) + f*y*(3.0*f*y - 4.0) / ((1.0 - f*y)**2))*k # 原entr_cs (JCP 2003 2PT)
                entr_st = (-math.log(f*dens_local*(debroglie**3)) + 2.5)*k
                w_gas = (entr_cs + entr_st) / (3.0*k)

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

                pbar.update(1)

########################### 保存结果 #####################################
# 基础元数据
base_metadata = {
    'temperature': temp,
    'Qcharge': Qc,
    'data_type': 'translational'  # 标识这是平动数据
}

# 保存熵
save_grid_data_npz(
    data_array=entr_3d,
    grid_sizes=grid_sizes,
    grid_spacings=grid_spacings,
    output_prefix='Strans_grid',
    data_name='Strans',
    **base_metadata
)

save_grid_data_npz(
    data_array=entr_gas_3d,
    grid_sizes=grid_sizes,
    grid_spacings=grid_spacings,
    output_prefix='Strans_gas_grid',
    data_name='Strans_gas',
    **base_metadata
)

save_grid_data_npz(
    data_array=entr_solid_3d,
    grid_sizes=grid_sizes,
    grid_spacings=grid_spacings,
    output_prefix='Strans_solid_grid',
    data_name='Strans_solid',
    **base_metadata
)

# 保存f值
save_grid_data_npz(
    data_array=f_3d,
    grid_sizes=grid_sizes,
    grid_spacings=grid_spacings,
    output_prefix='ftrans_grid',
    data_name='ftrans',
    **base_metadata
)

# 保存vdos_gas
save_grid_data_npz(
    data_array=vdos_gas_3d,
    grid_sizes=grid_sizes,
    grid_spacings=grid_spacings,
    output_prefix='vdos_gas_trans_grid',
    data_name='vdos_gas_trans',
    **base_metadata
)

# 保存vdos_solid
save_grid_data_npz(
    data_array=vdos_solid_3d,
    grid_sizes=grid_sizes,
    grid_spacings=grid_spacings,
    output_prefix='vdos_solid_trans_grid',
    data_name='vdos_solid_trans',
    **base_metadata
)

print("计算完成，已保存所有结果")