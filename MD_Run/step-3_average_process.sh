#!/bin/bash
#SBATCH --job-name=3D2PT_avg      # 作业名称
#SBATCH --output=avg_%j.log       # 日志输出文件
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00           # 最大运行时间2小时
#SBATCH --partition=xahcnormal        # 根据实际分区修改
#SBATCH --exclusive ##独占

# 设置环境变量
export PYTHON_SCRIPT_PATH="/work/home/acnv1rnxcy/3D-2PT-main/python_3"

# 激活 MD2pt 环境
source /work/home/acnv1rnxcy/miniconda3/bin/activate MD2pt

Qcharge=-1
Temp=298

mkdir -p average

file_prefixes=(
    "vacf_trans_grid"
    "vacf_ang_grid"
    "density_grid"
    "cos_dipole_angles_grid"
    "principal_axes_grid"
    "tetrahedral_order_grid"
    "mol_energy_grid"
)
# 从环境变量获取运行范围（默认值1-5）
start_run=${START_RUN:-1}
end_run=${END_RUN:-40}

CURRENT_DIR=$(pwd)

echo "当前处理的运行范围：run_${start_run} 到 run_${end_run}"
for prefix in "${file_prefixes[@]}"; do
    echo "处理 ${prefix} 文件..."
    
    input_files=""
    for ((i=start_run; i<=end_run; i++)); do
        if [ -f "${CURRENT_DIR}/run_${i}/${prefix}.npz" ]; then
            input_files="${input_files} ${CURRENT_DIR}/run_${i}/${prefix}.npz"
        fi
    done
    
    if [ ! -z "$input_files" ]; then
        echo "正在平均 ${prefix} 文件..."
        python ${PYTHON_SCRIPT_PATH}/Average_npz.py "${CURRENT_DIR}/average/${prefix}" ${input_files}
    else
        echo "警告：没有找到 ${prefix}.npz 文件"
    fi
done

echo "所有文件平均完成"

echo "开始FFT变换和熵的计算"
cd average
python ${PYTHON_SCRIPT_PATH}/Vacf2fft_npz.py vacf_trans_grid_averaged.npz vdos_trans_grid_averaged.npz --temperature ${Temp}
python ${PYTHON_SCRIPT_PATH}/Vacf2fft_npz.py vacf_ang_grid_averaged.npz vdos_ang_grid_averaged.npz --temperature ${Temp}

python ${PYTHON_SCRIPT_PATH}/TransVdos2entr_npz.py vdos_trans_grid_averaged.npz density_grid_averaged.npz -33.4273 ${Temp} ${Qcharge}
python ${PYTHON_SCRIPT_PATH}/RotVdos2entr_npz.py vdos_ang_grid_averaged.npz density_grid_averaged.npz cos_dipole_angles_grid_averaged.npz -33.4273 ${Temp} ${Qcharge}

echo "熵的计算完成"
