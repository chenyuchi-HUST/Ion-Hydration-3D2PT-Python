#!/bin/bash
#SBATCH --partition=xahcnormal
#SBATCH --job-name=PythonAnalysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=python_job_%j.out
#SBATCH --error=python_job_%j.err
#SBATCH --exclusive ##独占

# 清空环境
module purge

# 设置环境变量
export PYTHON_SCRIPT_PATH="/work/home/acnv1rnxcy/3D-2PT-main/python_3"

# 激活 MD2pt 环境
source /work/home/acnv1rnxcy/miniconda3/bin/activate MD2pt

# 设置 Numba 线程数
# export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 显示 Numba 版本和并行配置
python -c "
import numba
import os

print(f'Numba version: {numba.__version__}')
print(f'Numba thread count: {numba.config.NUMBA_NUM_THREADS}')
"

# 获取当前作业的 run 编号
run_number=$RUN_NUMBER

echo "开始在 run_$run_number 中进行 Python 分析"

# 检查目录是否存在
if [ ! -d "run_$run_number" ]; then
    echo "错误：目录 run_$run_number 不存在"
    exit 1
fi

cd run_$run_number || exit 1

# 检查 GridAnalysis.input 文件是否存在
if [ ! -f "../GridAnalysis.input" ]; then
    echo "错误：GridAnalysis.input 文件不存在"
    exit 1
fi

python ${PYTHON_SCRIPT_PATH}/GridAnalysis_multiF.py ../GridAnalysis.input

if [ $? -ne 0 ]; then
    echo "Python 脚本在 run_$run_number 中执行失败"
    exit 1
fi

echo "在 run_$run_number 中完成 Python 分析"