#!/bin/bash
#SBATCH --partition=xahcnormal
#SBATCH --job-name=CYC_MD
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=md_job_%A_%a.out
#SBATCH --error=md_job_%A_%a.err
#SBATCH --exclusive ##独占

# 加载环境
module purge
module add compiler/intel/2017.5.239             
module add mpi/intelmpi/2017.4.239                
module add mathlib/fftw/3.3.8-intel-2017-double

# 设置环境变量
export PATH=/work/home/acnv1rnxcy/cyc_package/lammps-3Mar20/src:$PATH

# 获取当前作业的 run 编号
run_number=$SLURM_ARRAY_TASK_ID

echo "Starting MD simulation in run_$run_number"

# 检查目录是否存在
if [ ! -d "run_$run_number" ]; then
    echo "Error: Directory run_$run_number does not exist"
    exit 1
fi

cd run_$run_number || exit 1

mpirun -np $SLURM_CPUS_PER_TASK lmp_intel_cpu_intelmpi < simulation.in > simulation.out 2>&1

if [ $? -ne 0 ]; then
    echo "LAMMPS simulation failed in run_$run_number"
    exit 1
fi

echo "Ending MD simulation in run_$run_number"