#!/bin/bash

N=${1:-2}  # 使用命令行参数或默认值 2

# 获取当前最大的 run 编号
get_max_run_number() {
    local max_run=0
    for dir in run_*; do
        if [[ -d "$dir" ]]; then
            num=${dir#run_}
            if [[ $num =~ ^[0-9]+$ ]] && [ $num -gt $max_run ]; then
                max_run=$num
            fi
        fi
    done
    echo $max_run
}

current_max=$(get_max_run_number)
start_run=$((current_max + 1))
end_run=$((start_run + N - 1))

# 直接执行准备脚本
echo "准备创建 $N 个新的 run 文件夹"
echo "将创建从 run_$start_run 到 run_$end_run 的新运行环境"

# 检查必要的文件是否存在
if [ ! -f "simulation.in" ]; then
    echo "错误：必要的文件 simulation.in 不存在。"
    exit 1
fi

# 创建新的 run 文件夹并复制文件
for i in $(seq $start_run $end_run); do
    echo "正在创建 run_$i ..."
    mkdir -p run_$i
    cp simulation.in run_$i/
    cp ./run_init/system_postNPT.data run_$i/ # 将初始运行环境的 system_postNPT.data 复制到新的 run 文件夹
    # cp system.in.settings run_$i/
    
    # 修改 simulation.in 中的随机种子
    new_seed=$((12345 + i))
    sed -i.bak "s/\(variable RandomSeed equal \)[0-9]*/\1$new_seed/" run_$i/simulation.in
    if [ $? -ne 0 ]; then
        echo "警告：无法修改 run_$i/simulation.in 中的随机种子"
    else
        echo "已更新 run_$i/simulation.in 中的随机种子为 $new_seed"
    fi
    rm -f run_$i/simulation.in.bak
    
    echo "完成创建 run_$i"
done

echo "新的 run 文件夹准备完成"
echo "创建的 run 文件夹列表："
ls -d run_*

# 提交 MD 计算作业
md_job_id=$(sbatch --parsable --array=$start_run-$end_run start_md.sh)
echo "已提交 MD 计算作业，作业 ID: $md_job_id，运行范围：$start_run-$end_run"
