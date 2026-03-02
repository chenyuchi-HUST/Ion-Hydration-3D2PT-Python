#!/bin/bash

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

# 设置默认值
start_run=${1:-1}  # 如果没有提供参数，默认为 1
max_run=$(get_max_run_number)
end_run=${2:-$max_run}  # 如果没有提供第二个参数，使用检测到的最大值

if [ $max_run -eq 0 ]; then
    echo "错误：没有找到 run 目录。"
    exit 1
fi

echo "处理范围：从 run_$start_run 到 run_$end_run"


# 步骤 : 使用 for 循环提交网格分析作业
echo "开始提交网格分析作业..."
for ((run=$start_run; run<=$end_run; run++)); do
    echo "提交 run_$run 的作业..."
    job_id=$(sbatch --parsable --export=ALL,RUN_NUMBER=$run start_grid_analysis.sh)
    if [ $? -ne 0 ]; then
        echo "提交 run_$run 的作业失败。"
    else
        echo "run_$run 的作业已提交，作业 ID: $job_id"
    fi
done

echo "所有网格分析作业已提交完成。"