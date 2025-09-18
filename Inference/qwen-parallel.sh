#!/bin/bash

# 记录开始时间
start_time=$(date +%s.%N)

python concurrent_inference.py /data/public/multimodal/wsr/datasets/fmow-sentinel/fmow-sentinel/fmow-sentinel/train/airport/ \
    --url http://127.0.0.1:8000/v1/chat/completions  \
    --model qwen2.5-vl-72b  \
    --backend qwen  \
    --json-mode  \
    --image-transport dataurl  \
    --prompts ./prompts.json  \
    --prompt-name hyperspectral-v7  \
    --hsi-panels \
    --hsi-triplets "1,2,3;4,5,6;7,8,9;10,12,13" \
    --band-indexing one-based \
    --clip-percent 2.0 \
    --max-side 1536 \
    --max-tokens 2500 \
    --top-p .8 \
    --out results_hsi_airport2.jsonl \
    --max-panels 6 \
    --temperature 0.3 \
    --detail \
    --verbose \
    --limit 50 \
    --max-workers 4  \

# 记录结束时间
end_time=$(date +%s.%N)

# 计算执行时间
execution_time=$(echo "$end_time - $start_time" | bc)

# 格式化时间显示
hours=$(echo "$execution_time/3600" | bc)
minutes=$(echo "($execution_time%3600)/60" | bc)
seconds=$(echo "$execution_time%60" | bc)

# 输出美观的执行时间
echo "==================== 执行时间统计 ===================="
echo "开始时间: $(date -d @$start_time)"
echo "结束时间: $(date -d @$end_time)"
echo "总执行时间: $hours 小时 $minutes 分钟 $(printf "%.2f" $seconds) 秒"
echo "===================================================="