#!/bin/bash

# 记录开始时间
start_time=$(date +%s.%N)

# 可见光推理
python concurrent_inference.py /data/public/multimodal/wsr/datasets/OpticalRSCap-13M \
  --url http://localhost:8000/v1/chat/completions \
  --model qwen2.5-vl-72b \
  --backend qwen \
  --image-transport dataurl \
  --json-mode \
  --prompts ./prompts.json \
  --prompt-name opticalrs-E \
  --exts .jpg,.jpeg,.png,.tif,.tiff \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-tokens 2500 \
  --out /data/public/multimodal/wzy/data/optical-result/OpticalRSCap-13M-50.jsonl \
  --max-workers 32  \

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