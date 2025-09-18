# 遥感图像 captioning 工具

一个用于使用 OpenAI 兼容后端（GLM / Qwen / vLLM）为遥感图像生成结构化描述的批处理工具。

## 功能特点
- 递归扫描文件夹中的图像和数组文件
- 可选的高光谱图像（HSI）面板化处理（波段三元组 → RGB 面板）
- 在一个用户消息中发送单个/多个图像（结构化多模态）
- 强大的 JSON 提取和重试机制
- 通过 JSONL 输出生成检查点/恢复功能
- 自动传输回退（在 HTTP 400 错误时从 file:// 自动切换到 base64 数据 URL）
- 无需手动特殊令牌；提示文本自动清理

## 环境要求
- Python 3.x
- 依赖库：numpy、requests、opencv-python、pillow
- 可选：tifffile、rasterio（用于 HSI 处理）
- 一个兼容 OpenAI 的 API 端点（本地或远程）

## 基本使用

### 简单的文件夹图像 captioning
```bash
python Captions_All.py /path/to/images --out results.jsonl
```

### 自定义 API 服务器和模型
```bash
python Captions_All.py /path/to/images --url http://localhost:8000/v1/chat/completions --model qwen2.5-vl-72b --out results.jsonl
```

### 使用自定义提示词
创建一个 `prompts.json` 文件配置您的提示词：
```json
{
  "default": {
    "system": "您是一名遥感图像分析专家。",
    "user": "分析这张遥感图像并提供有关...的详细信息"
  },
  "detailed": {
    "system": "您是一名专业的遥感分析师。",
    "user": "对这张图像进行全面分析，包括..."
  }
}
```

然后运行：
```bash
python Captions_All.py /path/to/images --prompts prompts.json --prompt-name detailed --out detailed_results.jsonl
```

## 并行处理（可见光+多光谱）

加入了` --max-workers`参数，设置并行计算的线程数量。

`qwen-parallel-optical.sh` 

```shell
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
```

`qwen-parallel.sh` 

```shell
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
```



## 高级选项

### 高光谱图像处理
通过从不同波段组合创建多个 RGB 面板来处理 HSI 文件：
```bash
python Captions_All.py /path/to/hsi_files --hsi-panels --hsi-triplets "4,3,2;8,4,3;12,8,4" --out hsi_results.jsonl
```

### 控制输出和处理
```bash
python Captions_All.py /path/to/images \
  --max-side 1536 \
  --temperature 0.3 \
  --top-p 0.95 \
  --max-tokens 2000 \
  --json-mode \
  --verbose \
  --out detailed_results.jsonl
```

### 恢复处理
如果处理被中断，当使用相同的输出文件运行脚本时，它将自动从中断点恢复：
```bash
python Captions_All.py /path/to/images --out results.jsonl  # 将恢复处理
```

### 覆盖现有结果
```bash
python Captions_All.py /path/to/images --out results.jsonl --overwrite
```

## 命令行参数

### 必需参数
- `folder`: 图像文件夹路径（递归搜索）

### 可选参数
- `--url`: OpenAI 兼容的 API 端点（默认：http://127.0.0.1:8000/v1/chat/completions）
- `--model`: 服务器可见的模型名称（默认：qwen2.5-vl-72b）
- `--out`: 输出 JSONL 文件路径（默认：results.jsonl）
- `--max-side`: 调整大小后图像的最大边长（默认：2048）
- `--exts`: 要处理的文件扩展名（逗号分隔）
- `--temperature`: 采样温度（默认：0.2）
- `--top-p`: Top-p 采样参数（默认：0.9）
- `--max-tokens`: 每个响应的最大令牌数（默认：1500）
- `--limit`: 仅处理前 N 个文件（0 = 所有文件）
- `--prompts`: 提示词 JSON 文件路径（默认：prompts.json）
- `--prompt-name`: 要使用的提示词配置名称（默认：default）
- `--max-retries`: 失败请求的最大重试次数（默认：5）

### 图像传输选项
- `--image-transport`: 图像传输方法（auto、file、dataurl）（默认：auto）
- `--image-part-type`: 图像部分类型（使用 "image_url" 与 vLLM 0.10.x 兼容）
- `--json-mode`: 启用严格的 JSON 输出模式

### HSI 处理选项
- `--hsi-panels`: 启用 HSI 面板化
- `--hsi-triplets`: 分号分隔的波段三元组（例如 "3,2,1;8,4,3"）
- `--band-indexing`: 波段索引模式（one-based 或 zero-based）
- `--clip-percent`: 对比度增强的百分位裁剪
- `--max-panels`: 要生成的最大面板数
- `--keep-panels-dir`: 持久化面板/中间文件的目录

## 输出格式
该工具生成一个 JSONL 文件，每个处理的图像对应一条记录，包含：
- 图像元数据（路径、原始大小、调整大小后的大小）
- 处理信息（耗时、重试次数）
- 带有结构化描述字段的提取 JSON 输出
- 可选的原始文本输出（使用 --detail 标志）
- 错误信息（如有）

## 注意事项
- 脚本自动处理不同的图像格式（JPG、PNG、TIFF、NPY、NPZ）
- 对于大图像，会自动调整大小以适应 max-side 限制
- 检查点/恢复功能使用输出 JSONL 文件跟踪已处理的图像
- 使用本地文件路径时，请确保 API 服务器配置了适当的本地媒体路径权限
        
