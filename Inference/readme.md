# Remote-Sensing Image Captioning Tool

A unified batch processing utility for generating structured captions for remote sensing images using OpenAI-compatible backends (GLM / Qwen / vLLM).

## Features
- Recursively scans folders for images and arrays
- Optional Hyperspectral Image (HSI) panelization (band triplets → RGB panels)
- Sends single/multi images in one user message (structured multimodal)
- Robust JSON extraction with retry mechanism
- Checkpoint/resume functionality via JSONL output
- Automatic transport fallback (file:// → base64 data URL on HTTP 400 errors)
- No manual special tokens required; prompts are sanitized automatically

## Requirements
- Python 3.x
- Dependencies: numpy, requests, opencv-python, pillow
- Optional: tifffile, rasterio (for HSI processing)
- An OpenAI-compatible API endpoint (local or remote)

## Basic Usage

### Simple Captioning of a Folder
```bash
python Captions_All.py /path/to/images --out results.jsonl
```

### Custom API Server and Model
```bash
python Captions_All.py /path/to/images --url http://localhost:8000/v1/chat/completions --model qwen2.5-vl-72b --out results.jsonl
```

### Using Custom Prompts
Create a `prompts.json` file with your prompt configurations:
```json
{
  "default": {
    "system": "You are a remote sensing image analysis expert.",
    "user": "Analyze this remote sensing image and provide details about..."
  },
  "detailed": {
    "system": "You are a professional remote sensing analyst.",
    "user": "Conduct a thorough analysis of this image including..."
  }
}
```

Then run:
```bash
python Captions_All.py /path/to/images --prompts prompts.json --prompt-name detailed --out detailed_results.jsonl
```

## Advanced Options

### Hyperspectral Image Processing
Process HSI files by creating multiple RGB panels from different band combinations:
```bash
python Captions_All.py /path/to/hsi_files --hsi-panels --hsi-triplets "4,3,2;8,4,3;12,8,4" --out hsi_results.jsonl
```

### Control Output and Processing
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

### Resuming Processing
If processing was interrupted, the script will automatically resume from where it left off when run with the same output file:
```bash
python Captions_All.py /path/to/images --out results.jsonl  # Will resume processing
```

### Overwriting Existing Results
```bash
python Captions_All.py /path/to/images --out results.jsonl --overwrite
```

## Command-line Arguments

### Required
- `folder`: Path to the image folder (recursive search)

### Optional
- `--url`: OpenAI-compatible API endpoint (default: http://127.0.0.1:8000/v1/chat/completions)
- `--model`: Model name visible to the server (default: qwen2.5-vl-72b)
- `--out`: Output JSONL file path (default: results.jsonl)
- `--max-side`: Maximum image side length after resizing (default: 2048)
- `--exts`: Comma-separated file extensions to process
- `--temperature`: Sampling temperature (default: 0.2)
- `--top-p`: Top-p sampling parameter (default: 0.9)
- `--max-tokens`: Maximum tokens per response (default: 1500)
- `--limit`: Process only the first N files (0 = all files)
- `--prompts`: Path to prompts JSON file (default: prompts.json)
- `--prompt-name`: Name of the prompt configuration to use (default: default)
- `--max-retries`: Maximum retry attempts for failed requests (default: 5)

### Image Transport Options
- `--image-transport`: Image transport method (auto, file, dataurl) (default: auto)
- `--image-part-type`: Image part type (use "image_url" for vLLM 0.10.x compatibility)
- `--json-mode`: Enable strict JSON output mode

### HSI Processing Options
- `--hsi-panels`: Enable HSI panelization
- `--hsi-triplets`: Semicolon-separated band triplets (e.g., "3,2,1;8,4,3")
- `--band-indexing`: Band indexing mode (one-based or zero-based)
- `--clip-percent`: Percentile clipping for contrast enhancement
- `--max-panels`: Maximum number of panels to generate
- `--keep-panels-dir`: Directory to persist panels/intermediates

## Output Format
The tool generates a JSONL file with one record per processed image, containing:
- Image metadata (path, original size, resized size)
- Processing information (time elapsed, retry count)
- Extracted JSON output with structured caption fields
- Optional raw text output (with --detail flag)
- Error information (if any)

## Notes
- The script automatically handles different image formats (JPG, PNG, TIFF, NPY, NPZ)
- For large images, automatic resizing is performed to fit within max-side constraints
- Checkpoint/resume functionality uses the output JSONL file to track processed images
- When using local file paths, ensure the API server is configured with appropriate local media path permissions
        
