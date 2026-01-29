# UniWorld 子进程集成方案

## 🎯 为什么需要这个方案？

UniWorld 和 lmms-eval 的依赖环境差异太大（特别是 `transformers` 和 `diffusers` 版本），直接集成会导致：
- ❌ 输出乱码
- ❌ 模型加载失败
- ❌ 依赖冲突

**解决方案**：将 UniWorld 运行在独立的 conda 环境中，通过 subprocess 调用。

---

## 📦 方案 1：子进程调用（推荐）

### ✅ 优点
- 完全环境隔离，互不影响
- 复用 UniWorld 原始环境
- lmms-eval 只需要处理输入输出
- 不需要修改 UniWorld 代码

### 📝 设置步骤

#### 1. 创建 UniWorld 独立环境

```bash
# 创建新环境
conda create -n uniworld python=3.10 -y
conda activate uniworld

# 安装 UniWorld 依赖
cd UniWorld/UniWorld-V1  # 或你的 UniWorld 路径
pip install -r requirements.txt

# 测试环境
python -c "from modeling.uniworld import UnivaQwen2p5VLForConditionalGeneration; print('OK')"
```

#### 2. 复制独立推理脚本

```bash
# 在 lmms-eval 根目录
cp uniworld_inference_standalone.py UniWorld/UniWorld-V1/
```

#### 3. 使用 lmms-eval（在 lmms-eval 环境）

```bash
conda activate lmms-eval  # 或你的 lmms-eval 环境

python -m lmms_eval \
    --model uniworld_subprocess \
    --model_args pretrained=./UniWorld/UniWorld-V1,conda_env=uniworld,script_path=./UniWorld/UniWorld-V1/uniworld_inference_standalone.py \
    --tasks chartqa100 \
    --batch_size 1 \
    --output_path ./logs/uniworld_subprocess
```

### 📊 工作流程

```
lmms-eval (conda env: lmms-eval)
    ↓
    准备输入：prompt + images
    ↓
    subprocess 调用 ↓
    ↓
UniWorld (conda env: uniworld)
    加载模型 → 推理 → 输出结果
    ↑
    ↓
lmms-eval 接收输出：text + images
    ↓
    保存结果
```

---

## 🔧 方案 2：Docker 容器化

如果需要更严格的隔离或在多台机器上部署：

### Dockerfile for UniWorld

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 安装基础依赖
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

# 安装 UniWorld
WORKDIR /app
COPY UniWorld /app/UniWorld
WORKDIR /app/UniWorld/UniWorld-V1
RUN pip install -r requirements.txt

# 复制推理脚本
COPY uniworld_inference_standalone.py /app/

EXPOSE 8000

# 启动 API 服务（可选）
CMD ["python", "/app/uniworld_api_server.py"]
```

### 使用 Docker

```bash
# 构建镜像
docker build -t uniworld:latest .

# 运行容器
docker run --gpus all -p 8000:8000 \
    -v /path/to/models:/models \
    -v /path/to/outputs:/outputs \
    uniworld:latest
```

---

## 🌐 方案 3：API 服务模式

将 UniWorld 作为 HTTP API 服务：

### 创建 UniWorld API 服务器

```python
# uniworld_api_server.py
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import uvicorn

app = FastAPI()

# 全局加载模型
model = None
processor = None

@app.on_event("startup")
async def load_model():
    global model, processor
    # 加载 UniWorld 模型
    from modeling.uniworld import UnivaQwen2p5VLForConditionalGeneration
    from transformers import AutoProcessor
    
    model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        "/models/UniWorld-V1",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("/models/UniWorld-V1")

class InferenceRequest(BaseModel):
    prompt: str
    mode: str = "text"

@app.post("/inference")
async def inference(
    request: InferenceRequest,
    images: list[UploadFile] = File(None)
):
    # 处理图片
    imgs = []
    if images:
        for img_file in images:
            img = Image.open(img_file.file)
            imgs.append(img)
    
    # 推理
    inputs = processor(text=[request.prompt], images=imgs, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=2048)
    text = processor.decode(outputs[0], skip_special_tokens=True)
    
    return {"text": text, "images": []}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 在 lmms-eval 中调用 API

```python
# lmms_eval/models/simple/uniworld_api.py
import requests

class UniWorldAPI(lmms):
    def __init__(self, api_url="http://localhost:8000", **kwargs):
        self.api_url = api_url
    
    def generate_until(self, requests):
        res = []
        for context, gen_kwargs, doc_to_visual, doc_id, task, split in requests:
            # 准备数据
            files = []
            if doc_to_visual:
                doc = self.task_dict[task][split][doc_id]
                images = doc_to_visual(doc)
                for img in images:
                    # 转换为字节
                    import io
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    buf.seek(0)
                    files.append(('images', buf))
            
            # 调用 API
            response = requests.post(
                f"{self.api_url}/inference",
                json={"prompt": context, "mode": "text"},
                files=files
            )
            
            result = response.json()
            res.append(json.dumps(result))
        
        return res
```

---

## 🎯 方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **子进程** | 简单、无需额外服务 | 每次调用需要重新加载模型 | 单机评测、小规模任务 |
| **Docker** | 完全隔离、可移植 | 需要 Docker 环境 | 多机部署、CI/CD |
| **API 服务** | 模型常驻内存、快速响应 | 需要维护服务 | 大规模评测、多任务并行 |

---

## 🚀 推荐使用流程

### 快速测试（子进程）

```bash
# 1. 设置 UniWorld 环境
conda create -n uniworld python=3.10 -y
conda activate uniworld
cd UniWorld/UniWorld-V1 && pip install -r requirements.txt

# 2. 切换到 lmms-eval 环境
conda activate lmms-eval

# 3. 运行评测
python -m lmms_eval \
    --model uniworld_subprocess \
    --model_args pretrained=./UniWorld/UniWorld-V1,conda_env=uniworld \
    --tasks chartqa100 \
    --batch_size 1 \
    --output_path ./logs/test
```

### 生产环境（API 服务）

```bash
# Terminal 1: 启动 UniWorld API 服务
conda activate uniworld
cd UniWorld/UniWorld-V1
python uniworld_api_server.py

# Terminal 2: 运行 lmms-eval
conda activate lmms-eval
python -m lmms_eval \
    --model uniworld_api \
    --model_args api_url=http://localhost:8000 \
    --tasks chartqa100,jigsaw100,maze100 \
    --batch_size 1 \
    --output_path ./logs/production
```

---

## ❓ 常见问题

### Q1: conda 环境找不到？
```bash
# 检查环境列表
conda env list

# 确保环境名称正确
conda activate uniworld
which python  # 应该指向 uniworld 环境
```

### Q2: 子进程调用超时？
```python
# 修改 uniworld_subprocess.py 中的 timeout
result = subprocess.run(
    cmd,
    timeout=600,  # 增加到 10 分钟
)
```

### Q3: GPU 内存不足？
```bash
# UniWorld 环境中启用 CPU offload
# 修改 uniworld_inference_standalone.py:
pipe.enable_sequential_cpu_offload()
pipe.enable_vae_slicing()
```

---

## 📌 总结

对于你的情况（环境差异导致乱码），**强烈推荐方案 1（子进程）**：
- ✅ 完全隔离环境
- ✅ 不需要修改现有代码
- ✅ 简单易用
- ✅ 适合单机评测

如果后续需要大规模评测或多机部署，再考虑升级到 API 服务模式。
