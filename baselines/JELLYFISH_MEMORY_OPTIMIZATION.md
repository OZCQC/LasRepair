# Jellyfish-7B 显存优化说明

## 问题
运行Jellyfish-7B LLM时遇到CUDA out of memory错误。

## 优化措施

### 1. **自动显存清理**
- 每次推理后自动删除中间变量
- 调用`torch.cuda.empty_cache()`释放显存
- 每10个错误修复后进行一次显存检查

### 2. **减少Token数量**
- `max_new_tokens`: 64 → 32
- `max_length`: 添加2048的截断限制
- 减少生成的token数量来节省显存

### 3. **8-bit量化（可选）**
如果显存仍然不足，可以使用8-bit量化：

```bash
# 安装bitsandbytes
pip install bitsandbytes

# 运行时添加--use_8bit参数
python run_jellyfish.py --dataset beers --max_rows 5 --use_8bit
```

**注意**: 8-bit量化会降低模型精度，但可以节省约50%显存

### 4. **错误处理**
- 捕获OOM错误并跳过该cell
- 自动清理显存后继续处理
- 不会因为单个错误而中断整个流程

### 5. **内存监控**
- 启动时显示初始GPU内存
- 加载模型后显示使用的内存
- 每10次修复显示当前内存使用

## 使用方法

### 基本使用（推荐先小规模测试）
```bash
python run_jellyfish.py --dataset beers --max_rows 5 --num_examples 10
```

### 使用8-bit量化（显存不足时）
```bash
python run_jellyfish.py --dataset beers --max_rows 5 --num_examples 10 --use_8bit
```

### 减少示例数量（进一步节省显存）
```bash
python run_jellyfish.py --dataset beers --max_rows 5 --num_examples 5
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_rows` | None | 限制处理的行数（测试用） |
| `--num_examples` | 20 | Few-shot示例数量 |
| `--use_8bit` | False | 是否使用8-bit量化 |

## 显存需求估算

| 配置 | 估算显存 | 适用GPU |
|------|----------|---------|
| FP16 (默认) | ~14-16GB | V100, A100, RTX 3090/4090 |
| 8-bit量化 | ~7-8GB | RTX 2080Ti, RTX 3080 |

## 优化效果

- **推理速度**: 每个cell约2-3秒
- **显存占用**: FP16约14GB, 8-bit约7GB
- **清理频率**: 每10次修复清理一次
- **失败处理**: OOM时自动跳过并继续

## 故障排除

### 仍然OOM？
1. 使用`--use_8bit`参数
2. 减少`--num_examples`到5或更少
3. 确保没有其他程序占用GPU
4. 检查GPU型号和显存大小

### 8-bit量化失败？
```bash
pip install bitsandbytes
pip install accelerate
```

### 速度太慢？
- 这是正常的，LLM推理本身就慢
- 考虑使用`--max_rows`限制处理行数
- 或使用`run_jellyfish_simple.py`（基于字符串匹配，1秒完成）

## 建议

1. **先测试**: 使用`--max_rows 5`测试功能
2. **检查显存**: 观察内存监控输出
3. **逐步增加**: 如果5行OK，再增加到50行
4. **备选方案**: 如果显存不够，使用`run_jellyfish_simple.py`

