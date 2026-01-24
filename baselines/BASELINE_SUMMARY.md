# Baseline Implementation Summary

## 完成的工作

为论文实验部分实现了7个baseline方法的统一pipeline：

### 1. Jellyfish ✅
- **方法**: 基于Jaro-Winkler字符串相似度
- **复杂度**: 简单
- **依赖**: `pip install jellyfish`
- **运行**: `python run_jellyfish.py --dataset <name>`
- **状态**: 完全实现，可直接运行

### 2. Raha ✅
- **方法**: 错误检测与修复框架
- **复杂度**: 中等
- **依赖**: 见raha/requirements.txt
- **运行**: `python run_raha.py --dataset <name>`
- **状态**: 完全实现，需要安装raha依赖

### 3. ActiveDetect ✅
- **方法**: 主动学习错误检测
- **复杂度**: 中等
- **依赖**: gensim, usaddress
- **运行**: `python run_activedetect.py --dataset <name>`
- **状态**: 实现了检测+简单修复策略

### 4. BigDansing ✅
- **方法**: 基于图和约束的修复
- **复杂度**: 中等
- **依赖**: pandas, numpy, tqdm
- **运行**: `python run_bigdansing.py --dataset <name>`
- **状态**: 包装了现有实现，需要约束文件

### 5. HoloClean ✅
- **方法**: 概率推理+约束
- **复杂度**: 复杂
- **依赖**: PostgreSQL + holoclean包
- **运行**: `python run_holoclean.py --dataset <name>`
- **状态**: 完全实现，需要PostgreSQL数据库

### 6. GIDCL ⚠️
- **方法**: 图增强+LLM
- **复杂度**: 非常复杂
- **依赖**: LLM模型 + GPU
- **运行**: `python run_gidcl.py --dataset <name>`
- **状态**: 提供了wrapper，需要预训练模型或使用预计算结果

### 7. MLNClean ⚠️
- **方法**: 马尔可夫逻辑网络
- **复杂度**: 复杂
- **依赖**: Java 8+ + Maven
- **运行**: `python run_mlnclean.py --dataset <name>`
- **状态**: 提供了wrapper，需要Maven编译

## 统一接口

所有baseline都遵循相同的接口：

```bash
python run_<baseline>.py \
    --dataset <dataset_name> \
    --data_dir /path/to/datasets \
    --output_dir /path/to/results
```

## 输出格式

每个baseline生成：
1. `results/<baseline>/<dataset>/repaired.csv` - 修复后的数据
2. `results/<baseline>/<dataset>/runtime.json` - 性能指标

runtime.json格式：
```json
{
  "precision": 0.85,
  "recall": 0.78,
  "f1": 0.81,
  "runtime": 123.45
}
```

## 批量运行

使用主脚本运行所有baselines：

```bash
# 运行所有baseline在所有数据集上
python run_all_baselines.py

# 选择特定的baseline和数据集
python run_all_baselines.py \
    --baselines jellyfish raha holoclean \
    --datasets beers flight hospital
```

## 建议运行顺序

按复杂度和依赖，建议按以下顺序运行：

1. **Jellyfish** - 最简单，无特殊依赖
2. **Raha** - 简单，只需Python包
3. **ActiveDetect** - 简单，只需Python包
4. **BigDansing** - 中等，需要约束文件
5. **HoloClean** - 复杂，需要PostgreSQL
6. **GIDCL** - 非常复杂，需要LLM
7. **MLNClean** - 复杂，需要Java环境

## 快速开始

### 最小化测试
```bash
# 1. 安装基础依赖
pip install pandas numpy jellyfish

# 2. 测试最简单的baseline
python run_jellyfish.py --dataset beers

# 3. 查看结果
cat results/jellyfish/beers/runtime.json
head results/jellyfish/beers/repaired.csv
```

### 完整设置
```bash
# 1. 运行设置脚本
bash setup_baselines.sh

# 2. 手动设置PostgreSQL (for HoloClean)
# 参考 README_BASELINES.md

# 3. 运行所有简单baselines
python run_all_baselines.py \
    --baselines jellyfish raha activedetect
```

## 数据集要求

每个数据集目录应包含：
- `dirty.csv` - 必需
- `clean.csv` - 必需  
- `<dataset>_constraints.txt` - 可选，BigDansing和HoloClean需要

当前支持的数据集：
- beers
- flight
- hospital
- movies
- rayyan
- shuttle
- tax_200k
- tax_20k
- walmart

## 性能分析

runtime.json包含了所有需要的指标：
- **runtime**: 运行时间(秒) - 用于效率分析
- **precision**: 修复精度
- **recall**: 修复召回率
- **f1**: F1分数

可以写脚本收集所有结果进行分析：
```python
import json
import glob

results = {}
for json_file in glob.glob("results/**/runtime.json", recursive=True):
    with open(json_file) as f:
        data = json.load(f)
    # 提取baseline和dataset名称
    parts = json_file.split('/')
    baseline = parts[1]
    dataset = parts[2]
    results[(baseline, dataset)] = data
```

## 注意事项

1. **简洁性优先**: 所有实现都尽可能简单
2. **可运行性优先**: 优先保证能正确运行
3. **错误处理**: 包含基本错误处理和提示
4. **时间记录**: 所有运行时间都被准确记录

## 已知限制

1. **GIDCL**: 需要大量GPU资源和预训练模型，当前使用预计算结果
2. **MLNClean**: Java-based，需要手动编译
3. **HoloClean**: 需要PostgreSQL数据库配置
4. **BigDansing**: 需要约束文件，部分数据集可能缺失

## 文件清单

- `run_jellyfish.py` - Jellyfish baseline
- `run_raha.py` - Raha baseline
- `run_activedetect.py` - ActiveDetect baseline
- `run_bigdansing.py` - BigDansing baseline
- `run_holoclean.py` - HoloClean baseline
- `run_gidcl.py` - GIDCL baseline
- `run_mlnclean.py` - MLNClean baseline
- `run_all_baselines.py` - 主运行脚本
- `setup_baselines.sh` - 依赖安装脚本
- `example_run.sh` - 使用示例
- `README_BASELINES.md` - 详细文档
- `BASELINE_SUMMARY.md` - 本文档

## 下一步

1. 运行简单baseline (jellyfish, raha)验证pipeline
2. 配置PostgreSQL后运行HoloClean
3. 收集所有结果进行分析
4. 如需GIDCL和MLNClean，进行额外配置

## 支持

遇到问题时：
1. 检查README_BASELINES.md的详细说明
2. 查看各baseline的原始文档
3. 确认依赖已正确安装
4. 检查数据集格式

