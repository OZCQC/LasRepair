# Baseline Pipeline Scripts

这个目录包含了所有baseline方法的统一pipeline脚本，用于论文实验。

## 概述

每个baseline都有一个独立的Python脚本 (`run_<baseline>.py`)，提供统一的接口：
- 读取dirty和clean CSV文件
- 运行data repair
- 保存修复结果到 `baselines/results/<baseline>/<dataset>/repaired.csv`
- 记录运行时间和性能指标到 `baselines/results/<baseline>/<dataset>/runtime.json`

## Baselines列表

1. **Jellyfish** - 基于Jaro-Winkler相似度的字符串匹配修复
2. **Raha** - 错误检测与修复框架
3. **ActiveDetect** - 主动错误检测 + 简单修复策略
4. **BigDansing** - 基于图和约束的修复方法
5. **HoloClean** - 基于概率推理和约束的修复
6. **GIDCL** - 基于图增强和LLM的数据清洗 (需要预训练模型)
7. **MLNClean** - 基于马尔可夫逻辑网络的清洗 (Java项目)

## 使用方法

### 单个baseline运行

```bash
# 运行Jellyfish
python run_jellyfish.py --dataset beers

# 运行Raha
python run_raha.py --dataset flight

# 自定义数据路径
python run_holoclean.py --dataset hospital \
    --data_dir /path/to/datasets \
    --output_dir /path/to/results
```

### 运行所有baselines

```bash
# 在所有数据集上运行所有baselines
python run_all_baselines.py

# 只运行特定的baselines
python run_all_baselines.py --baselines jellyfish raha holoclean

# 只在特定数据集上运行
python run_all_baselines.py --datasets beers flight hospital

# 组合使用
python run_all_baselines.py \
    --baselines jellyfish raha \
    --datasets beers flight
```

## 输出结构

```
baselines/results/
├── jellyfish/
│   ├── beers/
│   │   ├── repaired.csv        # 修复后的数据
│   │   └── runtime.json        # 运行时间和指标
│   ├── flight/
│   │   ├── repaired.csv
│   │   └── runtime.json
│   └── ...
├── raha/
│   └── ...
└── ...
```

## 环境配置

### 基础依赖

大多数baselines需要以下基础包：
```bash
pip install pandas numpy scikit-learn
```

### 特定baseline的依赖

#### Jellyfish
```bash
pip install jellyfish
```

#### Raha
```bash
cd baselines/raha
pip install -r requirements.txt
```

#### ActiveDetect
```bash
cd baselines/activedetect
pip install -e .
```

#### HoloClean
需要PostgreSQL数据库：
```bash
# 安装PostgreSQL
sudo apt-get install postgresql

# 创建数据库和用户
sudo -u postgres psql
CREATE DATABASE holo;
CREATE USER holocleanuser WITH PASSWORD 'abcd1234';
GRANT ALL PRIVILEGES ON DATABASE holo TO holocleanuser;

# 安装Python依赖
cd baselines/holoclean
pip install -r requirements.txt
```

#### BigDansing
```bash
cd baselines/bigdansing_holistic
pip install pandas numpy tqdm
```

#### GIDCL
需要LLM和大量计算资源：
```bash
# 需要预训练的模型和GPU
# 参考 baselines/GIDCL/README.md
```

#### MLNClean
需要Java和Maven：
```bash
cd baselines/MLNClean
mvn clean package
```

## 版本隔离 (可选)

如果遇到版本冲突，可以为每个baseline创建独立的conda环境：

```bash
# Jellyfish环境
conda create -n jellyfish python=3.8
conda activate jellyfish
pip install jellyfish pandas numpy

# Raha环境
conda create -n raha python=3.7
conda activate raha
cd baselines/raha && pip install -r requirements.txt

# HoloClean环境
conda create -n holoclean python=3.6
conda activate holoclean
cd baselines/holoclean && pip install -r requirements.txt
```

然后修改运行脚本使用对应的环境：
```bash
conda run -n jellyfish python run_jellyfish.py --dataset beers
conda run -n raha python run_raha.py --dataset flight
```

## 数据集格式

每个数据集应该包含：
- `dirty.csv` - 带错误的数据
- `clean.csv` - 干净的ground truth数据
- `<dataset>_constraints.txt` (可选) - 约束规则 (用于BigDansing和HoloClean)

## 性能指标

每个baseline会计算：
- **Precision**: 正确修复的错误 / 所有修复
- **Recall**: 正确修复的错误 / 所有错误
- **F1-Score**: Precision和Recall的调和平均
- **Runtime**: 运行时间(秒)

结果保存在 `runtime.json`:
```json
{
  "precision": 0.85,
  "recall": 0.78,
  "f1": 0.81,
  "runtime": 123.45
}
```

## 注意事项

1. **简洁性**: 所有脚本都设计得尽可能简单，只实现核心功能
2. **可运行性**: 优先保证代码能够正确运行
3. **效率分析**: 所有运行时间都被记录用于效率对比
4. **错误处理**: 脚本包含基本的错误处理，出错时会给出提示

## 复杂baseline说明

### GIDCL
- 需要预训练的LLM模型
- 需要大量GPU资源
- 如果没有预训练模型，脚本会使用预计算的结果或返回错误

### MLNClean
- 基于Java的项目
- 需要先用Maven编译: `mvn clean package`
- 需要特定的数据集结构

### BigDansing
- 需要约束规则文件
- 会在其自己的目录下生成中间结果

## 故障排除

### PostgreSQL连接问题 (HoloClean)
```bash
# 检查PostgreSQL是否运行
sudo systemctl status postgresql

# 测试连接
psql -U holocleanuser -W holo
```

### Python路径问题
如果遇到import错误，确保在baselines目录下运行脚本，或设置PYTHONPATH：
```bash
export PYTHONPATH=/data1/qianc/EMCL/baselines:$PYTHONPATH
```

### 内存不足
某些baseline (如HoloClean, GIDCL)可能需要大量内存。考虑：
- 使用较小的数据集
- 增加系统内存或swap
- 调整baseline的参数

## 联系与支持

如果遇到问题，请检查：
1. 各baseline原始仓库的README
2. requirements.txt中的依赖版本
3. 数据集格式是否正确

## 许可证

每个baseline方法保留其原始许可证。请参考各baseline目录下的LICENSE文件。

