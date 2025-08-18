# 中文命名实体识别训练项目

## 安装依赖

```bash
pip install -r requirements.txt
```

## 文件说明

- `train_offline_simple.py` - 主训练脚本，支持下载、训练、测试
- `download_model.py` - 单独的模型下载脚本（可选）
- `data_preprocessing.py` - 数据预处理工具
- `synthetic_training_data.json` - 自动生成的训练数据（100个样本）

## 输出目录

- `./trained_models/transformers_offline/` - Transformers训练的模型
- `./trained_models/spacy_offline/` - spaCy训练的模型
- `./logs/` - 训练日志

## 命令行选项

```bash
python train_offline_simple.py --help

# 主要选项:
--action {download,train,test}  # 执行动作：下载、训练或测试
--framework {transformers,spacy}  # 选择框架
--epochs EPOCHS                 # 训练轮数
--data DATA                     # 训练数据文件路径
```

## 使用方法

### 离线下载模型

```bash
python train_offline_simple.py --action download
```

### 训练Transformers模型

```bash
# 训练3个epochs（快速）
python train_offline_simple.py --framework transformers --epochs 3

# 训练更多epochs（更好效果）
python train_offline_simple.py --framework transformers --epochs 10
```

### 训练spaCy模型

```bash
# 训练spaCy模型（优先使用中文预训练模型 + pkuseg）
python train_offline_simple.py --framework spacy --epochs 10

# 说明：会自动选择最佳模型
# 1. 优先：zh_core_web_sm + pkuseg（需要pkuseg安装成功）
# 2. 备选：en_core_web_sm + jieba分词器（如果pkuseg不可用）
```

### 测试模型

```bash
# 测试Transformers模型
python train_offline_simple.py --action test --framework transformers

# 测试spaCy模型  
python train_offline_simple.py --action test --framework spacy
```

## 使用训练后的模型推理

```bash
python inference.py --framework transformers
python inference.py --framework spacy
```