#!/usr/bin/env python3
"""
使用spacy-transformers训练NER模型
结合了spaCy的易用性和Transformers的强大性能
"""

import json
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import random
from pathlib import Path
import argparse

def create_spacy_transformer_config():
    """创建spacy-transformers配置"""
    config = """
[system]
gpu_allocator = "pytorch"
seed = 0

[nlp]
lang = "zh"
pipeline = ["transformer", "ner"]
batch_size = 128
disabled = []
before_creation = null
after_creation = null
after_pipeline_creation = null
tokenizer = {"@tokenizers": "spacy-transformers.BertWordPieceTokenizer.v1", "vocab_file": "bert-base-chinese", "lowercase": true}

[components]

[components.transformer]
factory = "transformer"
max_batch_items = 4096
set_extra_annotations = {"@annotation_setters": "spacy-transformers.null_annotation_setter.v1"}

[components.transformer.model]
@architectures = "spacy-transformers.TransformerModel.v3"
name = "bert-base-chinese"
mixed_precision = false

[components.transformer.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96

[components.transformer.model.grad_scaler_config]

[components.transformer.model.tokenizer_config]
use_fast = true

[components.transformer.model.transformer_config]

[components.ner]
factory = "ner"
incorrect_spans_key = null
moves = null
scorer = {"@scorers": "spacy.ner_scorer.v1"}
update_with_oracle_cut_size = 100

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = false
nO = null

[components.ner.model.tok2vec]
@architectures = "spacy-transformers.TransformerListener.v1"
grad_factor = 1.0
pooling = {"@layers": "reduce_mean.v1"}
upstream = "*"

[corpora]

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${paths.train}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[training]
accumulate_gradient = 3
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = 0
gpu_allocator = "pytorch"
dropout = 0.1
patience = 1600
max_epochs = 0
max_steps = 20000
eval_frequency = 200
frozen_components = []
annotating_components = []
before_to_disk = null
before_update = null

[training.batcher]
@batchers = "spacy.batch_by_padded.v1"
discard_oversize = true
get_length = null
size = 2000
buffer = 256
size = {"@schedules": "compounding.v1", "start": 100, "stop": 1000, "compound": 1.001, "t": 0.0}

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = false

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001

[training.optimizer.learn_rate]
@schedules = "warmup_linear.v1"
warmup_steps = 250
total_steps = 20000
initial_rate = 0.00005

[training.score_weights]
ents_f = 1.0
ents_p = 0.0
ents_r = 0.0
ents_per_type = null

[pretraining]

[initialize]
vectors = null
init_tok2vec = null
vocab_data = null
lookups = null
before_init = null
after_init = null

[initialize.components]

[initialize.tokenizer]

[paths]
train = "./training_data/train.spacy"
dev = "./training_data/dev.spacy"
vectors = null
init_tok2vec = null
"""
    return config

def prepare_training_data(data_path):
    """准备训练数据"""
    print("准备训练数据...")
    
    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为spaCy格式
    training_data = []
    for item in data:
        text = item['text']
        entities = []
        
        # 从labels构建实体
        current_entity = None
        start_idx = 0
        
        for i, (token, label) in enumerate(zip(item['tokens'], item['labels'])):
            if label.startswith('B-'):
                # 保存前一个实体
                if current_entity:
                    entities.append(current_entity)
                # 开始新实体
                current_entity = (start_idx, start_idx + len(token), label[2:])
            elif label.startswith('I-') and current_entity and label[2:] == current_entity[2]:
                # 扩展当前实体
                current_entity = (current_entity[0], start_idx + len(token), current_entity[2])
            else:
                # 结束当前实体
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            
            start_idx += len(token)
        
        # 添加最后一个实体
        if current_entity:
            entities.append(current_entity)
        
        training_data.append((text, {"entities": entities}))
    
    return training_data

def train_spacy_transformers(data_path="training_data/final_dataset.json", 
                            output_path="./trained_models/spacy_transformers",
                            epochs=10):
    """训练spacy-transformers模型"""
    
    print("="*60)
    print("SpaCy-Transformers 训练")
    print("="*60)
    
    # 1. 准备数据
    training_data = prepare_training_data(data_path)
    print(f"加载了 {len(training_data)} 个训练样本")
    
    # 分割训练集和验证集
    random.shuffle(training_data)
    split = int(0.8 * len(training_data))
    train_data = training_data[:split]
    dev_data = training_data[split:]
    print(f"训练集: {len(train_data)}, 验证集: {len(dev_data)}")
    
    # 2. 创建空白模型
    print("\n初始化spacy-transformers模型...")
    nlp = spacy.blank("zh")
    
    # 3. 添加transformer组件
    try:
        # 添加transformer和ner组件
        from spacy_transformers import TransformerModel
        from spacy_transformers import Transformer
        
        # 添加transformer组件
        config = {
            "model": {
                "@architectures": "spacy-transformers.TransformerModel.v3",
                "name": "bert-base-chinese",
                "tokenizer_config": {"use_fast": True},
                "transformer_config": {},
                "mixed_precision": False,
                "grad_scaler_config": {}
            },
            "set_extra_annotations": {"@annotation_setters": "spacy-transformers.null_annotation_setter.v1"}
        }
        
        nlp.add_pipe("transformer", config=config)
        
        # 添加NER组件
        ner = nlp.add_pipe("ner")
        
        # 添加标签
        for _, annotations in training_data:
            for ent in annotations.get("entities", []):
                ner.add_label(ent[2])
        
        print("✓ 模型组件初始化成功")
        
    except ImportError:
        print("❌ spacy-transformers未安装，请运行: pip install spacy-transformers")
        return None
    
    # 4. 将数据转换为Example对象
    print("\n准备训练数据...")
    train_examples = []
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        train_examples.append(example)
    
    dev_examples = []
    for text, annotations in dev_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        dev_examples.append(example)
    
    # 5. 开始训练
    print(f"\n开始训练 ({epochs} epochs)...")
    
    # 初始化模型
    nlp.initialize(lambda: train_examples)
    
    # 训练循环
    for epoch in range(epochs):
        random.shuffle(train_examples)
        losses = {}
        
        # 批次训练
        batches = spacy.util.minibatch(train_examples, size=8)
        for batch in batches:
            nlp.update(batch, losses=losses)
        
        # 评估
        if epoch % 2 == 0:
            scores = nlp.evaluate(dev_examples)
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Loss: {losses}")
            print(f"  F-score: {scores['ents_f']:.3f}")
            print(f"  Precision: {scores['ents_p']:.3f}")
            print(f"  Recall: {scores['ents_r']:.3f}")
    
    # 6. 保存模型
    print(f"\n保存模型到 {output_path}...")
    nlp.to_disk(output_path)
    print("✓ 训练完成!")
    
    return nlp

def test_model(model_path="./trained_models/spacy_transformers"):
    """测试训练好的模型"""
    print("\n" + "="*60)
    print("测试 SpaCy-Transformers 模型")
    print("="*60)
    
    # 加载模型
    print(f"加载模型: {model_path}")
    nlp = spacy.load(model_path)
    
    # 测试句子
    test_sentences = [
        "今天温度是28摄氏度，CPU有8个核心",
        "使用HPLC分析，溶解度为5.5 mg/mL",
        "FTIR在3000 cm⁻¹处有吸收峰",
        "反应时间为30分钟，温度控制在100°C",
        "The melting point is 252°C as measured by DSC"
    ]
    
    for text in test_sentences:
        print(f"\n原文: {text}")
        doc = nlp(text)
        
        if doc.ents:
            print("识别的实体:")
            for ent in doc.ents:
                print(f"  - {ent.text} ({ent.label_})")
        else:
            print("  未识别到实体")

def main():
    parser = argparse.ArgumentParser(description='SpaCy-Transformers NER训练')
    parser.add_argument('--action', choices=['train', 'test', 'both'], 
                       default='train', help='执行的操作')
    parser.add_argument('--data', default='training_data/final_dataset.json',
                       help='训练数据路径')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--output', default='./trained_models/spacy_transformers',
                       help='模型保存路径')
    
    args = parser.parse_args()
    
    if args.action in ['train', 'both']:
        # 检查依赖
        try:
            import spacy_transformers
            print("✓ spacy-transformers已安装")
        except ImportError:
            print("❌ 需要安装spacy-transformers:")
            print("   pip install spacy-transformers")
            print("   或")
            print("   pip install 'spacy[transformers]'")
            return
        
        # 训练模型
        nlp = train_spacy_transformers(args.data, args.output, args.epochs)
    
    if args.action in ['test', 'both']:
        # 测试模型
        test_model(args.output)

if __name__ == "__main__":
    main()