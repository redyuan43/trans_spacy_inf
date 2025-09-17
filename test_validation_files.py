#!/usr/bin/env python3
"""
使用训练好的模型测试data目录下的zh.txt和en.txt文件
"""

import json
import sys
import argparse
from sentence_eos_processor import SentenceEOSProcessor

def test_transformers_model(file_path, enable_eos=True):
    """使用Transformers模型测试"""
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    print(f"=== 使用Transformers模型测试 {file_path} ===")

    # 加载模型
    model_path = "./trained_models/transformers_offline"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    # 标签映射
    id2label = {0: 'O', 1: 'B-TECH', 2: 'I-TECH', 3: 'B-NUM', 4: 'I-NUM', 5: 'B-UNIT', 6: 'I-UNIT'}

    # EOS处理器
    eos_processor = None
    if enable_eos:
        eos_processor = SentenceEOSProcessor()
        print("已启用句子结束符号(EOS)识别")
    
    # 读取测试文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    results = []
    
    for i, line in enumerate(lines[:10], 1):  # 只测试前10行
        text = line.strip()
        if not text:
            continue
            
        print(f"\n--- 第{i}行 ---")
        print(f"原文: {text}")
        
        # 分词和预测
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # 转换为tokens和labels
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [id2label[p.item()] for p in predictions[0]]
        
        # 提取实体
        entities = []
        current_entity = []
        current_label = None
        current_text = []
        
        for token, label in zip(tokens, labels):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
                
            # 处理WordPiece tokens
            if token.startswith('##'):
                token_text = token[2:]
            else:
                token_text = token
                
            if label.startswith('B-'):
                # 保存前一个实体
                if current_entity:
                    entities.append((''.join(current_text), current_label))
                # 开始新实体
                current_entity = [token]
                current_text = [token_text]
                current_label = label[2:]
            elif label.startswith('I-') and current_label == label[2:]:
                current_entity.append(token)
                current_text.append(token_text)
            else:
                # 结束当前实体
                if current_entity:
                    entities.append((''.join(current_text), current_label))
                    current_entity = []
                    current_text = []
                    current_label = None
        
        # 处理最后一个实体
        if current_entity:
            entities.append((''.join(current_text), current_label))
        
        print(f"识别的实体: {entities}")
        
        # 转换实体格式为(start, end, label)
        entities_tuple_format = []
        char_pos = 0
        for token in tokens:
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            token_text = token[2:] if token.startswith('##') else token

            # 查找实体
            for entity_text, entity_label in entities:
                if text[char_pos:char_pos+len(entity_text)] == entity_text:
                    entities_tuple_format.append((char_pos, char_pos+len(entity_text), entity_label))

            char_pos += len(token_text)

        result = {
            'text': text,
            'tokens': [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]']],
            'labels': [l for t, l in zip(tokens, labels) if t not in ['[CLS]', '[SEP]', '[PAD]']],
            'entities': entities_tuple_format
        }

        # 添加EOS后处理
        if eos_processor:
            result = eos_processor.process_prediction_result(text, result)

        results.append(result)
    
    return results

def test_spacy_model(file_path, enable_eos=True):
    """使用spaCy模型测试"""
    import spacy

    print(f"\n=== 使用spaCy模型测试 {file_path} ===")

    try:
        # 加载模型
        nlp = spacy.load("./trained_models/spacy_offline")
    except Exception as e:
        print(f"加载spaCy模型失败: {e}")
        return []

    # EOS处理器
    eos_processor = None
    if enable_eos:
        eos_processor = SentenceEOSProcessor()
        print("已启用句子结束符号(EOS)识别")
    
    # 读取测试文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    results = []
    
    for i, line in enumerate(lines[:10], 1):  # 只测试前10行
        text = line.strip()
        if not text:
            continue
            
        print(f"\n--- 第{i}行 ---")
        print(f"原文: {text}")
        
        # 预测
        doc = nlp(text)
        
        # 提取结果
        tokens = [token.text for token in doc]
        labels = [f"{token.ent_iob_}-{token.ent_type_}" if token.ent_iob_ != 'O' else 'O' for token in doc]
        entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

        print(f"分词结果: {tokens}")
        print(f"识别的实体: {[(text[start:end], label) for start, end, label in entities]}")

        result = {
            'text': text,
            'tokens': tokens,
            'labels': labels,
            'entities': entities
        }

        # 添加EOS后处理
        if eos_processor:
            result = eos_processor.process_prediction_result(text, result)

        results.append(result)
    
    return results

def test_spacy_transformers_model(file_path, enable_eos=True):
    """使用SpaCy-Transformers模型测试"""
    import spacy

    print(f"\n=== 使用SpaCy-Transformers模型测试 {file_path} ===")

    try:
        # 加载模型
        nlp = spacy.load("./trained_models/spacy_transformers")
    except Exception as e:
        print(f"加载SpaCy-Transformers模型失败: {e}")
        return []

    # EOS处理器
    eos_processor = None
    if enable_eos:
        eos_processor = SentenceEOSProcessor()
        print("已启用句子结束符号(EOS)识别")

    # 读取测试文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = []

    for i, line in enumerate(lines[:10], 1):  # 只测试前10行
        text = line.strip()
        if not text:
            continue

        print(f"\n--- 第{i}行 ---")
        print(f"原文: {text}")

        try:
            # 预测
            doc = nlp(text)

            # 提取结果
            tokens = [token.text for token in doc]
            labels = [f"{token.ent_iob_}-{token.ent_type_}" if token.ent_iob_ != 'O' else 'O' for token in doc]
            entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

            print(f"分词结果: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            print(f"识别的实体: {[(text[start:end], label) for start, end, label in entities]}")

            result = {
                'text': text,
                'tokens': tokens,
                'labels': labels,
                'entities': entities
            }

            # 添加EOS后处理
            if eos_processor:
                result = eos_processor.process_prediction_result(text, result)

            results.append(result)

        except Exception as e:
            print(f"预测失败: {e}")
            continue

    return results

def main():
    parser = argparse.ArgumentParser(description='测试训练好的模型')
    parser.add_argument('--framework', choices=['transformers', 'spacy', 'spacy_transformers', 'all'],
                       default='all', help='选择测试的框架')
    parser.add_argument('--file', choices=['zh', 'en', 'both'],
                       default='both', help='选择测试的文件')
    
    args = parser.parse_args()
    
    files_to_test = []
    if args.file in ['zh', 'both']:
        files_to_test.append('data/zh.txt')
    if args.file in ['en', 'both']:
        files_to_test.append('data/en.txt')
    
    all_results = {}
    
    for file_path in files_to_test:
        print(f"\n{'='*60}")
        print(f"测试文件: {file_path}")
        print(f"{'='*60}")
        
        file_results = {}

        if args.framework in ['transformers', 'all']:
            try:
                transformer_results = test_transformers_model(file_path, enable_eos=True)
                file_results['transformers'] = transformer_results
            except Exception as e:
                print(f"Transformers测试失败: {e}")

        if args.framework in ['spacy', 'all']:
            try:
                spacy_results = test_spacy_model(file_path, enable_eos=True)
                file_results['spacy'] = spacy_results
            except Exception as e:
                print(f"spaCy测试失败: {e}")

        if args.framework in ['spacy_transformers', 'all']:
            try:
                spacy_transformers_results = test_spacy_transformers_model(file_path, enable_eos=True)
                file_results['spacy_transformers'] = spacy_transformers_results
            except Exception as e:
                print(f"SpaCy-Transformers测试失败: {e}")
        
        all_results[file_path] = file_results
    
    # 保存结果
    output_file = f"validation_results_{args.framework}_{args.file}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 测试结果已保存到: {output_file}")

    # 显示测试总结
    print(f"\n{'='*60}")
    print("📊 测试总结")
    print(f"{'='*60}")
    for file_path, file_data in all_results.items():
        print(f"\n📁 文件: {file_path}")
        for framework, results in file_data.items():
            if results:
                print(f"  ✅ {framework}: {len(results)} 条结果")
            else:
                print(f"  ❌ {framework}: 测试失败或无结果")

if __name__ == "__main__":
    main()