#!/usr/bin/env python3
"""
对比测试：展示有无EOS功能的区别
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentence_eos_processor import SentenceEOSProcessor

def test_with_real_data():
    """使用真实数据测试EOS功能"""
    processor = SentenceEOSProcessor()

    # 从zh.txt取几个有句号的句子
    test_sentences = [
        "氯化钠为无色晶体。",
        "形态B是已知的热力学稳定性最好的形态。",
        "熔化后开始出现分解。",
        "在此范围内无固态变化。"
    ]

    print("=== EOS功能对比测试 ===\n")

    for i, text in enumerate(test_sentences, 1):
        print(f"测试句子 {i}: {text}")
        print(f"长度: {len(text)} 字符")

        # 模拟一些基础实体（简化版）
        entities_without_eos = []
        if "氯化钠" in text:
            start = text.find("氯化钠")
            entities_without_eos.append((start, start+3, "TECH"))
        if "形态B" in text:
            start = text.find("形态B")
            entities_without_eos.append((start, start+2, "TECH"))

        print(f"原始实体（无EOS）: {entities_without_eos}")

        # 构造模拟结果
        result_without_eos = {
            'text': text,
            'entities': entities_without_eos
        }

        # 应用EOS处理
        result_with_eos = processor.process_prediction_result(text, result_without_eos)

        print(f"添加EOS后的实体: {result_with_eos['entities']}")
        print(f"EOS位置: {result_with_eos.get('eos_positions', [])}")

        # 详细显示所有实体
        print("实体详情:")
        for start, end, label in result_with_eos['entities']:
            entity_text = text[start:end]
            print(f"  '{entity_text}' ({label}) [{start}:{end}]")

        print("-" * 80)

def test_model_output_format():
    """测试模型输出格式，显示EOS如何体现"""
    from sentence_eos_processor import SentenceEOSProcessor

    processor = SentenceEOSProcessor()

    # 模拟模型预测结果（像真实模型输出那样）
    test_text = "温度控制在25°C。压力为2.5 MPa。"

    # 模拟Transformers模型的输出格式
    model_result = {
        'text': test_text,
        'tokens': ['温', '度', '控', '制', '在', '25', '°', 'C', '。', '压', '力', '为', '2', '.', '5', ' ', 'M', 'P', 'a', '。'],
        'labels': ['O', 'O', 'O', 'O', 'O', 'B-NUM', 'I-NUM', 'B-UNIT', 'O', 'O', 'O', 'O', 'B-NUM', 'I-NUM', 'I-NUM', 'O', 'B-UNIT', 'I-UNIT', 'I-UNIT', 'O'],
        'entities': [(5, 7, 'NUM'), (7, 9, 'UNIT'), (12, 15, 'NUM'), (16, 19, 'UNIT')]  # 基于字符位置
    }

    print("=== 模型输出格式测试 ===\n")
    print(f"原始文本: {test_text}")
    print("=" * 40)

    print("\n【处理前】")
    print(f"实体数量: {len(model_result['entities'])}")
    for i, (start, end, label) in enumerate(model_result['entities'], 1):
        entity_text = test_text[start:end]
        print(f"  {i}. '{entity_text}' ({label}) [{start}:{end}]")

    # 应用EOS处理
    processed_result = processor.process_prediction_result(test_text, model_result)

    print("\n【处理后】")
    print(f"实体数量: {len(processed_result['entities'])}")
    print(f"EOS处理状态: {processed_result.get('eos_processed', False)}")
    print(f"EOS位置: {processed_result.get('eos_positions', [])}")

    # 分类显示实体
    tech_entities = []
    num_entities = []
    unit_entities = []
    eos_entities = []

    for start, end, label in processed_result['entities']:
        entity_text = test_text[start:end]
        if label == 'EOS':
            eos_entities.append((entity_text, start, end))
        elif label == 'TECH':
            tech_entities.append((entity_text, start, end))
        elif label == 'NUM':
            num_entities.append((entity_text, start, end))
        elif label == 'UNIT':
            unit_entities.append((entity_text, start, end))

    print(f"\n技术术语 (TECH): {len(tech_entities)}")
    for text, start, end in tech_entities:
        print(f"  • '{text}' [{start}:{end}]")

    print(f"\n数值 (NUM): {len(num_entities)}")
    for text, start, end in num_entities:
        print(f"  • '{text}' [{start}:{end}]")

    print(f"\n单位 (UNIT): {len(unit_entities)}")
    for text, start, end in unit_entities:
        print(f"  • '{text}' [{start}:{end}]")

    print(f"\n句子结束 (EOS): {len(eos_entities)}")
    for text, start, end in eos_entities:
        print(f"  • '{text}' [{start}:{end}]")

if __name__ == "__main__":
    test_with_real_data()
    print("\n" + "="*80 + "\n")
    test_model_output_format()