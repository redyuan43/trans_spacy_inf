#!/usr/bin/env python3
"""
简单测试EOS功能
"""

from sentence_eos_processor import SentenceEOSProcessor

def test_eos_functionality():
    """测试EOS功能的简单示例"""
    processor = SentenceEOSProcessor()

    # 测试用例
    test_cases = [
        {
            'text': '温度是25摄氏度。压力为2.5帕斯卡。',
            'entities': [(3, 5, 'NUM'), (5, 8, 'UNIT'), (12, 15, 'NUM'), (15, 18, 'UNIT')]
        },
        {
            'text': 'The temperature is 25°C. The pressure is 2.5 Pa.',
            'entities': [(19, 21, 'NUM'), (21, 23, 'UNIT'), (41, 44, 'NUM'), (45, 47, 'UNIT')]
        },
        {
            'text': '使用HPLC分析样品。结果显示纯度为99.5%，pH值为7.4。',
            'entities': [(2, 6, 'TECH'), (21, 25, 'NUM'), (30, 33, 'NUM')]
        }
    ]

    print("=== EOS功能测试 ===\n")

    for i, case in enumerate(test_cases, 1):
        print(f"测试 {i}:")
        print(f"原文: {case['text']}")
        print(f"原始实体: {case['entities']}")

        # 构造预测结果
        result = {
            'text': case['text'],
            'entities': case['entities']
        }

        # 处理EOS
        processed_result = processor.process_prediction_result(case['text'], result)

        print(f"处理后实体: {processed_result['entities']}")
        print(f"EOS位置: {processed_result.get('eos_positions', [])}")

        # 显示实体详情
        print("实体详情:")
        for start, end, label in processed_result['entities']:
            entity_text = case['text'][start:end]
            print(f"  {entity_text} ({label}) [{start}:{end}]")

        print("-" * 60)

if __name__ == "__main__":
    test_eos_functionality()