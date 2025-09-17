#!/usr/bin/env python3
"""
检查结果文件中的EOS标记
"""

import json
import sys

def check_eos_in_file(filename):
    """检查JSON文件中的EOS标记"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取文件 {filename}: {e}")
        return

    print(f"=== 检查文件: {filename} ===\n")

    total_sentences = 0
    sentences_with_eos = 0
    total_eos_count = 0
    total_other_entities = 0

    # 遍历数据结构
    if isinstance(data, dict):
        for file_path, file_data in data.items():
            print(f"📁 文件: {file_path}")

            if isinstance(file_data, dict):
                for framework, results in file_data.items():
                    print(f"  🔧 框架: {framework}")

                    if isinstance(results, list):
                        for i, result in enumerate(results):
                            total_sentences += 1

                            if 'entities' in result and 'text' in result:
                                text = result['text']
                                entities = result['entities']

                                # 统计EOS和其他实体
                                eos_entities = []
                                other_entities = []

                                for entity in entities:
                                    if isinstance(entity, (list, tuple)) and len(entity) >= 3:
                                        if entity[2] == 'EOS':
                                            eos_entities.append(entity)
                                            total_eos_count += 1
                                        else:
                                            other_entities.append(entity)
                                            total_other_entities += 1

                                if eos_entities:
                                    sentences_with_eos += 1
                                    print(f"    📝 句子 {i+1}: {text[:50]}{'...' if len(text) > 50 else ''}")
                                    print(f"       EOS数量: {len(eos_entities)}, 其他实体: {len(other_entities)}")

                                    # 显示EOS详情
                                    for start, end, label in eos_entities:
                                        eos_char = text[start:end]
                                        print(f"       EOS: '{eos_char}' 位置[{start}:{end}]")

                                    # 检查eos_processed标志
                                    if result.get('eos_processed', False):
                                        print(f"       ✅ EOS处理状态: 已处理")
                                    else:
                                        print(f"       ❓ EOS处理状态: 未知")
                                    print()
            print()

    # 输出统计信息
    print("=" * 60)
    print("📊 统计信息:")
    print(f"  总句子数: {total_sentences}")
    print(f"  包含EOS的句子: {sentences_with_eos}")
    print(f"  EOS覆盖率: {sentences_with_eos/total_sentences*100:.1f}%" if total_sentences > 0 else "  EOS覆盖率: 0%")
    print(f"  总EOS标记数: {total_eos_count}")
    print(f"  总其他实体数: {total_other_entities}")
    print(f"  EOS/总实体比: {total_eos_count/(total_eos_count+total_other_entities)*100:.1f}%" if (total_eos_count+total_other_entities) > 0 else "  EOS/总实体比: 0%")

def main():
    if len(sys.argv) < 2:
        print("用法: python check_eos_in_results.py <json_file>")
        print("\n示例:")
        print("  python check_eos_in_results.py validation_results_transformers_zh.json")
        print("  python check_eos_in_results.py validation_results_both_both.json")
        return

    filename = sys.argv[1]
    check_eos_in_file(filename)

if __name__ == "__main__":
    main()