#!/usr/bin/env python3
"""
实体标注辅助工具
帮助快速准确地标注NER训练数据
"""

import json
import re
from typing import List, Dict, Tuple
import argparse
from pathlib import Path

class AnnotationHelper:
    """标注辅助类"""
    
    def __init__(self):
        # 预定义的实体词典
        self.TECH_TERMS = [
            "FTIR", "HPLC", "GC-MS", "NMR", "XRD", "DSC", "TGA", "UV", "IR",
            "XRPD", "LC-MS", "ICP-MS", "SEM", "TEM", "AFM", "XPS", "ELISA",
            "PCR", "qPCR", "RT-PCR", "Western blot", "SDS-PAGE", "FPLC",
            "CPU", "GPU", "RAM", "SSD", "HDD", "API", "SDK", "IDE"
        ]
        
        self.UNITS = [
            # 温度
            "°C", "℃", "°F", "K", "摄氏度", "华氏度",
            # 浓度
            "mg/mL", "μg/mL", "ng/mL", "pg/mL", "g/L", "mol/L", "mM", "μM", "nM", "pM",
            # 长度
            "nm", "μm", "mm", "cm", "m", "km", "Å",
            # 时间
            "s", "ms", "μs", "ns", "min", "h", "天", "小时", "分钟", "秒",
            # 质量
            "mg", "μg", "ng", "pg", "g", "kg", "吨",
            # 体积
            "mL", "μL", "nL", "pL", "L",
            # 压力
            "Pa", "kPa", "MPa", "bar", "atm", "psi",
            # 其他
            "%", "pH", "rpm", "Hz", "kHz", "MHz", "GHz", "V", "mV", "A", "mA",
            "W", "mW", "kW", "J", "kJ", "cal", "kcal"
        ]
        
        # 数字正则表达式
        self.NUMBER_PATTERN = re.compile(r'\d+\.?\d*')
        
    def find_entity_positions(self, text: str, entity: str) -> Tuple[int, int]:
        """找到实体在文本中的位置"""
        start = text.find(entity)
        if start != -1:
            end = start + len(entity)
            return start, end
        return None, None
    
    def auto_find_numbers(self, text: str) -> List[Dict]:
        """自动查找所有数字"""
        entities = []
        for match in self.NUMBER_PATTERN.finditer(text):
            entities.append({
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "label": "NUM"
            })
        return entities
    
    def auto_find_tech_terms(self, text: str) -> List[Dict]:
        """自动查找技术术语"""
        entities = []
        for term in self.TECH_TERMS:
            if term in text:
                start = text.find(term)
                entities.append({
                    "text": term,
                    "start": start,
                    "end": start + len(term),
                    "label": "TECH"
                })
        return entities
    
    def auto_find_units(self, text: str) -> List[Dict]:
        """自动查找单位"""
        entities = []
        for unit in self.UNITS:
            if unit in text:
                start = text.find(unit)
                if start != -1:
                    entities.append({
                        "text": unit,
                        "start": start,
                        "end": start + len(unit),
                        "label": "UNIT"
                    })
        return entities
    
    def auto_annotate(self, text: str) -> Dict:
        """自动标注文本"""
        entities = []
        
        # 查找所有类型的实体
        tech_entities = self.auto_find_tech_terms(text)
        unit_entities = self.auto_find_units(text)
        num_entities = self.auto_find_numbers(text)
        
        # 合并实体（避免重叠）
        all_entities = tech_entities + unit_entities + num_entities
        
        # 按位置排序
        all_entities.sort(key=lambda x: x['start'])
        
        # 去除重叠的实体
        non_overlapping = []
        for entity in all_entities:
            if not non_overlapping:
                non_overlapping.append(entity)
            else:
                last = non_overlapping[-1]
                # 检查是否重叠
                if entity['start'] >= last['end']:
                    non_overlapping.append(entity)
        
        return {
            "text": text,
            "entities": non_overlapping
        }
    
    def convert_to_bio(self, text: str, entities: List[Dict]) -> List[str]:
        """转换为BIO标签格式"""
        labels = ['O'] * len(text)
        
        for entity in entities:
            start = entity['start']
            end = entity['end']
            label = entity['label']
            
            if start < len(labels):
                labels[start] = f'B-{label}'
                for i in range(start + 1, min(end, len(labels))):
                    labels[i] = f'I-{label}'
        
        return labels
    
    def verify_annotation(self, text: str, entities: List[Dict]) -> List[str]:
        """验证标注的正确性"""
        issues = []
        
        for i, entity in enumerate(entities):
            start = entity['start']
            end = entity['end']
            label = entity.get('label', 'UNKNOWN')
            
            # 检查边界
            if start < 0 or end > len(text):
                issues.append(f"实体{i}: 位置超出文本范围")
                continue
            
            # 提取实体文本
            entity_text = text[start:end]
            
            # 检查空实体
            if not entity_text:
                issues.append(f"实体{i}: 实体为空")
            elif entity_text.isspace():
                issues.append(f"实体{i}: 实体只包含空格")
            
            # 检查实体内容
            if 'text' in entity and entity['text'] != entity_text:
                issues.append(f"实体{i}: 标注文本'{entity['text']}'与实际文本'{entity_text}'不匹配")
        
        # 检查重叠
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1 = entities[i]
                e2 = entities[j]
                if not (e1['end'] <= e2['start'] or e2['end'] <= e1['start']):
                    issues.append(f"实体{i}和实体{j}重叠")
        
        return issues
    
    def create_training_format(self, text: str, entities: List[Dict]) -> Dict:
        """创建训练所需的格式"""
        # 转换为字符列表
        tokens = list(text)
        
        # 生成BIO标签
        labels = self.convert_to_bio(text, entities)
        
        # 创建annotations
        annotations = []
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label != 'O':
                annotations.append({
                    'token': token,
                    'label': label,
                    'start_pos': i,
                    'end_pos': i + 1
                })
        
        return {
            'text': text,
            'tokens': tokens,
            'labels': labels,
            'annotations': annotations
        }
    
    def interactive_annotation(self):
        """交互式标注模式"""
        print("="*60)
        print("交互式实体标注工具")
        print("="*60)
        print("输入文本进行自动标注，然后可以手动修正")
        print("命令：")
        print("  add <实体文本> <类型> - 添加实体")
        print("  remove <索引> - 删除实体")
        print("  save <文件名> - 保存结果")
        print("  quit - 退出")
        print("-"*60)
        
        annotations = []
        
        while True:
            text = input("\n请输入要标注的文本（输入quit退出）: ").strip()
            
            if text.lower() == 'quit':
                break
            
            if not text:
                continue
            
            # 自动标注
            result = self.auto_annotate(text)
            print(f"\n原文: {text}")
            print("\n自动识别的实体:")
            
            for i, entity in enumerate(result['entities']):
                entity_text = text[entity['start']:entity['end']]
                print(f"  {i}: [{entity['start']}:{entity['end']}] '{entity_text}' ({entity['label']})")
            
            # 手动修正
            while True:
                cmd = input("\n输入命令（直接回车接受当前标注）: ").strip()
                
                if not cmd:
                    # 接受当前标注
                    training_format = self.create_training_format(text, result['entities'])
                    annotations.append(training_format)
                    print("✓ 标注已保存")
                    break
                
                parts = cmd.split()
                if parts[0] == 'add' and len(parts) >= 3:
                    # 添加实体
                    entity_text = parts[1]
                    label = parts[2]
                    start = text.find(entity_text)
                    if start != -1:
                        result['entities'].append({
                            'text': entity_text,
                            'start': start,
                            'end': start + len(entity_text),
                            'label': label
                        })
                        print(f"✓ 已添加: '{entity_text}' ({label})")
                    else:
                        print(f"✗ 未找到文本: '{entity_text}'")
                
                elif parts[0] == 'remove' and len(parts) >= 2:
                    # 删除实体
                    try:
                        idx = int(parts[1])
                        if 0 <= idx < len(result['entities']):
                            removed = result['entities'].pop(idx)
                            print(f"✓ 已删除: {removed}")
                        else:
                            print(f"✗ 索引超出范围")
                    except ValueError:
                        print(f"✗ 无效的索引")
                
                elif parts[0] == 'save' and len(parts) >= 2:
                    # 保存到文件
                    filename = parts[1]
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(annotations, f, ensure_ascii=False, indent=2)
                    print(f"✓ 已保存到: {filename}")
        
        return annotations

def demo():
    """演示标注功能"""
    helper = AnnotationHelper()
    
    print("="*60)
    print("实体标注辅助工具演示")
    print("="*60)
    
    # 测试文本
    test_texts = [
        "FTIR测量显示在3000 cm⁻¹处有吸收峰",
        "反应温度控制在100°C，时间为30分钟",
        "使用HPLC分析，流速1.0 mL/min，检测波长254 nm",
        "CPU温度85度，内存16 GB，硬盘500 GB"
    ]
    
    for text in test_texts:
        print(f"\n原文: {text}")
        print("-"*40)
        
        # 自动标注
        result = helper.auto_annotate(text)
        
        print("自动识别的实体:")
        for entity in result['entities']:
            entity_text = text[entity['start']:entity['end']]
            print(f"  [{entity['start']:3}:{entity['end']:3}] '{entity_text:10}' → {entity['label']}")
        
        # 验证
        issues = helper.verify_annotation(text, result['entities'])
        if issues:
            print("发现问题:")
            for issue in issues:
                print(f"  ⚠️ {issue}")
        else:
            print("✓ 标注验证通过")
        
        # 转换为训练格式
        training_data = helper.create_training_format(text, result['entities'])
        print(f"\nBIO标签（前20个）:")
        for i, (char, label) in enumerate(zip(training_data['tokens'][:20], 
                                              training_data['labels'][:20])):
            if label != 'O':
                print(f"  {char} → {label}")

def main():
    parser = argparse.ArgumentParser(description='实体标注辅助工具')
    parser.add_argument('--mode', choices=['demo', 'interactive', 'file'], 
                       default='demo', help='运行模式')
    parser.add_argument('--input', help='输入文件路径')
    parser.add_argument('--output', help='输出文件路径')
    
    args = parser.parse_args()
    
    helper = AnnotationHelper()
    
    if args.mode == 'demo':
        demo()
    elif args.mode == 'interactive':
        helper.interactive_annotation()
    elif args.mode == 'file' and args.input:
        # 批量处理文件
        with open(args.input, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        
        annotations = []
        for text in texts:
            text = text.strip()
            if text:
                result = helper.auto_annotate(text)
                training_data = helper.create_training_format(text, result['entities'])
                annotations.append(training_data)
        
        # 保存结果
        output_file = args.output or 'annotated_data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        print(f"✓ 已处理 {len(annotations)} 条数据，保存到: {output_file}")

if __name__ == "__main__":
    main()