import re
import json
from typing import List, Tuple, Dict, Any

class SentenceEOSProcessor:
    """
    后处理工具：在NER结果基础上添加句子结束符号(EOS)标记
    """

    def __init__(self):
        # 中文句子结束符号
        self.chinese_eos_patterns = [
            r'[。！？]',  # 中文句号、感叹号、问号
            r'[.!?]',     # 英文标点
            r'[:：]\s*$', # 冒号结尾（用于列表项等）
        ]

        # 英文句子结束符号
        self.english_eos_patterns = [
            r'[.!?](?=\s|$)',  # 句号、感叹号、问号后跟空格或结尾
            r'[:]\s*$',        # 冒号结尾
        ]

        # 编译正则表达式
        self.chinese_eos_regex = re.compile('|'.join(self.chinese_eos_patterns))
        self.english_eos_regex = re.compile('|'.join(self.english_eos_patterns))

    def detect_language(self, text: str) -> str:
        """
        简单的语言检测
        """
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.strip())

        if total_chars == 0:
            return 'unknown'

        chinese_ratio = chinese_chars / total_chars
        return 'chinese' if chinese_ratio > 0.3 else 'english'

    def find_sentence_endings(self, text: str) -> List[int]:
        """
        找到句子结束位置
        返回字符位置列表
        """
        language = self.detect_language(text)

        if language == 'chinese':
            regex = self.chinese_eos_regex
        else:
            regex = self.english_eos_regex

        endings = []
        for match in regex.finditer(text):
            # 记录标点符号的结束位置
            endings.append(match.end())

        return endings

    def add_eos_to_entities(self, text: str, entities: List[Tuple]) -> List[Tuple]:
        """
        在实体列表中添加EOS标记

        Args:
            text: 原始文本
            entities: [(start, end, label), ...] 格式的实体列表

        Returns:
            包含EOS标记的实体列表
        """
        # 获取句子结束位置
        eos_positions = self.find_sentence_endings(text)

        # 创建新的实体列表
        new_entities = list(entities)  # 复制原有实体

        # 添加EOS标记
        for pos in eos_positions:
            # 检查是否与现有实体重叠
            overlaps = False
            for start, end, label in entities:
                if start <= pos <= end:
                    overlaps = True
                    break

            if not overlaps and pos <= len(text):
                # 找到实际的标点符号位置
                # 向前查找最近的标点符号
                actual_start = pos - 1
                while actual_start >= 0 and text[actual_start] not in '。！？.!?：:':
                    actual_start -= 1

                if actual_start >= 0:
                    new_entities.append((actual_start, pos, 'EOS'))

        # 按位置排序
        new_entities.sort(key=lambda x: x[0])

        return new_entities

    def add_eos_to_bio_tags(self, text: str, bio_tags: List[str]) -> List[str]:
        """
        在BIO标记序列中添加EOS标记

        Args:
            text: 原始文本
            bio_tags: BIO标记列表，与文本字符一一对应

        Returns:
            包含EOS标记的BIO标记列表
        """
        if len(bio_tags) != len(text):
            raise ValueError(f"BIO标记长度({len(bio_tags)})与文本长度({len(text)})不匹配")

        # 获取句子结束位置
        eos_positions = self.find_sentence_endings(text)

        # 创建新的BIO标记列表
        new_bio_tags = list(bio_tags)

        # 添加EOS标记
        for pos in eos_positions:
            if pos <= len(text):
                # 向前查找标点符号的实际位置
                actual_pos = pos - 1
                while actual_pos >= 0 and text[actual_pos] not in '。！？.!?：:':
                    actual_pos -= 1

                if actual_pos >= 0 and actual_pos < len(new_bio_tags):
                    # 只有当前位置不是其他实体时才标记为EOS
                    if new_bio_tags[actual_pos] == 'O':
                        new_bio_tags[actual_pos] = 'B-EOS'

        return new_bio_tags

    def process_prediction_result(self, text: str, prediction_result: Dict) -> Dict:
        """
        处理模型预测结果，添加EOS标记

        Args:
            text: 原始文本
            prediction_result: 模型预测结果字典

        Returns:
            处理后的预测结果
        """
        result = prediction_result.copy()

        # 处理实体列表格式
        if 'entities' in result:
            result['entities'] = self.add_eos_to_entities(text, result['entities'])

        # 处理BIO标记格式
        if 'bio_tags' in result and len(result['bio_tags']) == len(text):
            result['bio_tags'] = self.add_eos_to_bio_tags(text, result['bio_tags'])

        # 添加处理信息
        result['eos_processed'] = True
        result['eos_positions'] = self.find_sentence_endings(text)

        return result

    def process_validation_file(self, input_file: str, output_file: str):
        """
        处理验证结果文件，添加EOS标记
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            # 处理结果列表
            for item in data:
                if 'text' in item:
                    processed = self.process_prediction_result(item['text'], item)
                    item.update(processed)
        elif isinstance(data, dict):
            # 处理单个结果
            if 'text' in data:
                processed = self.process_prediction_result(data['text'], data)
                data.update(processed)

        # 保存处理后的结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"已处理文件并保存到: {output_file}")

def main():
    """
    命令行使用示例
    """
    processor = SentenceEOSProcessor()

    # 测试示例
    test_cases = [
        "温度是25摄氏度。压力为2.5帕斯卡。",
        "The temperature is 25°C. The pressure is 2.5 Pa.",
        "数据处理完成，请检查结果。",
        "Processing completed. Check the results."
    ]

    print("=== 句子结束符号识别测试 ===")
    for i, text in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {text}")

        # 模拟一些实体
        entities = []
        if "25" in text:
            start = text.find("25")
            entities.append((start, start + 2, "NUM"))
        if "摄氏度" in text:
            start = text.find("摄氏度")
            entities.append((start, start + 3, "UNIT"))
        if "°C" in text:
            start = text.find("°C")
            entities.append((start, start + 2, "UNIT"))
        if "2.5" in text:
            start = text.find("2.5")
            entities.append((start, start + 3, "NUM"))
        if "帕斯卡" in text:
            start = text.find("帕斯卡")
            entities.append((start, start + 3, "UNIT"))
        if "Pa" in text:
            start = text.find("Pa")
            entities.append((start, start + 2, "UNIT"))

        print(f"原始实体: {entities}")

        # 添加EOS标记
        new_entities = processor.add_eos_to_entities(text, entities)
        print(f"添加EOS后: {new_entities}")

        # 显示句子结束位置
        eos_positions = processor.find_sentence_endings(text)
        print(f"句子结束位置: {eos_positions}")

if __name__ == "__main__":
    main()