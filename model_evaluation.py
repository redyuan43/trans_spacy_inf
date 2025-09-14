#!/usr/bin/env python3
"""
NER模型评估和监控工具
提供训练过程监控、性能评估和结果可视化
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import argparse

class NERModelEvaluator:
    """NER模型评估器"""

    def __init__(self):
        self.history = {
            'loss': [],
            'learning_rate': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'epoch': []
        }
        self.entity_types = ['TECH', 'NUM', 'UNIT']

    def parse_training_log(self, log_file: str):
        """解析训练日志"""
        print(f"解析训练日志: {log_file}")

        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            if 'loss' in line.lower():
                # 尝试提取损失值
                try:
                    if 'epoch' in line.lower():
                        # Transformers格式: {'loss': 1.4398, 'epoch': 1.0}
                        data = eval(line)
                        self.history['loss'].append(data.get('loss', 0))
                        self.history['epoch'].append(data.get('epoch', 0))
                        if 'learning_rate' in data:
                            self.history['learning_rate'].append(data['learning_rate'])
                    else:
                        # SpaCy格式: Epoch 2/10, 损失: 732.85
                        parts = line.split('损失:')
                        if len(parts) > 1:
                            loss = float(parts[1].strip())
                            self.history['loss'].append(loss)
                except:
                    continue

        return self.history

    def evaluate_predictions(self, predictions_file: str):
        """评估预测结果"""
        print(f"\n评估预测结果: {predictions_file}")

        with open(predictions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_true = []
        all_pred = []

        # 收集所有标签
        for item in data:
            if 'true_labels' in item and 'pred_labels' in item:
                all_true.extend(item['true_labels'])
                all_pred.extend(item['pred_labels'])

        if not all_true:
            print("警告：没有找到真实标签，使用预测标签")
            # 如果没有真实标签，分析预测分布
            return self.analyze_predictions_only(data)

        # 计算整体指标
        report = classification_report(
            all_true, all_pred,
            output_dict=True,
            zero_division=0
        )

        # 打印结果
        self.print_evaluation_results(report)

        # 生成混淆矩阵
        self.plot_confusion_matrix(all_true, all_pred)

        return report

    def analyze_predictions_only(self, data: List[Dict]):
        """仅分析预测结果（无真实标签时）"""
        entity_counts = {}
        total_entities = 0

        for item in data:
            if 'entities' in item:
                for entity in item['entities']:
                    label = entity.get('label', entity.get('type', 'UNKNOWN'))
                    entity_counts[label] = entity_counts.get(label, 0) + 1
                    total_entities += 1

        print("\n预测实体统计:")
        print(f"总实体数: {total_entities}")
        print("\n各类实体分布:")
        for label, count in sorted(entity_counts.items()):
            percentage = (count / total_entities * 100) if total_entities > 0 else 0
            print(f"  {label:10} : {count:5} ({percentage:.1f}%)")

        return entity_counts

    def print_evaluation_results(self, report: Dict):
        """打印评估结果"""
        print("\n" + "="*60)
        print("模型评估结果")
        print("="*60)

        # 整体性能
        if 'weighted avg' in report:
            avg = report['weighted avg']
            print(f"\n整体性能:")
            print(f"  F1-Score : {avg['f1-score']:.3f}")
            print(f"  Precision: {avg['precision']:.3f}")
            print(f"  Recall   : {avg['recall']:.3f}")

            # 判断性能等级
            f1 = avg['f1-score']
            if f1 > 0.9:
                level = "优秀 ⭐⭐⭐⭐⭐"
            elif f1 > 0.8:
                level = "良好 ⭐⭐⭐⭐"
            elif f1 > 0.7:
                level = "及格 ⭐⭐⭐"
            else:
                level = "需改进 ⭐⭐"
            print(f"  性能等级 : {level}")

        # 各类实体性能
        print(f"\n各类实体性能:")
        print(f"{'实体类型':<10} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Support':>8}")
        print("-" * 50)

        for label in self.entity_types + ['O']:
            if label in report:
                metrics = report[label]
                print(f"{label:<10} {metrics['f1-score']:>8.3f} {metrics['precision']:>10.3f} "
                      f"{metrics['recall']:>8.3f} {metrics.get('support', 0):>8}")

    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.history['loss']:
            print("没有训练历史数据")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 损失曲线
        if self.history['loss']:
            axes[0, 0].plot(self.history['loss'], 'b-', label='Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].grid(True)

            # 判断趋势
            if len(self.history['loss']) > 1:
                trend = "下降" if self.history['loss'][-1] < self.history['loss'][0] else "上升"
                axes[0, 0].text(0.02, 0.98, f'趋势: {trend}',
                               transform=axes[0, 0].transAxes, va='top')

        # 学习率曲线
        if self.history['learning_rate']:
            axes[0, 1].plot(self.history['learning_rate'], 'g-', label='LR')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].grid(True)

        # F1分数曲线
        if self.history['f1']:
            axes[1, 0].plot(self.history['f1'], 'r-', label='F1')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].set_title('F1 Score Progress')
            axes[1, 0].grid(True)
            axes[1, 0].set_ylim([0, 1])

        # 精确率vs召回率
        if self.history['precision'] and self.history['recall']:
            axes[1, 1].plot(self.history['precision'], 'b-', label='Precision')
            axes[1, 1].plot(self.history['recall'], 'r-', label='Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Precision vs Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            axes[1, 1].set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=100)
        print("\n训练曲线已保存到: training_curves.png")
        plt.show()

    def plot_confusion_matrix(self, y_true: List, y_pred: List):
        """绘制混淆矩阵"""
        # 获取所有标签
        labels = sorted(list(set(y_true + y_pred)))

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # 绘制热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=100)
        print("混淆矩阵已保存到: confusion_matrix.png")
        plt.show()

    def diagnose_model(self, report: Dict):
        """诊断模型问题"""
        print("\n" + "="*60)
        print("模型诊断")
        print("="*60)

        if 'weighted avg' not in report:
            print("无法进行诊断（缺少评估数据）")
            return

        avg = report['weighted avg']
        precision = avg['precision']
        recall = avg['recall']
        f1 = avg['f1-score']

        problems = []
        suggestions = []

        # 诊断问题
        if precision > recall + 0.1:
            problems.append("高精确率，低召回率")
            suggestions.append("- 模型过于保守，遗漏了很多实体")
            suggestions.append("- 建议：增加训练数据，降低分类阈值")

        elif recall > precision + 0.1:
            problems.append("低精确率，高召回率")
            suggestions.append("- 模型过于激进，误识别了很多非实体")
            suggestions.append("- 建议：增加负样本，提高分类阈值")

        if f1 < 0.7:
            problems.append("整体性能较差")
            suggestions.append("- 可能数据量不足或标注质量问题")
            suggestions.append("- 建议：增加数据量，检查标注一致性")

        # 检查各类实体
        for entity_type in self.entity_types:
            if entity_type in report:
                entity_f1 = report[entity_type]['f1-score']
                if entity_f1 < 0.6:
                    problems.append(f"{entity_type}类实体识别效果差")
                    suggestions.append(f"- 增加{entity_type}类型的训练样本")

        # 输出诊断结果
        if problems:
            print("\n发现的问题:")
            for problem in problems:
                print(f"  ⚠️ {problem}")

            print("\n优化建议:")
            for suggestion in suggestions:
                print(f"  {suggestion}")
        else:
            print("  ✅ 模型表现良好，无明显问题")

    def compare_models(self, model1_results: str, model2_results: str):
        """比较两个模型"""
        print("\n" + "="*60)
        print("模型对比")
        print("="*60)

        # 加载结果
        with open(model1_results, 'r') as f:
            results1 = json.load(f)
        with open(model2_results, 'r') as f:
            results2 = json.load(f)

        # 提取指标（简化示例）
        model1_name = Path(model1_results).stem
        model2_name = Path(model2_results).stem

        print(f"\n{'指标':<15} {model1_name:<20} {model2_name:<20} {'差异':<10}")
        print("-" * 65)

        # 这里需要根据实际结果格式调整
        # 示例对比
        metrics = ['f1', 'precision', 'recall']
        for metric in metrics:
            val1 = 0.85  # 示例值
            val2 = 0.88  # 示例值
            diff = val2 - val1
            winner = "→" if abs(diff) < 0.01 else ("↑" if diff > 0 else "↓")
            print(f"{metric:<15} {val1:<20.3f} {val2:<20.3f} {diff:+.3f} {winner}")

def monitor_training():
    """实时监控训练（简化版）"""
    import time
    import sys

    print("开始监控训练...")
    print("按Ctrl+C停止\n")

    evaluator = NERModelEvaluator()

    try:
        step = 0
        while True:
            # 模拟训练步骤
            step += 1
            loss = max(0.1, 2.0 - step * 0.1 + np.random.random() * 0.2)

            # 更新历史
            evaluator.history['loss'].append(loss)

            # 显示进度
            sys.stdout.write(f"\rStep {step:4d} | Loss: {loss:.4f}")
            sys.stdout.flush()

            # 每10步评估一次
            if step % 10 == 0:
                f1 = min(0.95, 0.5 + step * 0.01)
                print(f" | F1: {f1:.3f}")

            time.sleep(0.5)

            if step >= 50:
                break

    except KeyboardInterrupt:
        print("\n\n训练监控已停止")

    # 绘制曲线
    evaluator.plot_training_curves()

def main():
    parser = argparse.ArgumentParser(description='NER模型评估工具')
    parser.add_argument('--mode', choices=['evaluate', 'monitor', 'compare', 'diagnose'],
                       default='evaluate', help='运行模式')
    parser.add_argument('--predictions', help='预测结果文件')
    parser.add_argument('--log', help='训练日志文件')
    parser.add_argument('--model1', help='模型1结果文件（对比用）')
    parser.add_argument('--model2', help='模型2结果文件（对比用）')

    args = parser.parse_args()

    evaluator = NERModelEvaluator()

    if args.mode == 'evaluate':
        if args.predictions:
            report = evaluator.evaluate_predictions(args.predictions)
            evaluator.diagnose_model(report)
        else:
            print("请提供预测结果文件: --predictions <file>")

    elif args.mode == 'monitor':
        monitor_training()

    elif args.mode == 'compare':
        if args.model1 and args.model2:
            evaluator.compare_models(args.model1, args.model2)
        else:
            print("请提供两个模型结果文件: --model1 <file> --model2 <file>")

    elif args.mode == 'diagnose':
        if args.predictions:
            report = evaluator.evaluate_predictions(args.predictions)
            evaluator.diagnose_model(report)
        else:
            print("请提供预测结果文件: --predictions <file>")

    # 如果有训练日志，绘制曲线
    if args.log:
        evaluator.parse_training_log(args.log)
        evaluator.plot_training_curves()

if __name__ == "__main__":
    main()