#!/usr/bin/env python3
"""
Data Preprocessing and Annotation Tool
数据预处理和标注工具
"""

import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import os

@dataclass
class TokenLabel:
    """Token with label information"""
    token: str
    label: str
    start_pos: int
    end_pos: int

class DataAnnotator:
    """Data annotation tool for creating training data"""
    
    def __init__(self):
        # Technical term patterns
        self.technical_patterns = {
            'chemical': r'氯化钠|luminex\s+chloride|NaCl|HCl|Form\s+[AB]',
            'units': r'mg/mL|mcg/mL|°C|nm|pH|Log\s+P|pKa',
            'techniques': r'FTIR|XRPD|TGA|DSC|UV|GVS',
            'solutions': r'DMF|RLF|HGF|TeSSIF|FeSSIF|SLF|SGF|FaSSIF',
            'numbers': r'\d+(?:\.\d+)?(?:[～\-]\d+(?:\.\d+)?)?'
        }
        
        # Label mapping
        self.label_mapping = {
            'chemical': 'TECH',
            'units': 'UNIT',
            'techniques': 'TECH',
            'solutions': 'TECH',
            'numbers': 'NUM'
        }
    
    def annotate_text(self, text: str) -> List[TokenLabel]:
        """Annotate text with BIO labels"""
        annotations = []
        
        # Find all matches
        all_matches = []
        for category, pattern in self.technical_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                all_matches.append({
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group(),
                    'category': category,
                    'label': self.label_mapping[category]
                })
        
        # Sort by start position
        all_matches.sort(key=lambda x: x['start'])
        
        # Create character-level annotations
        current_pos = 0
        for match in all_matches:
            # Add 'O' labels for text before this match
            while current_pos < match['start']:
                if text[current_pos].strip():
                    annotations.append(TokenLabel(
                        token=text[current_pos],
                        label='O',
                        start_pos=current_pos,
                        end_pos=current_pos + 1
                    ))
                current_pos += 1
            
            # Add labeled tokens for this match
            match_text = match['text']
            label = match['label']
            
            # For multi-character terms, use BIO labeling
            if len(match_text) == 1:
                annotations.append(TokenLabel(
                    token=match_text,
                    label=f'B-{label}',
                    start_pos=match['start'],
                    end_pos=match['end']
                ))
            else:
                # First character gets B- label
                annotations.append(TokenLabel(
                    token=match_text[0],
                    label=f'B-{label}',
                    start_pos=match['start'],
                    end_pos=match['start'] + 1
                ))
                
                # Remaining characters get I- labels
                for i, char in enumerate(match_text[1:], 1):
                    annotations.append(TokenLabel(
                        token=char,
                        label=f'I-{label}',
                        start_pos=match['start'] + i,
                        end_pos=match['start'] + i + 1
                    ))
            
            current_pos = match['end']
        
        # Add remaining 'O' labels
        while current_pos < len(text):
            if text[current_pos].strip():
                annotations.append(TokenLabel(
                    token=text[current_pos],
                    label='O',
                    start_pos=current_pos,
                    end_pos=current_pos + 1
                ))
            current_pos += 1
        
        return annotations
    
    def create_training_sample(self, text: str) -> Dict[str, Any]:
        """Create a training sample from text"""
        annotations = self.annotate_text(text)
        
        return {
            'text': text,
            'tokens': [ann.token for ann in annotations],
            'labels': [ann.label for ann in annotations],
            'annotations': [
                {
                    'token': ann.token,
                    'label': ann.label,
                    'start_pos': ann.start_pos,
                    'end_pos': ann.end_pos
                }
                for ann in annotations
            ]
        }
    
    def process_file(self, input_path: str, output_path: str):
        """Process a file and create annotated training data"""
        samples = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into sentences/lines
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for line in lines:
            if len(line) > 1:  # Skip very short lines
                sample = self.create_training_sample(line)
                samples.append(sample)
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        print(f"Processed {len(samples)} samples from {input_path}")
        print(f"Saved to {output_path}")
        
        return samples

class TrainingDataGenerator:
    """Generate training data from existing files"""
    
    def __init__(self):
        self.annotator = DataAnnotator()
    
    def generate_from_reference_files(self):
        """Generate training data from zh.txt and en.txt"""
        samples = []
        
        # Process Chinese file
        if os.path.exists('zh.txt'):
            print("Processing zh.txt...")
            zh_samples = self.annotator.process_file('zh.txt', 'zh_training_data.json')
            samples.extend(zh_samples)
        
        # Process English file
        if os.path.exists('en.txt'):
            print("Processing en.txt...")
            en_samples = self.annotator.process_file('en.txt', 'en_training_data.json')
            samples.extend(en_samples)
        
        # Create combined dataset
        if samples:
            combined_path = 'combined_training_data.json'
            with open(combined_path, 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            
            print(f"\nCombined training data saved to {combined_path}")
            print(f"Total samples: {len(samples)}")
            
            # Print statistics
            self._print_statistics(samples)
            
            return samples
        else:
            print("No reference files found")
            return []
    
    def _print_statistics(self, samples: List[Dict[str, Any]]):
        """Print training data statistics"""
        label_counts = {}
        total_tokens = 0
        
        for sample in samples:
            for label in sample['labels']:
                label_counts[label] = label_counts.get(label, 0) + 1
                total_tokens += 1
        
        print("\n" + "="*50)
        print("Training Data Statistics")
        print("="*50)
        print(f"Total samples: {len(samples)}")
        print(f"Total tokens: {total_tokens}")
        print("\nLabel distribution:")
        
        for label, count in sorted(label_counts.items()):
            percentage = (count / total_tokens) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
    
    def create_synthetic_data(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Create synthetic training data"""
        print(f"Generating {num_samples} synthetic samples...")
        
        # Templates for synthetic data generation
        chinese_templates = [
            "氯化钠的{tech}示例图见图{num}。",
            "{tech}值为{num}，{unit}为{num2}。",
            "Form {form}在{temp}°C时熔化。",
            "使用{tech}进行分析，结果为{num}。",
            "溶解度为{num} {unit}，pH值为{num2}。"
        ]
        
        english_templates = [
            "The {tech} analysis shows {num} {unit}.",
            "Form {form} exhibits {tech} at {temp}°C.",
            "The solubility is {num} {unit} with pH {num2}.",
            "{tech} measurement indicates {num} {unit}.",
            "Sample shows {tech} pattern at {num} nm."
        ]
        
        # Vocabulary
        tech_terms = ['FTIR', 'XRPD', 'TGA', 'DSC', 'UV', 'GVS']
        units = ['mg/mL', 'mcg/mL', 'nm']
        forms = ['A', 'B']
        numbers = ['0.1234', '7.1', '3.3', '252', '15.6']
        
        samples = []
        
        for i in range(num_samples):
            # Alternate between Chinese and English
            if i % 2 == 0:
                template = np.random.choice(chinese_templates)
                text = template.format(
                    tech=np.random.choice(tech_terms),
                    num=np.random.choice(numbers),
                    num2=np.random.choice(numbers),
                    unit=np.random.choice(units),
                    form=np.random.choice(forms),
                    temp=np.random.choice(['252', '180', '300'])
                )
            else:
                template = np.random.choice(english_templates)
                text = template.format(
                    tech=np.random.choice(tech_terms),
                    num=np.random.choice(numbers),
                    num2=np.random.choice(numbers),
                    unit=np.random.choice(units),
                    form=np.random.choice(forms),
                    temp=np.random.choice(['252', '180', '300'])
                )
            
            sample = self.annotator.create_training_sample(text)
            samples.append(sample)
        
        # Save synthetic data
        synthetic_path = 'synthetic_training_data.json'
        with open(synthetic_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        print(f"Synthetic training data saved to {synthetic_path}")
        self._print_statistics(samples)
        
        return samples

def main():
    """Main data preprocessing script"""
    import numpy as np
    
    print("数据预处理和标注工具")
    print("="*50)
    
    generator = TrainingDataGenerator()
    
    # Generate from reference files
    reference_samples = generator.generate_from_reference_files()
    
    # Generate synthetic data
    synthetic_samples = generator.create_synthetic_data(50)
    
    # Combine all data
    all_samples = reference_samples + synthetic_samples
    
    if all_samples:
        # Save final dataset
        final_path = 'final_training_data.json'
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)
        
        print(f"\nFinal training data saved to {final_path}")
        print(f"Total samples: {len(all_samples)}")
        
        print("\nNext steps:")
        print("1. Run: python training_module.py")
        print("2. The training will use the generated data files")
        print("3. Trained model will be saved to ./trained_models/")
    else:
        print("No training data generated")

if __name__ == "__main__":
    main()