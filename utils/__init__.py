"""
工具函数包
"""

import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_path):
    """保存配置文件"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_experiment_dir(base_dir, experiment_name):
    """创建实验目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"

    # 创建子目录
    subdirs = ['logs', 'models', 'figures', 'data', 'checkpoints']
    for subdir in subdirs:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)

    return exp_dir


def calculate_metrics(cm):
    """计算性能指标"""
    TN, FP, FN, TP = cm.ravel()

    metrics = {
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'TN': int(TN),
        'accuracy': (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0,
        'sensitivity': TP / (TP + FN) if (TP + FN) > 0 else 0,
        'specificity': TN / (TN + FP) if (TN + FP) > 0 else 0,
        'precision': TP / (TP + FP) if (TP + FP) > 0 else 0,
        'f1_score': 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    }

    metrics['gmean'] = np.sqrt(metrics['sensitivity'] * metrics['specificity'])

    return metrics


def print_metrics(metrics, title="性能指标"):
    """打印性能指标"""
    print(f"\n{'=' * 50}")
    print(f"{title}")
    print(f"{'=' * 50}")
    print(f"TP: {metrics['TP']}, FP: {metrics['FP']}")
    print(f"FN: {metrics['FN']}, TN: {metrics['TN']}")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"灵敏度(召回率): {metrics['sensitivity']:.4f}")
    print(f"特异度: {metrics['specificity']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"F1分数: {metrics['f1_score']:.4f}")
    print(f"几何均值: {metrics['gmean']:.4f}")


__all__ = [
    'load_config',
    'save_config',
    'create_experiment_dir',
    'calculate_metrics',
    'print_metrics'
]