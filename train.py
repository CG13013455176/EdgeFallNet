
import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from models import create_edgefallnet

sys.path.append('/')

# ============================================
# PyCharm直接运行时的配置
# 在PyCharm中直接运行时，可以修改这里的默认配置
# ============================================
CONFIG_PATH = "configs/fallld.yaml"  # 默认配置
USE_GPU = True
GPU_ID = "0"
DEBUG_MODE = False


# ============================================
class DetailedTextLogger(tf.keras.callbacks.Callback):
    """详细的文本日志记录器，格式更易读"""

    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def on_train_begin(self, logs=None):
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("训练日志\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"{'Epoch':<8} {'Loss':<12} {'Acc':<12} {'Val Loss':<12} {'Val Acc':<12}\n")
            f.write("-" * 60 + "\n")

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(f"{epoch + 1:<8} "
                    f"{logs.get('loss', 'N/A'):<12.4f} "
                    f"{logs.get('accuracy', 'N/A'):<12.4f} "
                    f"{logs.get('val_loss', 'N/A'):<12.4f} "
                    f"{logs.get('val_accuracy', 'N/A'):<12.4f}\n")


class FallDetectionTrainer:
    """跌倒检测模型训练器"""
    def __init__(self, config_path, use_gpu=True, gpu_id="0", debug_mode=False):
        """
        初始化训练器
        Args:
            config_path: 配置文件路径
            use_gpu: 是否使用GPU
            gpu_id: GPU设备ID
            debug_mode: 调试模式
        """
        # 设置GPU/CPU
        self._setup_hardware(use_gpu, gpu_id, debug_mode)

        # 加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 替换时间戳
        if self.config['experiment'].get('timestamp') == 'auto':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config['experiment']['timestamp'] = timestamp

        # 设置随机种子
        seed = self.config['dataset'].get('random_state', 42)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # 创建目录
        self._setup_directories()

        # 初始化日志
        self._setup_logging()

    def _setup_hardware(self, use_gpu, gpu_id, debug_mode):
        """设置硬件配置"""
        if use_gpu and gpu_id != '-1':
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

            # 设置GPU内存增长
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"使用GPU: {gpu_id}")
                except RuntimeError as e:
                    print(f"设置GPU时出错: {e}")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            print("使用CPU")

        # 设置调试模式
        if debug_mode:
            tf.config.run_functions_eagerly(True)
            tf.data.experimental.enable_debug_mode()
            print("调试模式已启用")

    def _setup_directories(self):
        """创建必要的目录结构"""
        # 实验名称
        exp_name = f"{self.config['experiment']['name']}_{self.config['experiment']['timestamp']}"

        # 创建主目录
        self.exp_dir = Path(self.config['logging']['experiment_dir']) / exp_name
        self.exp_log_dir = self.exp_dir / "logs"
        self.exp_model_dir = self.exp_dir / "models"
        self.exp_fig_dir = self.exp_dir / "figures"
        self.exp_data_dir = self.exp_dir / "data"

        # 创建所有目录
        for directory in [self.exp_dir, self.exp_log_dir, self.exp_model_dir,
                          self.exp_fig_dir, self.exp_data_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # 保存配置文件
        config_save_path = self.exp_dir / "config.yaml"
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)


        print(f"实验目录: {self.exp_dir}")

    def _setup_logging(self):
        """设置日志系统"""
        # 创建logger
        self.logger = logging.getLogger('FallDetection')
        self.logger.setLevel(logging.INFO)

        # 清除已有处理器
        self.logger.handlers.clear()

        # 创建文件处理器
        log_file = self.exp_log_dir / "training.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # 记录实验信息
        self.logger.info("=" * 60)
        self.logger.info(f"实验名称: {self.config['experiment']['name']}")
        self.logger.info(f"实验描述: {self.config['experiment'].get('description', '')}")
        self.logger.info(f"数据集: {self.config['dataset']['name']}")
        self.logger.info(f"模型: {self.config['model']['name']}")
        self.logger.info(f"时间戳: {self.config['experiment']['timestamp']}")
        self.logger.info("=" * 60)

    def load_data(self):
        """加载数据集"""
        self.logger.info("开始加载数据...")

        # 加载数据文件
        data_path = Path(self.config['dataset']['data_path'])
        label_path = Path(self.config['dataset']['label_path'])

        if not data_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path} 或 {label_path}")

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        with open(label_path, 'rb') as f:
            labels = pickle.load(f)

        # 转换为numpy数组
        X = np.array(data, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        # 如果需要reshape
        if self.config['dataset'].get('reshape', False):
            input_shape = self.config['dataset']['input_shape']
            X = X.reshape((len(X), input_shape[0], input_shape[1]))
            self.logger.info(f"数据reshape为: {X.shape}")

        self.logger.info(f"数据形状: {X.shape}")
        self.logger.info(f"标签形状: {y.shape}")
        self.logger.info(f"类别分布: {np.bincount(y.flatten())}")

        # 保存原始数据信息
        data_info = {
            'original_shape': data[0].shape if hasattr(data[0], 'shape') else str(type(data[0])),
            'num_samples': len(X),
            'input_shape': X.shape[1:],
            'num_classes': self.config['dataset']['num_classes'],
            'class_distribution': np.bincount(y.flatten()).tolist()
        }

        with open(self.exp_data_dir / "data_info.json", 'w') as f:
            json.dump(data_info, f, indent=2)

        return X, y

    def split_data(self, X, y):
        """划分数据集"""
        self.logger.info("开始划分数据集...")

        # 获取划分参数
        test_size = self.config['dataset']['test_size']
        val_size = self.config['dataset']['val_size']
        random_state = self.config['dataset']['random_state']
        stratify = self.config['dataset'].get('stratify', True)


        stratify_y = y if stratify else None
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y
        )


        val_ratio = val_size
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,  # 等分临时集
            random_state=random_state,
            stratify=y_temp if stratify else None
        )

        self.logger.info(f"训练集: {X_train.shape} ({(len(X_train) / len(X) * 100):.1f}%)")
        self.logger.info(f"验证集: {X_val.shape} ({(len(X_val) / len(X) * 100):.1f}%)")
        self.logger.info(f"测试集: {X_test.shape} ({(len(X_test) / len(X) * 100):.1f}%)")

        # 保存数据集信息
        data_split_info = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'train_percentage': (len(X_train) / len(X) * 100),
            'val_percentage': (len(X_val) / len(X) * 100),
            'test_percentage': (len(X_test) / len(X) * 100),
            'train_class_dist': np.bincount(y_train.flatten()).tolist(),
            'val_class_dist': np.bincount(y_val.flatten()).tolist(),
            'test_class_dist': np.bincount(y_test.flatten()).tolist()
        }

        with open(self.exp_data_dir / "data_split.json", 'w') as f:
            json.dump(data_split_info, f, indent=2)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self):
        """构建模型"""
        self.logger.info("开始构建模型...")

        # 创建模型
        model = create_edgefallnet(self.config)

        # 保存模型摘要
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        summary_str = "\n".join(model_summary)

        # 保存模型结构到文件
        summary_file = self.exp_log_dir / "model_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_str)

        # 保存模型结构图
        try:
            tf.keras.utils.plot_model(
                model,
                to_file=str(self.exp_fig_dir / "model_architecture.png"),
                show_shapes=True,
                show_layer_names=True,
                dpi=100
            )
        except Exception as e:
            self.logger.warning(f"无法保存模型结构图: {e}")

        self.logger.info("模型构建完成")
        self.logger.info(f"\n{summary_str}")

        return model

    def get_callbacks(self):
        """获取训练回调函数"""
        callbacks = []

        # 模型检查点
        if self.config['logging'].get('save_best_only', True):
            checkpoint_path = self.exp_model_dir / "best_model.h5"
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=self.config['logging'].get('monitor_metric', 'val_loss'),
                mode=self.config['logging'].get('monitor_mode', 'min'),
                save_best_only=True,
                verbose=self.config['logging'].get('verbose', 1)
            )
            callbacks.append(model_checkpoint)

        # 早停
        if self.config['training'].get('use_early_stopping', True):
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['training'].get('early_stopping_patience', 20),
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)

        # 学习率衰减
        if self.config['training'].get('use_reduce_lr', True):
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config['training'].get('reduce_lr_factor', 0.5),
                patience=self.config['training'].get('reduce_lr_patience', 10),
                min_lr=self.config['training'].get('min_lr', 1e-6),
                verbose=1
            )
            callbacks.append(reduce_lr)

        detailed_logger = DetailedTextLogger(
            filename=str(self.exp_log_dir / "training_details.txt")
        )
        callbacks.append(detailed_logger)



        if self.config['logging'].get('use_tensorboard', False):
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=str(self.exp_log_dir / "tensorboard"),
                histogram_freq=1
            )
            callbacks.append(tensorboard_callback)

        return callbacks

    def plot_training_history(self, history):
        """绘制训练历史图表"""
        if not history:
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 绘制训练和验证损失
        axes[0].plot(history.history.get('loss', []), 'g-', label='Training Loss', linewidth=2)
        axes[0].plot(history.history.get('val_loss', []), 'b-', label='Validation Loss', linewidth=2)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epochs', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # 绘制训练和验证准确率
        axes[1].plot(history.history.get('accuracy', []), 'g-', label='Training Accuracy', linewidth=2)
        axes[1].plot(history.history.get('val_accuracy', []), 'b-', label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epochs', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)



        # 添加模型信息文本
        model_info = (
            f"Model: {self.config['model']['name']}\n"
            f"Dataset: {self.config['dataset']['name']}\n"
            f"Best Val Loss: {min(history.history.get('val_loss', [0])):.4f}\n"
            f"Best Val Acc: {max(history.history.get('val_accuracy', [0])):.4f}\n"
            f"Final Train Loss: {history.history.get('loss', [0])[-1]:.4f}\n"
            f"Final Train Acc: {history.history.get('accuracy', [0])[-1]:.4f}"
        )


        # 保存图表
        loss_plot_path = self.exp_fig_dir / "training_history.png"
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"训练图表已保存到: {loss_plot_path}")

    def evaluate_model(self, model, X_test, y_test, X_val=None, y_val=None):
        """评估模型性能"""
        self.logger.info("开始评估模型...")

        results = {}

        # 在测试集上评估
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        results['test_loss'] = float(test_loss)
        results['test_accuracy'] = float(test_accuracy)

        # 在验证集上评估（如果有）
        if X_val is not None and y_val is not None:
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            results['val_loss'] = float(val_loss)
            results['val_accuracy'] = float(val_accuracy)

        # 预测
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = cm.tolist()

        # 计算性能指标
        TN, FP, FN, TP = cm.ravel()
        results['TP'] = int(TP)
        results['FN'] = int(FN)
        results['FP'] = int(FP)
        results['TN'] = int(TN)

        # 计算各项指标
        results['accuracy'] = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        results['sensitivity'] = TP / (TP + FN) if (TP + FN) > 0 else 0
        results['specificity'] = TN / (TN + FP) if (TN + FP) > 0 else 0
        results['precision'] = TP / (TP + FP) if (TP + FP) > 0 else 0
        results['f1_score'] = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

        # 几何均值
        results['gmean'] = np.sqrt(results['sensitivity'] * results['specificity'])

        # 分类报告
        report = classification_report(y_test, y_pred, output_dict=True)
        results['classification_report'] = report

        # 保存评估结果
        self._save_evaluation_results(results)

        return results

    def _save_evaluation_results(self, results):
        """保存评估结果到文件"""
        eval_file = self.exp_log_dir / "evaluation_results.txt"

        with open(eval_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("模型评估结果\n")
            f.write("=" * 60 + "\n\n")

            f.write("性能指标:\n")
            f.write("-" * 40 + "\n")
            f.write(f"测试集损失: {results.get('test_loss', 0):.4f}\n")
            f.write(f"测试集准确率: {results.get('test_accuracy', 0):.4f}\n")

            if 'val_loss' in results:
                f.write(f"验证集损失: {results.get('val_loss', 0):.4f}\n")
                f.write(f"验证集准确率: {results.get('val_accuracy', 0):.4f}\n")

            f.write(f"准确率: {results.get('accuracy', 0):.4f}\n")
            f.write(f"灵敏度(召回率): {results.get('sensitivity', 0):.4f}\n")
            f.write(f"特异度: {results.get('specificity', 0):.4f}\n")
            f.write(f"精确率: {results.get('precision', 0):.4f}\n")
            f.write(f"F1分数: {results.get('f1_score', 0):.4f}\n")
            f.write(f"几何均值: {results.get('gmean', 0):.4f}\n\n")

            f.write("混淆矩阵:\n")
            f.write("-" * 40 + "\n")
            f.write(f"TP: {results.get('TP', 0)}, FN: {results.get('FN', 0)}\n")
            f.write(f"FP: {results.get('FP', 0)}, TN: {results.get('TN', 0)}\n\n")

            cm = results.get('confusion_matrix', [[0, 0], [0, 0]])
            f.write("混淆矩阵详细:\n")
            f.write(f"[[TN={cm[0][0]}, FP={cm[0][1]}],\n")
            f.write(f" [FN={cm[1][0]}, TP={cm[1][1]}]]\n\n")

            f.write("分类报告:\n")
            f.write("-" * 40 + "\n")
            report = results.get('classification_report', {})
            for key, value in report.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"  {subkey}: {subvalue:.4f}\n")
                else:
                    f.write(f"{key}: {value:.4f}\n")


        # 记录到日志
        self.logger.info("=" * 60)
        self.logger.info("模型评估结果")
        self.logger.info("=" * 60)
        self.logger.info(f"测试集损失: {results.get('test_loss', 0):.4f}")
        self.logger.info(f"测试集准确率: {results.get('test_accuracy', 0):.4f}")
        self.logger.info(f"准确率: {results.get('accuracy', 0):.4f}")
        self.logger.info(f"灵敏度: {results.get('sensitivity', 0):.4f}")
        self.logger.info(f"特异度: {results.get('specificity', 0):.4f}")
        self.logger.info(f"几何均值: {results.get('gmean', 0):.4f}")

    def save_model(self, model, format='h5'):
        """保存模型"""
        model_name = self.config['model']['name'].replace(" ", "_")
        model_save_dir = Path(self.config['logging']['model_save_dir'])
        model_save_dir.mkdir(parents=True, exist_ok=True)

        if format == 'h5':
            # 保存为HDF5格式
            model_path = self.exp_model_dir / f"{model_name}.h5"
            model.save(model_path)

            # 同时保存到模型目录
            final_model_path = model_save_dir / f"{model_name}_{self.config['experiment']['timestamp']}.h5"
            model.save(final_model_path)

        elif format == 'tf':
            # 保存为TensorFlow SavedModel格式
            model_path = self.exp_model_dir / f"{model_name}_saved_model"
            model.save(str(model_path))

        elif format == 'tflite':
            # 保存为TFLite格式
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()

            tflite_path = self.exp_model_dir / f"{model_name}.tflite"
            tflite_path.write_bytes(tflite_model)

            # 同时保存到模型目录
            final_tflite_path = model_save_dir / f"{model_name}_{self.config['experiment']['timestamp']}.tflite"
            final_tflite_path.write_bytes(tflite_model)

        self.logger.info(f"模型已保存到: {self.exp_model_dir}")

    def train(self):
        """执行训练流程"""
        try:
            # 1. 加载数据
            X, y = self.load_data()

            # 2. 划分数据集
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

            # 3. 构建模型
            model = self.build_model()

            # 4. 获取回调函数
            callbacks = self.get_callbacks()

            # 5. 训练模型
            self.logger.info("开始训练模型...")
            self.logger.info(f"训练样本数: {len(X_train)}")
            self.logger.info(f"验证样本数: {len(X_val)}")
            self.logger.info(f"测试样本数: {len(X_test)}")

            history = model.fit(
                X_train, y_train,
                batch_size=self.config['training']['batch_size'],
                epochs=self.config['training']['epochs'],
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=self.config['logging'].get('verbose', 1)
            )

            # 6. 绘制训练历史
            self.plot_training_history(history)

            # 7. 保存最终模型
            save_format = self.config['logging'].get('save_format', 'h5')
            self.save_model(model, format=save_format)

            # 8. 评估模型
            eval_results = self.evaluate_model(model, X_test, y_test, X_val, y_val)

            # 9. 保存训练历史
            #self._save_training_history(history, eval_results)

            # 10. 记录训练完成
            self.logger.info("=" * 60)
            self.logger.info("训练完成!")
            self.logger.info(f"实验结果保存在: {self.exp_dir}")
            self.logger.info("=" * 60)

            return {
                'history': history.history,
                'evaluation': eval_results,
                'model': model,
                'experiment_dir': str(self.exp_dir)
            }

        except Exception as e:
            self.logger.error(f"训练过程中发生错误: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise



def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='跌倒检测模型训练脚本')
    parser.add_argument('--config', type=str, default=CONFIG_PATH,
                        help=f'配置文件路径 (默认: {CONFIG_PATH})')
    parser.add_argument('--gpu', type=str, default=GPU_ID,
                        help=f'GPU设备ID (默认: {GPU_ID}, 使用-1表示CPU)')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')

    args = parser.parse_args()

    # 创建训练器并开始训练
    trainer = FallDetectionTrainer(args.config, use_gpu=(args.gpu != '-1'),
                                   gpu_id=args.gpu, debug_mode=args.debug)
    results = trainer.train()

    print(f"\n{'=' * 60}")
    print(f"训练完成！")
    print(f"实验结果保存在: {results['experiment_dir']}")
    print(f"{'=' * 60}")

    # 打印关键结果
    eval_results = results['evaluation']
    print(f"\n关键评估指标:")
    print(f"测试准确率: {eval_results.get('test_accuracy', 0):.4f}")
    print(f"灵敏度: {eval_results.get('sensitivity', 0):.4f}")
    print(f"特异度: {eval_results.get('specificity', 0):.4f}")
    print(f"几何均值: {eval_results.get('gmean', 0):.4f}")


if __name__ == "__main__":
    print("开始跌倒检测模型训练...")
    print(f"默认配置文件: {CONFIG_PATH}")
    print(f"默认使用GPU: {USE_GPU} (ID: {GPU_ID})")
    print(f"调试模式: {DEBUG_MODE}")
    print("-" * 50)
    main()