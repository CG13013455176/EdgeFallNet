# EdgeFallNet 核心代码提炼与 Gemini 3 Pro 附录生成指令

## 一、核心代码文件清单（论文附录用）

本项目共需提取 **4 个核心文件** 作为论文附录代码。按重要性排序：

| 序号 | 文件路径 | 核心功能 | 论文中对应章节 |
|------|----------|----------|----------------|
| **1** | `models/model.py` | EdgeFallNet 模型架构定义（Fire模块 + 完整网络） | 模型设计/方法章节 |
| **2** | `train.py` | 训练流程主脚本（数据加载、训练、评估全流程） | 实验设置章节 |
| **3** | `configs/fallld.yaml` | FallAllD 数据集超参数配置 | 实验设置章节 |
| **4** | `configs/umafall.yaml` | UMAFALL 数据集超参数配置 | 实验设置章节 |

> **可选补充**: 如果论文附录篇幅允许，可额外包含：
> - `datasets/FallAllDDataset.py` — FallAllD 数据预处理逻辑
> - `datasets/UMAFALLDataset.py` — UMAFALL 数据预处理逻辑

---

## 二、各核心文件完整源码

### 文件 1：models/model.py（模型架构定义 —— 最核心）

```python
"""
EdgeFallNet 模型定义模块

基于 SqueezeNet 的轻量化 1D CNN，用于跌倒检测。
采用 SeparableConv1D 减少参数量，适合边缘设备部署。
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def fire_module(x, squeeze_filters, expand1x1_filters, expand3x3_filters, use_separable=True):
    """
    Fire 模块（SqueezeNet 核心组件）

    结构：Squeeze层(1x1卷积) -> Expand层(1x1卷积 + 3x3深度可分离卷积) -> 拼接

    Args:
        x: 输入张量
        squeeze_filters: Squeeze层的滤波器数量
        expand1x1_filters: Expand 1x1分支的滤波器数量
        expand3x3_filters: Expand 3x3分支的滤波器数量
        use_separable: 是否在expand3x3中使用SeparableConv1D

    Returns:
        拼接后的输出张量
    """
    # Squeeze 层：使用 1x1 卷积压缩通道数
    squeeze = layers.Conv1D(
        filters=squeeze_filters,
        kernel_size=1,
        padding='same',
        activation='relu',
        name='squeeze'
    )(x)

    # Expand 1x1 分支：使用 1x1 卷积扩展通道
    expand1x1 = layers.Conv1D(
        filters=expand1x1_filters,
        kernel_size=1,
        padding='same',
        activation='relu',
        name='expand1x1'
    )(squeeze)

    # Expand 3x3 分支：使用深度可分离卷积捕获局部特征
    if use_separable:
        expand3x3 = layers.SeparableConv1D(
            filters=expand3x3_filters,
            kernel_size=3,
            padding='same',
            activation='relu',
            name='expand3x3'
        )(squeeze)
    else:
        expand3x3 = layers.Conv1D(
            filters=expand3x3_filters,
            kernel_size=3,
            padding='same',
            activation='relu',
            name='expand3x3'
        )(squeeze)

    # 将两个分支的输出沿通道维度拼接
    output = layers.Concatenate(name='merge')([expand1x1, expand3x3])

    return output


def build_edgefallnet(config):
    """
    构建 EdgeFallNet 模型结构

    架构：
        Input -> 初始可分离卷积 -> MaxPooling ->
        Fire Module 1 -> MaxPooling ->
        Fire Module 2 -> MaxPooling ->
        Flatten -> Dense -> Dropout -> Output

    Args:
        config: 配置字典，包含 model 和 dataset 子配置

    Returns:
        编译前的 Keras Model 实例
    """
    model_cfg = config['model']
    input_shape = tuple(config['dataset']['input_shape'])

    initial_conv_filters = model_cfg.get('initial_conv_filters', 16)
    fire1_squeeze = model_cfg.get('fire1_squeeze', 8)
    fire1_expand1x1 = model_cfg.get('fire1_expand1x1', 4)
    fire1_expand3x3 = model_cfg.get('fire1_expand3x3', 6)
    fire2_squeeze = model_cfg.get('fire2_squeeze', 8)
    fire2_expand1x1 = model_cfg.get('fire2_expand1x1', 4)
    fire2_expand3x3 = model_cfg.get('fire2_expand3x3', 6)
    dense_units = model_cfg.get('dense_units', 8)
    dropout_rate = model_cfg.get('dropout_rate', 0.1)
    use_separable = model_cfg.get('use_separable', True)
    num_classes = model_cfg.get('num_classes', 1)

    # 输入层
    inputs = layers.Input(shape=input_shape, name='input_layer')

    # 初始卷积层：使用 SeparableConv1D 提取初始特征
    x = layers.SeparableConv1D(
        filters=initial_conv_filters,
        kernel_size=5,
        padding='same',
        activation='relu',
        name='initial_conv'
    )(inputs)

    # 第一个池化层
    x = layers.MaxPooling1D(pool_size=5, name='pool1')(x)

    # Fire Module 1
    x = fire_module(x, fire1_squeeze, fire1_expand1x1, fire1_expand3x3,
                    use_separable=use_separable)
    # 重命名内部层以匹配日志中的名称
    x._name = 'squeeze_fire1_fire'
    for layer in x.layers:
        if layer.name == 'squeeze':
            layer._name = 'squeeze_fire1'
        elif layer.name == 'expand1x1':
            layer._name = 'expand1x1_fire1'
        elif layer.name == 'expand3x3':
            layer._name = 'expand3x3_fire1'
        elif layer.name == 'merge':
            layer._name = 'merge_fire1'

    # 第二个池化层
    x = layers.MaxPooling1D(pool_size=5, name='pool2')(x)

    # Fire Module 2
    x = fire_module(x, fire2_squeeze, fire2_expand1x1, fire2_expand3x3,
                    use_separable=use_separable)
    for layer in x.layers:
        if layer.name == 'squeeze':
            layer._name = 'squeeze_fire2'
        elif layer.name == 'expand1x1':
            layer._name = 'expand1x1_fire2'
        elif layer.name == 'expand3x3':
            layer._name = 'expand3x3_fire2'
        elif layer.name == 'merge':
            layer._name = 'merge_fire2'

    # 第三个池化层
    x = layers.MaxPooling1D(pool_size=5, name='pool3')(x)

    # 展平层
    x = layers.Flatten(name='flatten')(x)

    # 全连接层
    x = layers.Dense(dense_units, activation='relu', name='dense1')(x)

    # Dropout 正则化
    x = layers.Dropout(dropout_rate, name='dropout')(x)

    # 输出层（二分类使用 sigmoid 激活）
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)

    # 创建模型
    model = models.Model(inputs=inputs, outputs=outputs, name='EdgeFallNet')

    return model


def compile_model(model, config):
    """
    编译 EdgeFallNet 模型

    配置优化器、损失函数和评估指标

    Args:
        model: 未编译的 Keras Model 实例
        config: 配置字典，包含 training 子配置
    """
    training_cfg = config['training']
    learning_rate = training_cfg.get('learning_rate', 0.001)
    optimizer_name = training_cfg.get('optimizer', 'adam').lower()
    loss = training_cfg.get('loss', 'binary_crossentropy')
    metrics = training_cfg.get('metrics', ['accuracy'])

    # 根据配置选择优化器
    if optimizer_name == 'adamax':
        optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )


def create_edgefallnet(config):
    """
    完整创建并编译 EdgeFallNet 模型

    依次调用 build_edgefallnet 和 compile_model，
    返回可直接用于训练的模型实例。

    Args:
        config: 配置字典，包含 model、dataset 和 training 子配置

    Returns:
        已编译的 Keras Model 实例
    """
    model = build_edgefallnet(config)
    compile_model(model, config)
    return model
```

### 文件 2：train.py（训练主脚本）

（代码较长，约650行，见项目源文件 train.py，核心类为 `FallDetectionTrainer`）

**关键函数说明**：
- `DetailedTextLogger` — 训练日志回调
- `FallDetectionTrainer.__init__()` — 初始化（GPU设置、配置加载、目录创建）
- `FallDetectionTrainer.load_data()` — 加载 .pkl 数据文件
- `FallDetectionTrainer.split_data()` — 划分训练/验证/测试集（7:1.5:1.5）
- `FallDetectionTrainer.build_model()` — 调用 create_edgefallnet 构建模型
- `FallDetectionTrainer.train()` — 完整训练流程编排
- `FallDetectionTrainer.evaluate_model()` — 计算混淆矩阵、灵敏度、特异度等指标

### 文件 3 & 4：配置文件

**configs/fallld.yaml**：
```yaml
experiment:
  name: "FallAllD_Experiment"
  timestamp: "auto"

dataset:
  name: "FallAllD"
  input_shape: [4760, 6]
  num_classes: 1
  test_size: 0.3
  val_size: 0.5
  random_state: 42
  stratify: true
  reshape: false

model:
  name: "EdgeFallNet_FallAllD"
  initial_conv_filters: 16
  fire1_squeeze: 8
  fire1_expand1x1: 4
  fire1_expand3x3: 6
  fire2_squeeze: 8
  fire2_expand1x1: 4
  fire2_expand3x3: 6
  dense_units: 8
  dropout_rate: 0.1
  use_separable: true
  num_classes: 1

training:
  batch_size: 32
  epochs: 120
  learning_rate: 0.05
  optimizer: "adamax"
  loss: "binary_crossentropy"
  metrics: ["accuracy"]
  use_early_stopping: true
  early_stopping_patience: 20
  use_reduce_lr: true
  reduce_lr_factor: 0.5

logging:
  save_format: "h5"
  save_best_only: true
```

**configs/umafall.yaml**：
```yaml
experiment:
  name: "UMAFALL_Experiment"
  timestamp: "auto"

dataset:
  name: "UMAFALL"
  input_shape: [300, 6]
  num_classes: 1
  test_size: 0.3
  val_size: 0.5
  random_state: 42
  reshape: true

model:
  name: "EdgeFallNet_UMAFALL"
  initial_conv_filters: 8
  fire1_squeeze: 4
  fire1_expand1x1: 2
  fire1_expand3x3: 6
  fire2_squeeze: 4
  fire2_expand1x1: 2
  fire2_expand3x3: 6
  dense_units: 8
  dropout_rate: 0.1
  use_separable: true
  num_classes: 1

training:
  batch_size: 32
  epochs: 120
  learning_rate: 0.01
  optimizer: "adamax"
  loss: "binary_crossentropy"
  metrics: ["accuracy"]
  use_early_stopping: true
  early_stopping_patience: 20
  use_reduce_lr: true
  reduce_lr_factor: 0.5

logging:
  save_format: "h5"
  save_best_only: true
```

---

## 三、Gemini 3 Pro 精准指令（复制以下内容发送给 Gemini 3 Pro）

请将以下指令**完整复制**后发送给 Gemini 3 Pro，即可获得格式化的论文附录代码：

---

### 指令 A：模型架构代码格式化（附录核心）

```
你是一位学术论文排版专家，精通计算机科学领域的论文写作规范。我需要将以下深度学习模型代码整理为"附录A：EdgeFallNet模型核心代码"，要求符合IEEE/ACM期刊或国内核心期刊（如《计算机学报》《软件学报》）的附录代码排版规范。

【任务要求】
1. 请将以下Python代码重新排版为论文附录格式：
   - 每个函数前添加中文注释说明其功能和在算法中的角色
   - 关键行添加行内注释解释算法含义（如"// Squeeze层：1×1卷积压缩通道数"）
   - 使用等宽字体格式的代码块
   - 为每个代码块添加"代码清单X.X"编号和标题
   - 在代码前后添加简短的文字说明该段代码的作用

2. 排版风格参考：
   - 代码清单采用"代码清单 A.1 函数名"格式
   - 每段代码不超过50行，过长函数需要合理分段展示
   - 保留所有原始代码逻辑不变，仅调整注释和排版

3. 需要处理的代码如下（共4个函数，来自models/model.py）：

【代码1 - Fire模块（SqueezeNet核心组件）】
def fire_module(x, squeeze_filters, expand1x1_filters, expand3x3_filters, use_separable=True):
    """Fire模块：Squeeze(1x1卷积) -> Expand(1x1 + 可分离3x3卷积) -> Concatenate"""
    squeeze = layers.Conv1D(filters=squeeze_filters, kernel_size=1, padding='same', activation='relu', name='squeeze')(x)
    expand1x1 = layers.Conv1D(filters=expand1x1_filters, kernel_size=1, padding='same', activation='relu', name='expand1x1')(squeeze)
    if use_separable:
        expand3x3 = layers.SeparableConv1D(filters=expand3x3_filters, kernel_size=3, padding='same', activation='relu', name='expand3x3')(squeeze)
    else:
        expand3x3 = layers.Conv1D(filters=expand3x3_filters, kernel_size=3, padding='same', activation='relu', name='expand3x3')(squeeze)
    output = layers.Concatenate(name='merge')([expand1x1, expand3x3])
    return output

【代码2 - EdgeFallNet网络构建】
def build_edgefallnet(config):
    """构建EdgeFallNet: Input->SepConv1D(k=5)->MaxPool->FireModule1->MaxPool->FireModule2->MaxPool->Flatten->Dense->Dropout->Output"""
    model_cfg = config['model']; input_shape = tuple(config['dataset']['input_shape'])
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = layers.SeparableConv1D(filters=model_cfg['initial_conv_filters'], kernel_size=5, padding='same', activation='relu', name='initial_conv')(inputs)
    x = layers.MaxPooling1D(pool_size=5, name='pool1')(x)
    x = fire_module(x, model_cfg['fire1_squeeze'], model_cfg['fire1_expand1x1'], model_cfg['fire1_expand3x3'], use_separable=model_cfg['use_separable'])
    x = layers.MaxPooling1D(pool_size=5, name='pool2')(x)
    x = fire_module(x, model_cfg['fire2_squeeze'], model_cfg['fire2_expand1x1'], model_cfg['fire2_expand3x3'], use_separable=model_cfg['use_separable'])
    x = layers.MaxPooling1D(pool_size=5, name='pool3')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(model_cfg['dense_units'], activation='relu', name='dense1')(x)
    x = layers.Dropout(model_cfg['dropout_rate'], name='dropout')(x)
    outputs = layers.Dense(model_cfg['num_classes'], activation='sigmoid', name='output')(x)
    return models.Model(inputs=inputs, outputs=outputs, name='EdgeFallNet')

【代码3 - 模型编译】
def compile_model(model, config):
    """配置Adamax优化器、二元交叉熵损失"""
    cfg = config['training']; lr = cfg.get('learning_rate', 0.001)
    opt_name = cfg.get('optimizer', 'adam').lower()
    optimizers = {'adamax': lambda: tf.keras.optimizers.Adamax(lr), 'adam': lambda: tf.keras.optimizers.Adam(lr), 'sgd': lambda: tf.keras.optimizers.SGD(lr), 'rmsprop': lambda: tf.keras.optimizers.RMSprop(lr)}
    model.compile(optimizer=optimizers.get(opt_name, lambda: tf.keras.optimizers.Adam(lr))(), loss=cfg.get('loss','binary_crossentropy'), metrics=cfg.get('metrics',['accuracy']))

【代码4 - 模型工厂函数】
def create_edgefallnet(config):
    """组合build+compile，返回可训练模型"""
    model = build_edgefallnet(config); compile_model(model, config); return model

【附加信息】
- 本模型总参数量仅 3,731 个
- 输入维度：(时间步长, 6) — 6表示三轴加速度计 + 三轴陀螺仪
- 输出：sigmoid激活，二分类（ADL=0 / FALL=1）
- 基于TensorFlow/Keras框架实现
- 论文题目：《EdgeFallNet：面向智能手环的实时跌倒检测轻量化模型设计与实现》

请输出完整的、可直接复制到Word论文文档中的附录内容。
```

---

### 指令 B：训练流程代码格式化（附录补充）

```
你是一位学术论文排版专家。我需要将以下深度学习训练流程代码整理为论文附录格式。

【背景】
这是论文《EdgeFallNet：面向智能手环的实时跌倒检测轻量化模型设计与实现》的训练脚本核心代码。
请将其整理为"附录B：训练与评估流程核心代码"。

【任务要求】
1. 提取 FallDetectionTrainer 类的核心方法，精简非关键代码（如日志格式化细节），保留算法逻辑
2. 重点展示以下方法的骨架：
   - __init__: 环境初始化（GPU设置、随机种子）
   - load_data: 数据加载（pickle读取、numpy转换）
   - split_data: 数据划分（train_test_split分层采样，比例70%:15%:15%）
   - build_model: 模型构建调用
   - get_callbacks: 回调函数（ModelCheckpoint + EarlyStopping + ReduceLROnPlateau）
   - evaluate_model: 评估指标计算（混淆矩阵、灵敏度、特异度、G-Mean）
   - train: 主流程编排
3. 添加"代码清单B.X"编号
4. 关键超参数用注释标注具体值（如 batch_size=32, epochs=120, lr=0.05）

【核心代码片段】

class FallDetectionTrainer:
    def __init__(self, config_path, use_gpu=True, gpu_id="0", debug_mode=False):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id if use_gpu else ''
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        with open(config_path, 'r') as f: self.config = yaml.safe_load(f)
        np.random.seed(42); tf.random.set_seed(42)  # 固定随机种子

    def load_data(self):
        with open(self.config['dataset']['data_path'], 'rb') as f:
            X = np.array(pickle.load(f), dtype=np.float32)
        with open(self.config['dataset']['label_path'], 'rb') as f:
            y = np.array(pickle.load(f), dtype=np.int32)
        return X, y

    def split_data(self, X, y):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_callbacks(self):
        callbacks = [
            ModelCheckpoint(filepath="best_model.h5", monitor='val_loss', mode='min', save_best_only=True),
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
            DetailedTextLogger(filename="training_details.txt")
        ]
        return callbacks

    def evaluate_model(self, model, X_test, y_test):
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()
        results = {
            'accuracy': (TP+TN)/(TP+TN+FP+FN), 'sensitivity': TP/(TP+FN),
            'specificity': TN/(TN+FP), 'precision': TP/(TP+FP),
            'f1_score': 2*TP/(2*TP+FP+FN), 'gmean': np.sqrt(results['sensitivity']*results['specificity'])
        }
        return results

    def train(self):
        X, y = self.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        model = create_edgefallnet(self.config)
        history = model.fit(X_train, y_train, batch_size=32, epochs=120,
                           validation_data=(X_val, y_val), callbacks=self.get_callbacks())
        eval_results = self.evaluate_model(model, X_test, y_test)
        return {'history': history.history, 'evaluation': eval_results}

【实验结果参考】
- FallAllD数据集: Acc=91.01%, Sensitivity=93.75%, Specificity=90.45%, G-Mean=92.08%
- UMAFALL数据集: Acc=98.17%, Sensitivity=96.77%, Specificity=98.72%, G-Mean=97.74%

请输出可直接用于Word论文附录的格式化内容。
```

---

### 指令 C：配置文件与数据预处理格式化（附录可选）

```
你是一位学术论文排版专家。请将以下内容整理为论文附录C格式。

【内容1 - 超参数配置表】
将两个YAML配置文件转换为论文表格格式，表格列包括：参数类别、参数名、FallAllD值、UMAFALL值、说明。

【内容2 - 数据预处理伪代码】
将以下数据处理逻辑转为算法风格的伪代码（Algorithm X 格式）：

FallAllD预处理流程:
1. 加载 ACC_ADL.pkl, ACC_FALL.pkl (加速度原始数据)
2. 加载 GCC_ADL.pkl, GCC_FALL.pkl (陀螺仪原始数据)
3. 加速度归一化: value * 0.000244
4. 陀螺仪归一化: value * 0.07 * 8 / 2000
5. 特征融合: acc(3轴) + gyr(3轴) → 6维特征向量
6. 交替采样平衡数据集: ADL样本与FALL样本交替排列
7. 标签编码: ADL→0, FALL→1
8. 保存为 FallAllD_Data.pkl 和 FallAllD_lable.pkl

UMAFALL预处理流程:
1. 加载 ADL.txt, FALL.txt (CSV格式, 401×3)
2. 归一化: value / 8
3. 交替采样 + 标签编码(ADL→0, FALL→1)
4. 保存为 UMAFALL_Data.pkl 和 UMAFALL_lable.pkl

请输出标准论文附录格式的内容。
```

---

## 四、使用建议

### 推荐的附录组织方式

```
附录A  EdgeFallNet 核心模型代码
    A.1  Fire 模块实现（fire_module 函数）
    A.2  EdgeFallNet 网络构建（build_edgefallnet 函数）
    A.3  模型编译配置（compile_model 函数）
    A.4  模型工厂接口（create_edgefallnet 函数）

附录B  训练与评估流程代码
    B.1  FallDetectionTrainer 类初始化与环境配置
    B.2  数据加载与划分策略
    B.3  回调函数与训练策略
    B.4  模型评估与性能指标计算

附录C  （可选）超参数配置与数据预处理
    C.1  超参数配置汇总表
    C.2  数据预处理算法伪代码
```

### 发送给 Gemini 3 Pro 的操作步骤

1. 打开 [Google Gemini](https://gemini.google.com)，选择 **Gemini 1.5 Pro** 或更新版本
2. **完整复制**上述「指令 A」（模型架构代码格式化），粘贴并发送
3. 等待生成完成后，再发送「指令 B」（训练流程代码格式化）
4. 如需配置表和预处理伪代码，继续发送「指令 C」
5. 将生成的结果复制到 Word 论文文档的附录部分
6. 在 Word 中统一调整代码字体为 **Consolas** 或 **Courier New**，字号 **小五号（9pt）**

### 注意事项

- 发送指令时**不要省略任何内容**，包括方括号内的附加信息
- 如生成的代码格式有误，可追加提示："请调整为更紧凑的格式，适合双栏排版"
- 建议先生成模型代码（附录A），确认满意后再生成训练代码（附录B）
