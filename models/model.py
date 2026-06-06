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
