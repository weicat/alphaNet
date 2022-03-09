"""时间序列计算层、神经网络模型定义.
复现华泰金工 alpha net V2、V3 版本.
V2:
```
input: (batch_size, history time steps, features)
                 stride = 5
input -> expand features -> BN -> LSTM -> BN -> Dense(linear)
```
V3:
```
input: (batch_size, history time steps, features)
                stride = 5
        +-> expand features -> BN -> GRU -> BN -+
input --|       stride = 10                     |- concat -> Dense(linear)
        +-> expand features -> BN -> GRU -> BN -+
```
(BN: batch normalization)
该module定义了计算不同时间序列特征的层，工程上使用tensorflow
进行高度向量化的计算，训练时较高效。
"""
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod

import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _LowerNoDiagonalMask(nn.Module):
    """获取不含对角元素的矩阵下三角mask.
    Notes:
        Provide a mask giving the lower triangular of a matrix
        without diagonal elements.
    """

    def __init__(self):
        super(_LowerNoDiagonalMask, self).__init__()

    def __call__(self, shape, **kwargs):
        """计算逻辑."""
        ones = torch.ones(shape)
        mask = torch.tril(ones, -1) > 0
        # lower triangle removing the diagonal elements
        # mask = torch.(mask_lower - mask_diag, dtype=torch.bool)
        return mask.to(device)


def __get_dimensions__(input_shape, stride):
    """计算相关维度长度.
    Notes:
        output_length = 原来的时间长度 / stride的长度
    Args:
        input_shape: pass the inputs of layer to the function
        stride (int): the stride of the custom layer
    Returns:
        (features, output_length)
    Raises:
        ValueError: 如果历史长度不是stride的整数倍
    """
    if type(stride) is not int or stride <= 1:
        raise ValueError("Illegal Argument: stride should be an integer "
                         "greater than 1")
    time_steps = input_shape[1]
    features = input_shape[2]
    output_length = time_steps // stride

    if time_steps % stride != 0:
        raise ValueError("Error, time_steps 应该是 stride的整数倍")

    return features, output_length


class _StrideLayer(nn.Module, _ABC):
    """计算每个stride的统计值的基类."""

    def __init__(self, stride=10, **kwargs):
        """计算每个stride的统计值的基类.
        Args:
            stride (int): time steps需要是stride的整数倍
        """
        if stride <= 1:
            raise ValueError("Illegal Argument: stride should be "
                             "greater than 1")
        super(_StrideLayer, self).__init__(**kwargs)
        self.stride = stride
        self.out_shape = None
        self.intermediate_shape = None

    def build(self, input_shape):
        """构建该层，计算维度信息."""
        (features, output_length) = __get_dimensions__(input_shape, self.stride)
        self.out_shape = [-1, output_length, features]
        self.intermediate_shape = [-1, self.stride, features]

    def get_config(self):
        """获取参数，保存模型需要的函数."""
        config = super().state_dict().copy()
        config.update({'stride': self.stride})
        return config


class Std(_StrideLayer):
    """计算每个序列各stride的标准差.
    Notes:
        计算每个feature各个stride的standard deviation
    """

    def forward(self, inputs, *args, **kwargs):
        """函数主逻辑实现部分.
        Args:
            inputs (tensor): 输入dimension为(batch_size, time_steps, features)
        Returns:
            dimension 为(batch_size, time_steps / stride, features)
        """
        if self.intermediate_shape is None:
            self.build(inputs.shape)
        strides = torch.reshape(inputs, self.intermediate_shape)

        # compute standard deviations for each stride
        std = torch.std(strides, dim=-2)
        return torch.reshape(std, self.out_shape)


class ZScore(_StrideLayer):
    """计算每个序列各stride的均值除以其标准差.
    Notes:
        并非严格意义上的z-score,
        计算公式为每个feature各个stride的mean除以各自的standard deviation
    """

    def forward(self, inputs, *args, **kwargs):
        """函数主逻辑实现部分.
        Args:
            inputs (tensor): 输入dimension为(batch_size, time_steps, features)
        Returns:
            dimension 为(batch_size, time_steps / stride, features)
        """
        if self.intermediate_shape is None:
            self.build(inputs.shape)
        strides = torch.reshape(inputs, self.intermediate_shape)

        # compute standard deviations for each stride
        std = torch.std(strides, dim=-2)

        # compute means for each stride
        means = torch.mean(strides, dim=-2)

        # divide means by standard deviations for each stride
        z_score = torch.nan_to_num(torch.div(means, std), 0, 0, 0)
        return torch.reshape(z_score, self.out_shape)


class LinearDecay(_StrideLayer):
    """计算每个序列各stride的线性衰减加权平均.
    Notes:
        以线性衰减为权重，计算每个feature各个stride的均值：
        如stride为10，则某feature该stride的权重为(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    """

    def forward(self, inputs, *args, **kwargs):
        """函数主逻辑实现部分.
        Args:
            inputs (tensor): 输入dimension为(batch_size, time_steps, features)
        Returns:
            dimension 为(batch_size, time_steps / stride, features)
        """
        # get linear decay kernel
        if self.intermediate_shape is None:
            self.build(inputs.shape)
        single_kernel = torch.linspace(1.0, self.stride, self.stride).to(device)
        kernel = single_kernel.repeat(self.intermediate_shape[2])
        kernel = kernel / torch.sum(single_kernel)

        # reshape tensors into:
        # (bash_size * (time_steps / stride), stride, features)
        kernel = torch.reshape(kernel, self.intermediate_shape[1:])
        inputs = torch.reshape(inputs, self.intermediate_shape)

        # broadcasting kernel to inputs batch dimension
        linear_decay = torch.sum(kernel * inputs, dim=1)
        linear_decay = torch.reshape(linear_decay, self.out_shape)
        return linear_decay


class Return(nn.Module):
    """计算每个序列各stride的回报率.
    Notes:
        计算公式为每个stride最后一个数除以第一个数再减去一
    """

    def __init__(self, stride=10, **kwargs):
        """回报率.
        Args:
            stride (int): time steps需要是stride的整数倍
        """
        if stride <= 1:
            raise ValueError("Illegal Argument: stride should be "
                             "greater than 1")
        super(Return, self).__init__(**kwargs)
        self.stride = stride

    def build(self, input_shape):
        """构建该层，计算维度信息."""
        time_steps = input_shape[1]
        if time_steps % self.stride != 0:
            raise ValueError("Error, time_steps 应该是 stride的整数倍")

    def forward(self, inputs, *args, **kwargs):
        """函数主逻辑实现部分.
        Args:
            inputs (tensor): 输入dimension为(batch_size, time_steps, features)
        Returns:
            dimension 为(batch_size, time_steps / stride, features)
        """
        # get the endings of each strides as numerators
        numerators = inputs[:, (self.stride - 1)::self.stride, :]

        # get the beginnings of each strides as denominators
        denominators = inputs[:, 0::self.stride, :]

        return torch.nan_to_num(torch.divide(numerators, denominators), 0, 0, 0) - 1.0

    def get_config(self):
        """获取参数，保存模型需要的函数."""
        config = super().state_dict().copy()
        config.update({'stride': self.stride})
        return config


class _OuterProductLayer(nn.Module, _ABC):

    def __init__(self, stride=10, **kwargs):
        """外乘类的扩张层.
        Args:
            stride (int): time steps需要是stride的整数倍
        """
        if stride <= 1:
            raise ValueError("Illegal Argument: stride should be "
                             "greater than 1")
        super(_OuterProductLayer, self).__init__(**kwargs)
        self.stride = stride
        self.intermediate_shape = None
        self.out_shape = None
        self.lower_mask = None

    def build(self, input_shape):
        """构建该层，计算维度信息."""
        (features,
         output_length) = __get_dimensions__(input_shape, self.stride)
        self.intermediate_shape = (-1, self.stride, features)
        output_features = int(features * (features - 1) / 2)
        self.out_shape = (-1, output_length, output_features)
        self.lower_mask = _LowerNoDiagonalMask()((features, features))

    def get_config(self):
        """获取参数，保存模型需要的函数."""
        config = super().state_dict().copy()
        config.update({'stride': self.stride})
        return config

    @_abstractmethod
    def forward(self, inputs, *args, **kwargs):
        """逻辑实现部分."""
        ...


class Covariance(_OuterProductLayer):
    """计算每个stride各时间序列片段的covariance.
    Notes:
        计算每个stride每两个feature之间的covariance大小，
        输出feature数量为features * (features - 1) / 2
    """

    def forward(self, inputs: torch.Tensor, *args, **kwargs):
        """函数主逻辑实现部分.
        Args:
            inputs (tensor): 输入dimension为(batch_size, time_steps, features)
        Returns:
            dimension 为(batch_size, time_steps / stride,
            features * (features - 1) / 2)
        """
        if self.intermediate_shape is None:
            self.build(inputs.shape)
        # compute means for each stride
        means = F.avg_pool1d(inputs.transpose(1, 2),
                             kernel_size=(self.stride,),
                             stride=self.stride,
                             )

        # subtract means for each stride
        means_broadcast = means.unsqueeze(-1).repeat((1, 1, 1, self.stride)).flatten(-2, -1)
        means_subtracted = torch.subtract(inputs, means_broadcast.transpose(1, 2))
        means_subtracted = torch.reshape(means_subtracted,
                                         self.intermediate_shape)

        # compute covariance matrix
        covariance_matrix = torch.einsum("ijk,ijm->ikm",
                                         means_subtracted,
                                         means_subtracted)
        covariance_matrix = covariance_matrix / (self.stride - 1)

        # get the lower part of the covariance matrix
        # without the diagonal elements
        covariances = torch.masked_select(covariance_matrix,
                                          self.lower_mask,
                                          )
        covariances = torch.reshape(covariances, self.out_shape)
        return covariances


class Correlation(_OuterProductLayer):
    """计算每个stride各时间序列的相关系数.
    Notes:
        计算每个stride每两个feature之间的correlation coefficient，
        输出feature数量为features * (features - 1) / 2
    """

    def forward(self, inputs, *args, **kwargs):
        """函数主逻辑实现部分.
        Args:
            inputs (tensor): 输入dimension为(batch_size, time_steps, features)
        Returns:
            dimension 为(batch_size, time_steps / stride,
            features * (features - 1) / 2)
        """
        # compute means for each stride
        if self.intermediate_shape is None:
            self.build(inputs.shape)
        # compute means for each stride
        means = F.avg_pool1d(inputs.transpose(1, 2),
                             kernel_size=(self.stride,),
                             stride=self.stride,
                             )

        # subtract means for each stride
        means_broadcast = means.unsqueeze(-1).repeat((1, 1, 1, self.stride)).flatten(-2, -1)
        means_subtracted = torch.subtract(inputs, means_broadcast.transpose(1, 2))
        means_subtracted = torch.reshape(means_subtracted,
                                         self.intermediate_shape)

        # compute standard deviations for each strides
        squared_diff = torch.square(means_subtracted)
        mean_squared_error = torch.mean(squared_diff, dim=1)
        std = torch.sqrt(mean_squared_error)

        # get denominator of correlation matrix
        denominator_matrix = torch.einsum("ik,im->ikm", std, std)

        # compute covariance matrix
        covariance_matrix = torch.einsum("ijk,ijm->ikm",
                                         means_subtracted,
                                         means_subtracted)
        covariance_matrix = covariance_matrix / self.stride

        # take the lower triangle of each matrix without diagonal
        covariances = torch.masked_select(covariance_matrix,
                                          self.lower_mask,
                                          )
        denominators = torch.masked_select(denominator_matrix,
                                           self.lower_mask,
                                           )
        correlations = torch.nan_to_num(torch.div(covariances, denominators), 0, 0, 0)
        correlations = torch.reshape(correlations, self.out_shape)
        return correlations


class FeatureExpansion(nn.Module):
    """计算时间序列特征扩张层，汇总6个计算层.
    Notes:
        该层扩张时间序列的feature数量，并通过stride缩短时间序列长度，
        其包括一下一些feature:
            - standard deviation
            - mean / standard deviation
            - linear decay average
            - return of each stride
            - covariance of each two features for each stride
            - correlation coefficient of each two features for each stride
    """

    def __init__(self, stride=10, **kwargs):
        """时间序列特征扩张.
        Args:
            stride (int): time steps需要是stride的整数倍
        """
        if type(stride) is not int or stride <= 1:
            raise ValueError("Illegal Argument: stride should be an integer "
                             "greater than 1")
        super(FeatureExpansion, self).__init__(**kwargs)
        self.stride = stride
        self.std = Std(stride=self.stride)
        self.z_score = ZScore(stride=self.stride)
        self.linear_decay = LinearDecay(stride=self.stride)
        self.return_ = Return(stride=self.stride)
        self.covariance = Covariance(stride=self.stride)
        self.correlation = Correlation(stride=self.stride)

    def forward(self, inputs, *args, **kwargs):
        """函数主逻辑实现部分.
        Args:
            inputs (tensor): 输入dimension为(batch_size, time_steps, features)
        Returns:
            dimension 为(batch_size, time_steps / stride,
            features * (features + 3))
        """
        std_output = self.std(inputs)
        z_score_output = self.z_score(inputs)
        decay_linear_output = self.linear_decay(inputs)
        return_output = self.return_(inputs)
        covariance_output = self.covariance(inputs)
        correlation_output = self.correlation(inputs)
        return torch.cat((std_output,
                          z_score_output,
                          decay_linear_output,
                          return_output,
                          covariance_output,
                          correlation_output), dim=2)

    def get_config(self):
        """获取参数，保存模型需要的函数."""
        config = super().state_dict().copy()
        config.update({'stride': self.stride})
        return config

class AlphaNet(nn.Module):
    def __init__(self,max_epoch,early_stop_rounds,save_path):
        super(AlphaNet, self).__init__()
        self.max_epoch = max_epoch
        self.early_stop_rounds =early_stop_rounds
        self.save_path = save_path

class AlphaNetV2(AlphaNet):
    """神经网络模型，继承``keras.Model``类.
    alpha net v2版本模型.
    Notes:
        复现华泰金工 alpha net V2 版本
        ``input: (batch_size, history time steps, features)``
    """

    def __init__(self,
                 dropout=0.0,
                 l2=0.001,
                 stride=10,
                 feature=9,
                 classification=False,
                 categories=0,
                 save_path= None,
                 max_epoch = 100,
                 early_stop_rounds = 5,
                 *args,
                 **kwargs):
        """Alpha net v3.
        Notes:
            alpha net v2 版本的全tensorflow实现，结构详见代码展开
        Args:
            dropout: 跟在特征扩张以及Batch Normalization之后的dropout，默认无dropout
            l2: 输出层的l2-regularization参数
        """
        super(AlphaNetV2, self).__init__(max_epoch,early_stop_rounds,save_path)
        self.l2 = l2
        self.dropout = dropout
        self.stride = stride
        self.expanded = FeatureExpansion(stride=self.stride)
        self.normalized = nn.BatchNorm1d(feature * (feature + 3))
        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(feature * (feature + 3), 30, batch_first=True)
        self.normalized_2 = nn.BatchNorm1d(30)
        self.activation = None
        if classification:
            if categories < 1:
                raise ValueError("categories should be at least 1")
            elif categories == 1:
                self.outputs = nn.Linear(self.normalized_2.num_features, 1)
                self.activation = nn.Sigmoid()
            else:
                self.outputs = nn.Linear(self.normalized_2.num_features, categories)
                self.activation = nn.Softmax()
        else:
            self.outputs = nn.Linear(self.normalized_2.num_features, 1)
        nn.init.trunc_normal_(self.outputs.weight)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, inputs):
        """计算逻辑实现."""
        expanded = self.expanded(inputs).transpose(1, 2)
        normalized = self.normalized(expanded).transpose(1, 2)
        lstm, state = self.lstm(normalized)
        normalized2 = self.normalized_2(lstm[:, -1, :])
        dropout = self.dropout(normalized2)
        output = self.outputs(dropout)
        if self.activation:
            output = self.activation(output)
        return output

    def get_loss(selfself, predict, true):
        loss = nn.MSELoss()
        return loss(predict, true)

    def fit(self,train_data,val_data):
        stop_steps = 0
        train_loss = 0
        best_loss = np.inf
        for step in range(self.max_epoch):
            if stop_steps >= self.early_stop_rounds:
                break
            self.train()
            for i, data in tqdm(enumerate(train_data), total=len(train_data)):
                inputs, labels = data
                tmp = self.forward(inputs)
                loss = self.get_loss(tmp.ravel(), labels)
                self.optimizer.zero_grad()  # 所有参数的梯度清零
                loss.backward()  # 即反向传播求梯度
                self.optimizer.step()
            with torch.no_grad():
                self.eval()
                val_loss = AverageMeter()
                for i, data in tqdm(enumerate(val_data), total=len(val_data)):
                    inputs, labels = data
                    tmp = self.forward(inputs)
                    loss = self.get_loss(tmp.ravel(), labels)
                    val_loss.update(loss, len(tmp.ravel()))
            print(val_loss.avg)
            stop_steps += 1
            if val_loss.avg < best_loss:
                best_loss = val_loss.avg
                stop_steps = 0
                torch.save(self.state_dict(), self.save_path)
    def predict(self,test_data):
        pred = []
        label = []
        for i, data in tqdm(enumerate(test_data), total=len(test_data)):
            inputs, labels = data
            tmp = self.forward(inputs)
            pred.append(tmp)
            label.append(labels)
        return pred,label

    # def score(self,feature,label):
    #     return
    def get_config(self):
        """获取参数，保存模型需要的函数."""
        config = super().state_dict().copy()
        config.update({'dropout': self.dropout,
                       'l2': self.l2,
                       'stride': self.stride})
        return config


class AlphaNetV3(AlphaNet):
    """神经网络模型，继承``keras.Model``类.
    alpha net v3版本模型.
    Notes:
        复现华泰金工 alpha net V3 版本
        ``input: (batch_size, history time steps, features)``
    """

    def __init__(self,
                 dropout=0.0,
                 l2=0.001,
                 classification=False,
                 categories=0,
                 feature=9,
                 recurrent_unit="GRU",
                 hidden_units=30,
                 save_path= None,
                 max_epoch = 100,
                 early_stop_rounds = 5,
                 *args,
                 **kwargs):
        """Alpha net v3.
        Notes:
            alpha net v3 版本的全tensorflow实现，结构详见代码展开
        Args:
            dropout: 跟在特征扩张以及Batch Normalization之后的dropout，默认无dropout
            l2: 输出层的l2-regularization参数
            classification: 是否为分类问题
            categories: 分类问题的类别数量
            recurrent_unit (str): 该参数可以为"GRU"或"LSTM"
        """
        super(AlphaNetV3, self).__init__(max_epoch,early_stop_rounds,save_path)
        self.l2 = l2
        self.dropout = dropout
        self.expanded10 = FeatureExpansion(stride=10)
        self.expanded5 = FeatureExpansion(stride=5)
        self.normalized10 = torch.nn.BatchNorm1d(feature * (feature + 3))
        self.normalized5 = torch.nn.BatchNorm1d(feature * (feature + 3))
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        if recurrent_unit == "GRU":
            self.recurrent10 = torch.nn.GRU(feature * (feature + 3), hidden_units, batch_first=True)
            self.recurrent5 = torch.nn.GRU(feature * (feature + 3), hidden_units, batch_first=True)
        elif recurrent_unit == "LSTM":
            self.recurrent10 = torch.nn.LSTM(feature * (feature + 3), hidden_units, batch_first=True)
            self.recurrent5 = torch.nn.LSTM(feature * (feature + 3), hidden_units, batch_first=True)
        else:
            raise ValueError("Unknown recurrent_unit")
        self.normalized10_2 = torch.nn.BatchNorm1d(hidden_units)
        self.normalized5_2 = torch.nn.BatchNorm1d(hidden_units)
        self.activation = None
        if classification:
            if categories < 1:
                raise ValueError("categories should be at least 1")
            elif categories == 1:
                self.outputs = nn.Linear(self.normalized10_2.num_features + self.normalized5_2.num_features, 1)
                self.activation = nn.Sigmoid()
            else:
                self.outputs = nn.Linear(self.normalized10_2.num_features + self.normalized5_2.num_features, categories)
                self.activation = nn.Softmax()
        else:
            self.outputs = nn.Linear(self.normalized10_2.num_features + self.normalized5_2.num_features, 1)
        nn.init.trunc_normal_(self.outputs.weight)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, inputs, mask=None):
        """计算逻辑实现."""
        expanded10 = self.expanded10(inputs)
        expanded5 = self.expanded5(inputs)
        normalized10 = self.normalized10(expanded10.transpose(1, 2)).transpose(1, 2)
        normalized5 = self.normalized5(expanded5.transpose(1, 2)).transpose(1, 2)
        recurrent10, state10 = self.recurrent10(normalized10)
        recurrent5, state5 = self.recurrent5(normalized5)
        normalized10_2 = self.normalized10_2(recurrent10.transpose(1, 2)).transpose(1, 2)
        normalized5_2 = self.normalized5_2(recurrent5.transpose(1, 2)).transpose(1, 2)
        concat = torch.cat([normalized10_2[:, -1, :], normalized5_2[:, -1, :]], dim=-1)
        dropout = self.dropout_layer(concat)
        output = self.outputs(dropout)
        if self.activation:
            output = self.activation(output)
        return output

    def get_loss(selfself, predict, true):
        loss = nn.MSELoss()
        return loss(predict, true)

    def fit(self,train_data,val_data):
        stop_steps = 0
        train_loss = 0
        best_loss = np.inf
        print(self.max_epoch)
        for step in range(self.max_epoch):
            if stop_steps >= self.early_stop_rounds:
                break
            self.train()
            for i, data in tqdm(enumerate(train_data), total=len(train_data)):
                inputs, labels = data
                tmp = self.forward(inputs)
                loss = self.get_loss(tmp.ravel(), labels)
                self.optimizer.zero_grad()  # 所有参数的梯度清零
                loss.backward()  # 即反向传播求梯度
                self.optimizer.step()
            with torch.no_grad():
                self.eval()
                val_loss = AverageMeter()
                for i, data in tqdm(enumerate(val_data), total=len(val_data)):
                    inputs, labels = data
                    tmp = self.forward(inputs)
                    loss = self.get_loss(tmp.ravel(), labels)
                    val_loss.update(loss, len(tmp.ravel()))
            print(val_loss.avg)
            stop_steps += 1
            if val_loss.avg < best_loss:
                best_loss = val_loss.avg
                stop_steps = 0
                torch.save(self.state_dict(), self.save_path)
    def predict(self,test_data):
        pred = []
        label = []
        for i, data in tqdm(enumerate(test_data), total=len(test_data)):
            inputs, labels = data
            tmp = self.forward(inputs)
            pred.append(tmp)
            label.append(labels)
        return pred,label

    def get_config(self):
        """获取参数，保存模型需要的函数."""
        config = super().get_config().copy()
        config.update({'dropout': self.dropout,
                       'l2': self.l2})
        return config

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count