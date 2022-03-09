from typing import List as _List

import numpy as np
import xarray as xr


class TimeSeriesData:
    """单个时间序列信息.

        Notes:
            用于储存个股的数据信息及预测label，全部使用xarray，日期格式为整数: ``YYYYMMDD``。
            数据分三个部分：时间，数据，标签，第一个维度都是时间，数据的第二个维度为特征。

    """

    def __init__(self,
                 data: xr.Dataset,
                 labels: xr.DataArray):
        self.data = data
        self.labels = labels


class TrainValData:
    """根据训练天数、验证天数、样本历史长度、训练起点生成不同训练阶段的数据."""

    def __init__(self,
                 data: xr.Dataset,
                 train_length: int = 1200,
                 validate_length: int = 300,
                 history_length: int = 30,
                 train_val_gap: int = 10,
                 sample_step: int = 2,
                 fill_na: np.float32 = np.NAN,
                 normalize: bool = False
                 ):
        """用于获取不同阶段的训练集和验证集.
        Notes:
            ``time_series_list``储存全部的时间序列信息，
            其中每支股票序列为一个单独``TimeSeriesData``，
            完整数据为``List[TimeSeriesData]``类型。

            此外需要提供训练集总交易天数(``train_length``)、
            验证集总交易天数(``validate_length``)、
            单个样本用到的历史长度(``history_length``)、
            采样步进大小(``sample_step``)。

            使用方法为：通过get(start_date)方法获取从start_date
            开始的训练机和验证集。通过逐渐增大start_date训练多个模型回测。

            ``train_val_gap``参数为验证集第一天与训练集最后一天中间间隔的天数，
            如果是相临，则train_val_gap = 0。设置该参数的目的如下：

            如果希望预测未来十天的累计收益，则预测时用到的输入数据为最近的历史数据来预测
            未来十天的累计收益，即用t(-history)到t(0)的数据来预测t(1)到t(11)的累计收益
            而训练时因为要用到十天累计收益做标签，最近的一个十天累计收益是从t(-10)到t(0)，
            用到的历史数据则必须是t(-history-11)到t(-11)的数据。
            而validation时，如果第一个预测点是t(1)(明天收盘价)至t(11)的累计收益，
            则与最后一个训练的数据即：t(-10)至t(0)之间间隔了10天，
            使用``train_val_gap=10``。

            可选项为fill_na，缺失数据填充值，默认为np.Na
            训练时跳过所有有缺失数据的样本。

        Args:
            time_series_list: TimeSeriesData 列表
            train_length: 训练集天数
            validate_length: 验证集天数
            history_length: 每个样本的历史天数
            train_val_gap: 训练集与验证集的间隔
            sample_step: 采样sample时步进的天数
            fill_na: 默认填充为np.NaN，训练时会跳过有确实数据的样本
            normalize: 是否对非率值做每个历史片段的max/min标准化
        """
        self._star_idx = (tmp.date<20200101).sum().values[0]


if __name__ == '__main__':
    tmp = xr.open_mfdataset('my_data/*.nc', chunks={"date": 240}, parallel=True, engine='h5netcdf').to_array(name='feature')
