import os
import numpy as np
from logging import getLogger, disable, INFO

from libcity.data.dataset import AbstractDataset
from libcity.data.utils import generate_dataloader
from libcity.utils import ensure_dir
from libcity.utils import (
    StandardScaler,
    NormalScaler,
    NoneScaler,
    MinMax01Scaler,
    MinMax11Scaler,
    LogScaler,
    ensure_dir,
)


class NpzDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        # disable(INFO)
        # config/data/NpzDataset.json
        self.dataset = self.config.get("dataset", "")
        self.batch_size = self.config.get("batch_size", 64)
        self.cache_dataset = self.config.get("cache_dataset", True)
        self.num_workers = self.config.get("num_workers", 0)
        self.pad_with_last_sample = self.config.get("pad_with_last_sample", True)
        self.train_rate = self.config.get("train_rate", 0.7)
        self.eval_rate = self.config.get("eval_rate", 0.1)
        self.scaler_type = self.config.get("scaler", "none")
        self.ext_scaler_type = self.config.get("ext_scaler", "none")
        self.load_external = self.config.get("load_external", False)
        self.normal_external = self.config.get("normal_external", False)
        self.add_time_in_day = self.config.get("add_time_in_day", False)
        self.add_day_in_week = self.config.get("add_day_in_week", False)
        self.input_window = self.config.get("input_window", 12)
        self.output_window = self.config.get("output_window", 12)

        
        self.data_path = "./raw_data/" + self.dataset + "/"
        if not os.path.exists(self.data_path):
            raise ValueError(
                "Dataset {} not exist! Please ensure the path "
                "'./raw_data/{}/' exist!".format(self.dataset, self.dataset)
            )
        # 加载数据集的config.json文件
        self.weight_col = self.config.get("weight_col", "")
        self.data_col = self.config.get("data_col", "")
        self.ext_col = self.config.get("ext_col", "")
        self.data_files = self.config.get("data_files", self.dataset)
        self.output_dim = self.config.get("output_dim", 1)
        self.time_intervals = self.config.get("time_intervals", 300)  # s
        self.init_weight_inf_or_zero = self.config.get("init_weight_inf_or_zero", "inf")
        self.set_weight_link_or_dist = self.config.get(
            "set_weight_link_or_dist", "dist"
        )
        self.bidir_adj_mx = self.config.get("bidir_adj_mx", False)
        self.calculate_weight_adj = self.config.get("calculate_weight_adj", False)
        self.weight_adj_epsilon = self.config.get("weight_adj_epsilon", 0.1)
        self.distance_inverse = self.config.get("distance_inverse", False)

        # infer parameters through input train data
        self.feature_dim = None
        self.ext_dim = None
        self.adj_mx = np.load(self.data_path + "adj_mx.npz")['adj_mx']
        self.data = None
        self.feature_name = {"X": "float", "y": "float"}  # 此类的输入只有X和y
        self.scaler = None
        self.ext_scaler = None
        self.num_nodes = self.config.get("num_nodes", 128)
        self.num_batches = 0
        self._logger = getLogger()

        # create cache folder
        self.cache_file_folder = self.config['cache_file_folder']
        self.cache_file_name = os.path.join(
            "./libcity/cache/dataset_cache/",
            "traffic_state_{}.npz".format(self.parameters_str),
        )
        ensure_dir(self.cache_file_folder)

    def get_data(self):
        raw_train_data = np.load(os.path.join(self.data_path, "train.npz"))
        x_train = raw_train_data["x"]
        y_train = raw_train_data["y"]
        raw_val_data = np.load(os.path.join(self.data_path, "val.npz"))
        x_val = raw_val_data["x"]
        y_val = raw_val_data["y"]
        raw_test_data = np.load(os.path.join(self.data_path, "test.npz"))
        x_test = raw_test_data["x"]
        y_test = raw_test_data["y"]
        self.feature_dim = x_train.shape[-1]
        self.ext_dim = self.feature_dim - self.output_dim
        self.scaler = self._get_scalar(
            self.scaler_type,
            x_train[..., : self.output_dim],
            y_train[..., : self.output_dim],
        )

        x_train[..., : self.output_dim] = self.scaler.transform(
            x_train[..., : self.output_dim]
        )
        y_train[..., : self.output_dim] = self.scaler.transform(
            y_train[..., : self.output_dim]
        )
        x_val[..., : self.output_dim] = self.scaler.transform(
            x_val[..., : self.output_dim]
        )
        y_val[..., : self.output_dim] = self.scaler.transform(
            y_val[..., : self.output_dim]
        )
        x_test[..., : self.output_dim] = self.scaler.transform(
            x_test[..., : self.output_dim]
        )
        y_test[..., : self.output_dim] = self.scaler.transform(
            y_test[..., : self.output_dim]
        )

        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        test_data = list(zip(x_test, y_test))
        # 转Dataloader
        (
            self.train_dataloader,
            self.eval_dataloader,
            self.test_dataloader,
        ) = generate_dataloader(
            train_data,
            eval_data,
            test_data,
            self.feature_name,
            self.batch_size,
            self.num_workers,
            pad_with_last_sample=self.pad_with_last_sample,
        )
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是网格的个数，
        len_row是网格的行数，len_column是网格的列数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {
            "scaler": self.scaler,
            "adj_mx": self.adj_mx,
            "num_nodes": self.num_nodes,
            "feature_dim": self.feature_dim,
            "ext_dim": self.ext_dim,
            "output_dim": self.output_dim,
            "num_batches": self.num_batches,
        }

    def _get_scalar(self, scaler_type, x_train, y_train):
        """
        根据全局参数`scaler_type`选择数据归一化方法

        Args:
            x_train: 训练数据X
            y_train: 训练数据y

        Returns:
            Scaler: 归一化对象
        """
        if scaler_type == "normal":
            scaler = NormalScaler(maxx=max(x_train.max(), y_train.max()))
            self._logger.info("NormalScaler max: " + str(scaler.max))
        elif scaler_type == "standard":
            scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
            self._logger.info(
                "StandardScaler mean: " + str(scaler.mean) + ", std: " + str(scaler.std)
            )
        elif scaler_type == "minmax01":
            scaler = MinMax01Scaler(
                maxx=max(x_train.max(), y_train.max()),
                minn=min(x_train.min(), y_train.min()),
            )
            self._logger.info(
                "MinMax01Scaler max: " + str(scaler.max) + ", min: " + str(scaler.min)
            )
        elif scaler_type == "minmax11":
            scaler = MinMax11Scaler(
                maxx=max(x_train.max(), y_train.max()),
                minn=min(x_train.min(), y_train.min()),
            )
            self._logger.info(
                "MinMax11Scaler max: " + str(scaler.max) + ", min: " + str(scaler.min)
            )
        elif scaler_type == "log":
            scaler = LogScaler()
            self._logger.info("LogScaler")
        elif scaler_type == "none":
            scaler = NoneScaler()
            self._logger.info("NoneScaler")
        else:
            raise ValueError("Scaler type error!")
        return scaler

    @property
    def parameters_str(self):
        return (
            str(self.dataset)
            + "_"
            + str(self.input_window)
            + "_"
            + str(self.output_window)
            + "_"
            + str(self.train_rate)
            + "_"
            + str(self.eval_rate)
            + "_"
            + str(self.scaler_type)
            + "_"
            + str(self.batch_size)
            + "_"
            + str(self.load_external)
            + "_"
            + str(self.add_time_in_day)
            + "_"
            + str(self.add_day_in_week)
            + "_"
            + str(self.pad_with_last_sample)
        )
