import unittest
import tensorflow as tf
import numpy as np
import os
import shutil
from pathlib import Path
import sys

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

class TestDataGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """在所有测试开始前设置环境"""
        # 切换到codes目录
        os.chdir(os.path.join(PROJECT_ROOT, 'codes'))
        # 将codes目录添加到Python路径
        if str(PROJECT_ROOT / 'codes') not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT / 'codes'))

        # 在切换工作目录后再导入所需模块
        global get_trajectory, get_dataset, get_args, from_pickle
        from codes.data import get_trajectory, get_dataset
        from codes.get_args import get_args
        from codes.utils import from_pickle

    def setUp(self):
        """设置测试环境"""
        # 创建临时保存目录
        self.temp_save_dir = PROJECT_ROOT / 'temp_test_data'
        self.temp_save_dir.mkdir(exist_ok=True)

        # 配置测试用例
        self.test_configs = [
            {
                'name': 'nD_Rosenbrock',
                'input_dim': 6,
                'samples': 5,
                'len_sample': 2,
                'test_fraction': 0.2
            },
            {
                'name': '2D_Neal_funnel',
                'input_dim': 4,
                'samples': 5,
                'len_sample': 2,
                'test_fraction': 0.2
            },
            {
                'name': '5D_illconditioned_Gaussian',
                'input_dim': 10,
                'samples': 5,
                'len_sample': 2,
                'test_fraction': 0.2
            },
            {
                'name': 'Allen_Cahn',
                'input_dim': 50,
                'samples': 5,
                'len_sample': 2,
                'test_fraction': 0.2
            }
        ]

    def get_modified_args(self, config):
        """获取修改后的参数"""
        args = get_args()
        args.dist_name = config['name']
        args.input_dim = config['input_dim']
        args.num_samples = config['samples']
        args.len_sample = config['len_sample']
        args.test_fraction = config['test_fraction']
        args.save_dir = str(self.temp_save_dir)
        return args

    def test_get_trajectory(self):
        """测试轨迹生成函数"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                # 获取修改后的参数
                args = self.get_modified_args(config)

                # 测试默认参数
                traj_split, deriv_split, t_eval = get_trajectory(args=args)

                # 检查输出维度
                self.assertEqual(len(traj_split), config['input_dim'])
                self.assertEqual(len(deriv_split), config['input_dim'])

                # 检查每个分量的维度
                for traj, deriv in zip(traj_split, deriv_split):
                    self.assertEqual(traj.shape[0], 1)  # batch size
                    self.assertEqual(deriv.shape[0], 1)  # batch size
                    self.assertEqual(traj.shape[2], 1)  # 每个分量是标量
                    self.assertEqual(deriv.shape[2], 1)  # 每个分量是标量

                # 测试自定义参数
                custom_t_span = [0, 2]
                custom_dt = 0.1
                y0 = tf.zeros([1, config['input_dim']])

                traj_split, deriv_split, t_eval = get_trajectory(
                    t_span=custom_t_span,
                    dt=custom_dt,
                    y0=y0,
                    args=args
                )

                # 检查时间点数量
                expected_steps = int((custom_t_span[1] - custom_t_span[0]) / custom_dt)
                self.assertEqual(t_eval.shape[0], expected_steps + 1)

                # 检查轨迹的连续性
                for traj in traj_split:
                    traj_squeezed = tf.squeeze(traj, axis=2)  # 移除最后的维度1
                    diff = tf.reduce_max(tf.abs(traj_squeezed[:, 1:] - traj_squeezed[:, :-1]))
                    self.assertLess(float(diff), 10.0)

    def test_get_dataset(self):
        """测试数据集生成函数"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                # 获取修改后的参数
                args = self.get_modified_args(config)

                # 设置自定义时间范围
                custom_t_span = [0, 2]

                # 生成数据集
                dataset = get_dataset(
                    seed=42,
                    samples=config['samples'],
                    test_split=0.8,  # 80% 训练集
                    args=args,
                    t_span=custom_t_span
                )

                # 检查数据集的结构
                expected_keys = ['coords', 'dcoords', 'test_coords', 'test_dcoords']
                self.assertTrue(all(key in dataset for key in expected_keys))

                # 计算预期的数据点数量
                dt = 0.025  # 默认步长
                n_steps = int((custom_t_span[1] - custom_t_span[0]) / dt)
                points_per_sample = n_steps + 1
                total_points = config['samples'] * points_per_sample

                train_size = int(total_points * 0.8)  # 80% 训练集
                test_size = total_points - train_size

                # 检查维度
                self.assertEqual(dataset['coords'].shape[0], train_size)
                self.assertEqual(dataset['coords'].shape[1], config['input_dim'])
                self.assertEqual(dataset['test_coords'].shape[0], test_size)
                self.assertEqual(dataset['test_coords'].shape[1], config['input_dim'])

                # 验证保存的文件
                save_path = self.temp_save_dir / f"{config['name']}{args.len_sample}.pkl"
                self.assertTrue(save_path.exists())

                # 加载并验证保存的数据
                loaded_data = from_pickle(save_path)
                for key in expected_keys:
                    self.assertTrue(tf.reduce_all(tf.equal(dataset[key], loaded_data[key])))

                # 检查数据的有限性
                for key in expected_keys:
                    self.assertTrue(tf.reduce_all(tf.math.is_finite(dataset[key])))

    def tearDown(self):
        """清理测试环境"""
        # 删除临时目录及其内容
        if self.temp_save_dir.exists():
            shutil.rmtree(self.temp_save_dir)
        # 清理TensorFlow会话
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()