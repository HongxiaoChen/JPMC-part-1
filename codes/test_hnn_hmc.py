import unittest
import tensorflow as tf
import numpy as np
import os
import sys
from pathlib import Path
from codes.functions import functions

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


class TestHNNHMC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """在所有测试开始前设置环境"""
        # 切换到codes目录
        os.chdir(os.path.join(PROJECT_ROOT, 'codes'))
        # 将codes目录添加到Python路径
        if str(PROJECT_ROOT / 'codes') not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT / 'codes'))

        # 在切换工作目录后再导入所需模块
        global HNNSampler, get_args, MLP, HNN
        from codes.hnn_hmc import HNNSampler
        from codes.get_args import get_args
        from codes.nn_models import MLP
        from codes.hnn import HNN

    def setUp(self):
        """设置测试环境"""
        # 配置测试用例
        self.test_configs = [
            {
                'name': 'nD_Rosenbrock',
                'input_dim': 6,
                'latent_dim': 100,
                'weight_file': 'nD_Rosenbrock100',  # 移除扩展名
                'settings': [
                    {'step_size': 0.01, 'trajectory_length': 1},
                    {'step_size': 0.05, 'trajectory_length': 5},
                ]
            },
            {
                'name': '2D_Neal_funnel',
                'input_dim': 4,
                'latent_dim': 2,
                'weight_file': '2D_Neal_funnel250',  # 移除扩展名
                'settings': [
                    {'step_size': 0.01, 'trajectory_length': 1},
                    {'step_size': 0.05, 'trajectory_length': 5},
                ]
            },
            {
                'name': '5D_illconditioned_Gaussian',
                'input_dim': 10,
                'latent_dim': 5,
                'weight_file': '5D_illconditioned_Gaussian250',  # 移除扩展名
                'settings': [
                    {'step_size': 0.01, 'trajectory_length': 1},
                    {'step_size': 0.05, 'trajectory_length': 5},
                ]
            },
            {
                'name': 'nD_Rosenbrock',
                'input_dim': 20,
                'latent_dim': 10,
                'weight_file': '10D_Rosenbrock250',  # 移除扩展名
                'settings': [
                    {'step_size': 0.01, 'trajectory_length': 1},
                    {'step_size': 0.05, 'trajectory_length': 5},
                ]
            }
        ]

    def get_modified_args(self, config, setting):
        """获取修改后的参数"""
        args = get_args()
        args.dist_name = config['name']
        args.input_dim = config['input_dim']
        args.latent_dim = config['latent_dim']
        args.hmc_step_size = setting['step_size']
        args.trajectory_length = setting['trajectory_length']
        # 使用较小的样本数以加快测试
        args.hmc_samples = 15
        args.num_burnin = 5
        args.num_chains = 2
        # 使用PROJECT_ROOT来构建完整路径
        args.save_dir = str(PROJECT_ROOT / 'codes' / 'files' / config['weight_file'])
        return args

    def test_hnn_hmc_initialization(self):
        """测试HNN-HMC初始化"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                # 检查权重文件（检查.index文件存在）
                weight_path_index = PROJECT_ROOT / 'codes' / 'files' / f"{config['weight_file']}.index"
                weight_path_data = PROJECT_ROOT / 'codes' / 'files' / f"{config['weight_file']}.data-00000-of-00001"

                print(f"Checking weight files:")
                print(f"Index file: {weight_path_index}")
                print(f"Data file: {weight_path_data}")

                # 检查两个文件都存在
                self.assertTrue(weight_path_index.exists(),
                                f"Weight index file not found: {weight_path_index}")
                self.assertTrue(weight_path_data.exists(),
                                f"Weight data file not found: {weight_path_data}")

                args = self.get_modified_args(config, config['settings'][0])
                sampler = HNNSampler(args)

                # 检查初始化的参数
                self.assertEqual(sampler.state_dim, args.input_dim // 2)
                self.assertEqual(sampler.total_dim, args.input_dim)
                self.assertEqual(sampler.step_size, args.hmc_step_size)
                self.assertEqual(sampler.trajectory_length, args.trajectory_length)
                self.assertEqual(len(sampler.trajectories), 0)

                # 验证HNN模型结构
                self.assertEqual(sampler.hnn_model.__class__.__name__, 'HNN')
                self.assertEqual(sampler.hnn_model.differentiable_model.__class__.__name__, 'MLP')
                self.assertEqual(sampler.hnn_model.dim, args.input_dim // 2)

    def test_hnn_hmc_sampling(self):
        """测试HNN-HMC采样"""
        for config in self.test_configs:
            for setting in config['settings']:
                with self.subTest(distribution=config['name'],
                                  step_size=setting['step_size'],
                                  trajectory_length=setting['trajectory_length']):
                    args = self.get_modified_args(config, setting)
                    sampler = HNNSampler(args)

                    # 执行采样
                    samples, acceptance = sampler.sample()

                    # 检查输出维度
                    expected_shape = (args.num_chains, args.hmc_samples, args.input_dim)
                    self.assertEqual(samples.shape, expected_shape)

                    # 检查acceptance的维度
                    expected_acceptance_shape = (args.num_chains,
                                                 args.hmc_samples - args.num_burnin)
                    self.assertEqual(acceptance.shape, expected_acceptance_shape)

                    # 检查样本的有限性
                    self.assertTrue(tf.reduce_all(tf.math.is_finite(samples)))

                    # 检查接受率是否在合理范围内
                    self.assertTrue(tf.reduce_all(acceptance >= 0))
                    self.assertTrue(tf.reduce_all(acceptance <= 1))

                    # 检查平均接受率
                    mean_acceptance = tf.reduce_mean(acceptance)
                    self.assertGreater(float(mean_acceptance), 0.1)

    def test_trajectory_storage(self):
        """测试轨迹存储功能"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                args = self.get_modified_args(config, config['settings'][0])
                sampler = HNNSampler(args)

                # 执行采样
                samples, _ = sampler.sample()

                # 获取存储的轨迹
                trajectories = sampler.get_trajectories()

                # 检查轨迹数量
                expected_trajectory_count = args.num_chains * (args.hmc_samples - args.num_burnin)
                self.assertEqual(len(trajectories), expected_trajectory_count)

                # 检查轨迹的维度
                n_steps = args.trajectory_length * int(1 / args.hmc_step_size)
                expected_trajectory_shape = (n_steps + 1, 1, args.input_dim)
                self.assertEqual(trajectories[0].shape, expected_trajectory_shape)

    def test_energy_conservation(self):
        """测试能量守恒"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                args = self.get_modified_args(config, config['settings'][0])
                sampler = HNNSampler(args)

                # 执行采样
                sampler.sample()

                # 获取轨迹
                trajectories = sampler.get_trajectories()

                # 检查每条轨迹的能量变化
                for traj in trajectories:
                    initial_state = tf.convert_to_tensor(traj[0], dtype=tf.float32)
                    final_state = tf.convert_to_tensor(traj[-1], dtype=tf.float32)

                    initial_H = float(sampler.hnn_model.compute_hamiltonian(initial_state))
                    final_H = float(sampler.hnn_model.compute_hamiltonian(final_state))

                    # 使用HNN计算的能量变化应该较小
                    energy_diff = abs(final_H - initial_H)
                    self.assertLess(energy_diff, 1.0)

    def test_error_handling(self):
        """测试错误处理"""
        config = self.test_configs[0]
        args = self.get_modified_args(config, config['settings'][0])

        # 测试无效的step_size
        args.hmc_step_size = -0.1
        with self.assertRaises(ValueError):
            HNNSampler(args)

        # 测试无效的trajectory_length
        args.hmc_step_size = 0.01
        args.trajectory_length = -1
        with self.assertRaises(ValueError):
            HNNSampler(args)

        # 测试无效的维度
        args.trajectory_length = 5
        args.input_dim = -1
        with self.assertRaises(ValueError):
            HNNSampler(args)

        # 测试无效的latent_dim
        args.input_dim = 6
        args.latent_dim = -1
        with self.assertRaises(ValueError):
            HNNSampler(args)

        # 测试不存在的权重文件
        args.latent_dim = 100
        args.save_dir = "nonexistent_path"
        with self.assertRaises(tf.errors.NotFoundError):
            HNNSampler(args)

    def tearDown(self):
        """清理测试环境"""
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()