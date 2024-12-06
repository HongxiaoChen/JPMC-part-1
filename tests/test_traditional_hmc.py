import unittest
import tensorflow as tf
import numpy as np
import os
import sys
from pathlib import Path
from codes.functions import functions
# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


class TestTraditionalHMC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """在所有测试开始前设置环境"""
        # 切换到codes目录
        os.chdir(os.path.join(PROJECT_ROOT, 'codes'))
        # 将codes目录添加到Python路径
        if str(PROJECT_ROOT / 'codes') not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT / 'codes'))

        # 在切换工作目录后再导入所需模块
        global TraditionalHMC, get_args
        from codes.traditional_hmc import TraditionalHMC
        from codes.get_args import get_args

    def setUp(self):
        """设置测试环境"""
        # 配置测试用例
        self.test_configs = [
            {
                'name': 'nD_Rosenbrock',
                'input_dim': 6,
                'settings': [
                    {'step_size': 0.01, 'trajectory_length': 1},
                    {'step_size': 0.05, 'trajectory_length': 5},
                ]
            },
            {
                'name': '2D_Neal_funnel',
                'input_dim': 4,
                'settings': [
                    {'step_size': 0.01, 'trajectory_length': 1},
                    {'step_size': 0.05, 'trajectory_length': 5},
                ]
            },
            {
                'name': '5D_illconditioned_Gaussian',
                'input_dim': 10,
                'settings': [
                    {'step_size': 0.01, 'trajectory_length': 1},
                    {'step_size': 0.05, 'trajectory_length': 5},
                ]
            },
            {
                'name': 'Allen_Cahn',
                'input_dim': 50,
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
        args.hmc_step_size = setting['step_size']
        args.trajectory_length = setting['trajectory_length']
        # 使用较小的样本数以加快测试
        args.hmc_samples = 5
        args.num_burnin = 3
        args.num_chains = 2
        return args

    def test_hmc_initialization(self):
        """测试HMC初始化"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                args = self.get_modified_args(config, config['settings'][0])
                hmc = TraditionalHMC(args)

                # 检查初始化的参数
                self.assertEqual(hmc.state_dim, args.input_dim // 2)
                self.assertEqual(hmc.total_dim, args.input_dim)
                self.assertEqual(hmc.step_size, args.hmc_step_size)
                self.assertEqual(hmc.trajectory_length, args.trajectory_length)
                self.assertEqual(len(hmc.trajectories), 0)

    def test_hmc_sampling(self):
        """测试HMC采样"""
        for config in self.test_configs:
            for setting in config['settings']:
                with self.subTest(distribution=config['name'],
                                  step_size=setting['step_size'],
                                  trajectory_length=setting['trajectory_length']):
                    args = self.get_modified_args(config, setting)
                    hmc = TraditionalHMC(args)

                    # 执行采样
                    samples, acceptance = hmc.sample()

                    # 检查输出维度
                    expected_shape = (args.num_chains, args.hmc_samples, args.input_dim)
                    self.assertEqual(samples.shape, expected_shape)

                    # 检查acceptance的维度
                    expected_acceptance_shape = (args.num_chains,
                                                 args.hmc_samples - args.num_burnin)
                    self.assertEqual(acceptance.shape, expected_acceptance_shape)

                    # 检查样本的有限性
                    self.assertTrue(tf.reduce_all(tf.math.is_finite(samples)))

                    # 检查接受率是否在合理范围内 (0到1之间)
                    self.assertTrue(tf.reduce_all(acceptance >= 0))
                    self.assertTrue(tf.reduce_all(acceptance <= 1))

                    # 检查平均接受率是否在合理范围内 (通常期望在0.2到0.9之间)
                    mean_acceptance = tf.reduce_mean(acceptance)
                    self.assertGreater(float(mean_acceptance), 0.1)


    def test_trajectory_storage(self):
        """测试轨迹存储功能"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                args = self.get_modified_args(config, config['settings'][0])
                hmc = TraditionalHMC(args)

                # 执行采样
                samples, _ = hmc.sample()

                # 获取存储的轨迹
                trajectories = hmc.get_trajectories()

                # 检查轨迹数量（应该等于非burn-in样本数量）
                expected_trajectory_count = args.num_chains * (args.hmc_samples - args.num_burnin)
                self.assertEqual(len(trajectories), expected_trajectory_count)

                # 检查轨迹的维度
                n_steps = int(args.trajectory_length / args.hmc_step_size)
                expected_trajectory_shape = (n_steps + 1, 1, args.input_dim)
                self.assertEqual(trajectories[0].shape, expected_trajectory_shape)

    def test_energy_conservation(self):
        """测试能量守恒"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                args = self.get_modified_args(config, config['settings'][0])
                hmc = TraditionalHMC(args)

                # 执行采样
                hmc.sample()

                # 获取轨迹
                trajectories = hmc.get_trajectories()

                # 检查每条轨迹的能量变化
                for traj in trajectories:
                    initial_state = tf.convert_to_tensor(traj[0], dtype=tf.float32)
                    final_state = tf.convert_to_tensor(traj[-1], dtype=tf.float32)

                    initial_energy = float(functions(initial_state, args))
                    final_energy = float(functions(final_state, args))

                    # 能量变化应该相对较小
                    energy_diff = abs(final_energy - initial_energy)
                    self.assertLess(energy_diff, 1.0)

    def test_error_handling(self):
        """测试错误处理"""
        config = self.test_configs[0]
        args = self.get_modified_args(config, config['settings'][0])

        # 测试无效的步长
        args.hmc_step_size = -0.1
        with self.assertRaises(ValueError):
            TraditionalHMC(args)

        # 测试无效的轨迹长度
        args.hmc_step_size = 0.01
        args.trajectory_length = -1
        with self.assertRaises(ValueError):
            TraditionalHMC(args)

        # 测试无效的维度
        args.trajectory_length = 5
        args.input_dim = -1
        with self.assertRaises(ValueError):
            TraditionalHMC(args)

        # 测试无效的输入维度（奇数维度）
        args.input_dim = 3  # 奇数维度会导致state_dim计算出错
        with self.assertRaises(ValueError):
            hmc = TraditionalHMC(args)
            hmc.sample()

        # 测试极端步长（导致数值不稳定）
        args.input_dim = 6  # 恢复有效维度
        args.hmc_step_size = 1e10
        with self.assertRaises(tf.errors.InvalidArgumentError):
            hmc = TraditionalHMC(args)
            hmc.sample()

        # 测试无效的样本数
        args.hmc_step_size = 0.01  # 恢复正常步长
        args.hmc_samples = 0
        with self.assertRaises(ValueError):
            hmc = TraditionalHMC(args)
            hmc.sample()

        # 测试burn-in大于样本数
        args.hmc_samples = 10
        args.num_burnin = 20
        with self.assertRaises(ValueError):
            hmc = TraditionalHMC(args)
            hmc.sample()
    def tearDown(self):
        """清理测试环境"""
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()