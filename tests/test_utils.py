import unittest
import sys
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import tempfile
import logging

from codes.utils import (dynamics_fn, traditional_leapfrog, L2_loss,
                         to_pickle, from_pickle, compute_gradients,
                         compute_training_loss, setup_logger,
                         compute_ess, compute_average_ess)
from codes.functions import functions
from codes.get_args import get_args


class TestUtils(unittest.TestCase):
    def setUp(self):
        # 设置基本测试环境
        self.args = get_args()
        self.args.dist_name = '1D_Gauss_mix'  # 使用一维高斯混合作为测试用例
        self.args.input_dim = 2

    def test_dynamics_fn(self):
        test_cases = [
            ('1D_Gauss_mix', 2),
            ('2D_Neal_funnel', 4),
            ('5D_illconditioned_Gaussian', 10),
            ('nD_Rosenbrock', 6),
            ('Allen_Cahn', 50),
            ('Elliptic', 100)
        ]

        for dist_name, input_dim in test_cases:
            with self.subTest(dist_name=dist_name):
                self.args.dist_name = dist_name
                self.args.input_dim = input_dim

                # 创建测试数据
                z = tf.constant([[1.0] * input_dim], dtype=tf.float32)

                # 计算导数
                derivatives = dynamics_fn(lambda x, args: functions(x, args), z, self.args)

                # 检查输出形状
                self.assertEqual(derivatives.shape, z.shape)
                # 检查输出是否为有限值
                self.assertTrue(tf.reduce_all(tf.math.is_finite(derivatives)))

    def test_traditional_leapfrog(self):
        # 基础测试
        z0 = tf.constant([[1.0, 0.5]], dtype=tf.float32)
        t_span = [0.0, 1.0]
        n_steps = 10

        # 执行积分
        traj, derivatives = traditional_leapfrog(
            lambda x, args: functions(x, args),
            z0, t_span, n_steps, self.args
        )

        # 检查输出形状
        self.assertEqual(traj.shape, (n_steps + 1, z0.shape[0], z0.shape[1]))
        self.assertEqual(derivatives.shape, (n_steps + 1, z0.shape[0], z0.shape[1]))

        # 检查轨迹的连续性
        diff = tf.reduce_max(tf.abs(traj[1:] - traj[:-1]))
        self.assertLess(diff, 1.0)

        # 为不同分布设置不同的差异阈值
        diff_thresholds = {
            '1D_Gauss_mix': 1.0,
            '2D_Neal_funnel': 2.0,
            '5D_illconditioned_Gaussian': 10.0,
            'nD_Rosenbrock': 5.0
        }

        # 测试不同分布和维度
        test_cases = [
            ('1D_Gauss_mix', 2),
            ('2D_Neal_funnel', 4),
            ('5D_illconditioned_Gaussian', 10),
            ('nD_Rosenbrock', 6)
        ]

        for dist_name, input_dim in test_cases:
            with self.subTest(dist_name=dist_name):
                self.args.dist_name = dist_name
                self.args.input_dim = input_dim

                # 测试不同形状的输入
                input_shapes = [
                    (tf.constant([[1.0] * input_dim], dtype=tf.float32), "batch_shape"),
                    (tf.constant([1.0] * input_dim, dtype=tf.float32), "single_shape")
                ]

                for z0_test, shape_type in input_shapes:
                    t_span = [0.0, 1.0]
                    n_steps = 10

                    # 测试整数步长
                    traj, derivatives = traditional_leapfrog(
                        lambda x, args: functions(x, args),
                        z0_test, t_span, n_steps, self.args
                    )

                    expected_shape = (n_steps + 1, 1, input_dim) if shape_type == "batch_shape" \
                        else (n_steps + 1, 1, input_dim)
                    self.assertEqual(traj.shape, expected_shape)

                    # 检查轨迹的连续性，使用分布特定的阈值
                    diff = tf.reduce_max(tf.abs(traj[1:] - traj[:-1]))
                    self.assertLess(diff, diff_thresholds[dist_name])

                    # 检查导数的有限性
                    self.assertTrue(tf.reduce_all(tf.math.is_finite(derivatives)))

                    # 检查能量守恒（对于保守系统）
                    if dist_name in ['1D_Gauss_mix', '2D_Neal_funnel']:
                        initial_energy = functions(traj[0], self.args)
                        final_energy = functions(traj[-1], self.args)
                        energy_diff = tf.abs(final_energy - initial_energy)
                        self.assertLess(float(energy_diff), 0.1)

        # 测试浮点数步长 - 应该能正常工作，因为函数内部会进行类型转换
        n_steps_float = 10.0
        try:
            traj, derivatives = traditional_leapfrog(
                lambda x, args: functions(x, args),
                z0, t_span, n_steps_float, self.args
            )
            # 验证结果形状
            self.assertEqual(traj.shape, (int(n_steps_float) + 1, z0.shape[0], z0.shape[1]))
        except Exception as e:
            self.fail(f"Float n_steps raised unexpected error: {e}")

        # 测试时间范围边界情况
        t_span_cases = [
            ([0.0, 1.0], None),  # 正常时间范围
            ([0.0, 1e3], None)  # 大时间范围
        ]

        for t_span_test, expected_error in t_span_cases:
            if expected_error is None:
                try:
                    traj, _ = traditional_leapfrog(
                        lambda x, args: functions(x, args),
                        z0, t_span_test, n_steps, self.args
                    )
                except Exception as e:
                    self.fail(f"t_span={t_span_test} raised unexpected error: {e}")
            else:
                with self.assertRaises(expected_error):
                    traditional_leapfrog(
                        lambda x, args: functions(x, args),
                        z0, t_span_test, n_steps, self.args
                    )

    def test_traditional_leapfrog_invalid_inputs(self):
        """单独测试无效输入的情况"""
        z0 = tf.constant([[1.0, 0.5]], dtype=tf.float32)
        n_steps = 10

        # 测试反向时间范围
        t_span_reverse = [1.0, 0.0]
        with self.assertRaises(tf.errors.InvalidArgumentError):
            traditional_leapfrog(
                lambda x, args: functions(x, args),
                z0, t_span_reverse, n_steps, self.args
            )

        # 测试零长度时间范围
        t_span_zero = [1.0, 1.0]
        with self.assertRaises(tf.errors.InvalidArgumentError):
            traditional_leapfrog(
                lambda x, args: functions(x, args),
                z0, t_span_zero, n_steps, self.args
            )

        # 测试负数步长
        n_steps_negative = -1
        with self.assertRaises(tf.errors.InvalidArgumentError):
            traditional_leapfrog(
                lambda x, args: functions(x, args),
                z0, [0.0, 1.0], n_steps_negative, self.args
            )

        # 测试零步长
        n_steps_zero = 0
        with self.assertRaises(tf.errors.InvalidArgumentError):
            traditional_leapfrog(
                lambda x, args: functions(x, args),
                z0, [0.0, 1.0], n_steps_zero, self.args
            )



    def test_L2_loss(self):
        # 测试L2损失函数
        u = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        v = tf.constant([[1.1, 2.1], [3.1, 4.1]])

        loss = L2_loss(u, v)

        # 检查损失值是否为标量
        self.assertEqual(loss.shape, ())
        # 检查损失值是否为正数
        self.assertGreater(loss, 0)
        # 手动计算预期损失
        expected_loss = ((1-1.1)**2 + (2-2.1)**2 + (3-3.1)**2 + (4-4.1)**2) / 4
        self.assertAlmostEqual(float(loss), expected_loss, places=6)

    def test_pickle_operations(self):
        # 测试pickle存取功能
        test_data = {"test": "data"}

        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # 测试保存
            to_pickle(test_data, tmp.name)
            # 测试加载
            loaded_data = from_pickle(tmp.name)

        self.assertEqual(test_data, loaded_data)

        # 清理临时文件
        os.unlink(tmp.name)

    def test_setup_logger(self):
        # 测试日志设置
        logger = setup_logger("test_logger")

        # 验证logger类型
        self.assertIsInstance(logger, logging.Logger)
        # 验证handler数量
        self.assertEqual(len(logger.handlers), 2)

        # 关闭所有handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        # 清理日志文件
        log_dir = Path('logs')
        if log_dir.exists():
            for log_file in log_dir.glob('test_logger_*.log'):
                log_file.unlink()
            log_dir.rmdir()

    def test_ess_computations(self):
        # 测试不同维度的输入
        test_dims = [2, 4, 6, 8]

        for dim in test_dims:
            samples = tf.random.normal([1, 1000, dim])
            burn_in = 100

            # 测试单维度ESS
            ess_values = compute_ess(samples, burn_in)
            self.assertEqual(len(ess_values), dim // 2)

            # 测试平均ESS
            avg_ess = compute_average_ess(samples, burn_in)
            self.assertIsInstance(avg_ess, float)
            self.assertGreater(avg_ess, 0)

        # 测试奇数维度输入
        odd_dim = 5
        samples_odd = tf.random.normal([1, 1000, odd_dim])

        with self.assertRaises(ValueError) as context:
            ess_values = compute_ess(samples_odd, burn_in)

        self.assertTrue("Input dimension must be even" in str(context.exception))

    def tearDown(self):
        # 清理测试环境
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()