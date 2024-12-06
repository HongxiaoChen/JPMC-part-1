import unittest
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from codes.functions import nearest_neighbor_derivative, compute_f_hat_with_nearest_neighbor, f_obs


class TestFunctions(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        # 创建测试数据
        self.n_points = 50
        self.x_samples = tf.random.uniform((1, self.n_points), 0, 3)
        self.y_samples = tf.random.uniform((1, self.n_points), 0, 3)
        self.g_values = tf.random.normal((1, self.n_points))

    def test_nearest_neighbor_derivative_dimensions(self):
        """测试nearest_neighbor_derivative的维度处理"""

        # 测试正确输入维度
        d_g_dx, d_g_dy = nearest_neighbor_derivative(
            self.x_samples,
            self.y_samples,
            self.g_values
        )

        # 检查输出维度
        self.assertEqual(d_g_dx.shape, (1, self.n_points))
        self.assertEqual(d_g_dy.shape, (1, self.n_points))

        # 测试无效输入维度 - 缺少batch维度
        invalid_x = tf.random.uniform((self.n_points,))
        invalid_y = tf.random.uniform((self.n_points,))
        invalid_g = tf.random.normal((self.n_points,))

        with self.assertRaises(tf.errors.InvalidArgumentError):
            nearest_neighbor_derivative(invalid_x, invalid_y, invalid_g)

        # 测试不匹配的点数
        invalid_x = tf.random.uniform((1, self.n_points + 1))
        with self.assertRaises(tf.errors.InvalidArgumentError):
            nearest_neighbor_derivative(invalid_x, self.y_samples, self.g_values)

    def test_nearest_neighbor_derivative_values(self):
        """测试nearest_neighbor_derivative的计算结果"""
        d_g_dx, d_g_dy = nearest_neighbor_derivative(
            self.x_samples,
            self.y_samples,
            self.g_values
        )

        # 检查输出的有限性
        self.assertTrue(tf.reduce_all(tf.math.is_finite(d_g_dx)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(d_g_dy)))

    def test_compute_f_hat_dimensions(self):
        """测试compute_f_hat_with_nearest_neighbor的维度处理"""
        # 获取样本数据
        f_obs_values, x_samples, y_samples = f_obs()

        # 计算u_x和u_y
        u_x = tf.cos(2 * x_samples) * 2
        u_y = tf.cos(2 * y_samples) * 2

        # 创建测试q值
        q = tf.random.normal((1, self.n_points))

        # 测试正确输入维度
        f_hat = compute_f_hat_with_nearest_neighbor(
            x_samples, y_samples, q, u_x, u_y
        )

        # 检查输出维度
        self.assertEqual(f_hat.shape, (1, self.n_points))

        # 测试无效输入维度 - 缺少batch维度
        invalid_x = tf.random.uniform((self.n_points,))
        invalid_y = tf.random.uniform((self.n_points,))
        invalid_q = tf.random.normal((self.n_points,))
        invalid_u_x = tf.cos(2 * invalid_x) * 2
        invalid_u_y = tf.cos(2 * invalid_y) * 2

        with self.assertRaises(tf.errors.InvalidArgumentError):
            compute_f_hat_with_nearest_neighbor(
                invalid_x, invalid_y, invalid_q, invalid_u_x, invalid_u_y
            )

        # 测试不匹配的点数
        invalid_x = tf.random.uniform((1, self.n_points + 1))
        with self.assertRaises(tf.errors.InvalidArgumentError):
            compute_f_hat_with_nearest_neighbor(
                invalid_x, y_samples, q, u_x, u_y
            )

    def test_compute_f_hat_values(self):
        """测试compute_f_hat_with_nearest_neighbor的计算结果"""
        # 获取样本数据
        f_obs_values, x_samples, y_samples = f_obs()

        # 计算u_x和u_y
        u_x = tf.cos(2 * x_samples) * 2
        u_y = tf.cos(2 * y_samples) * 2

        # 创建测试q值
        q = tf.random.normal((1, self.n_points))

        # 计算f_hat
        f_hat = compute_f_hat_with_nearest_neighbor(
            x_samples, y_samples, q, u_x, u_y
        )

        # 检查输出的有限性
        self.assertTrue(tf.reduce_all(tf.math.is_finite(f_hat)))

        # 检查输出是否在裁剪范围内
        self.assertTrue(tf.reduce_all(tf.abs(f_hat) <= 200.0))

    def tearDown(self):
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()