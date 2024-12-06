import unittest
import tensorflow as tf
import numpy as np
import os
import sys
from pathlib import Path

# 获取项目根目录的路径
PROJECT_ROOT = Path(__file__).parent.parent

# 将codes目录添加到Python路径
sys.path.append(str(PROJECT_ROOT))

from codes.utils import compute_training_loss, compute_gradients, leapfrog
from codes.hnn import HNN
from codes.nn_models import MLP
from codes.get_args import get_args


class TestHNNUtils(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        self.args = get_args()
        self.model_configs = [
            {
                'name': 'nD_Rosenbrock100',
                'input_dim': 6,
                'latent_dim': 100,
                'dist_name': 'nD_Rosenbrock'
            },
            {
                'name': '2D_Neal_funnel250',
                'input_dim': 4,
                'latent_dim': 2,
                'dist_name': '2D_Neal_funnel'
            },
            {
                'name': '5D_illconditioned_Gaussian250',
                'input_dim': 10,
                'latent_dim': 5,
                'dist_name': '5D_illconditioned_Gaussian'
            },
            {
                'name': '10D_Rosenbrock250',
                'input_dim': 20,
                'latent_dim': 10,
                'dist_name': 'nD_Rosenbrock'
            }
        ]

    def load_model(self, config):
        """加载指定配置的模型"""
        self.args.input_dim = config['input_dim']
        self.args.latent_dim = config['latent_dim']
        self.args.dist_name = config['dist_name']

        # 创建MLP模型
        differentiable_model = MLP(
            input_dim=self.args.input_dim,
            hidden_dim=self.args.hidden_dim,
            latent_dim=self.args.latent_dim,
            nonlinearity=self.args.nonlinearity
        )

        # 创建HNN模型
        model = HNN(
            input_dim=self.args.input_dim,
            differentiable_model=differentiable_model
        )

        # 编译模型
        model.compile(optimizer=tf.keras.optimizers.Adam(self.args.learn_rate))

        # 构建完整的权重文件路径
        weight_path = os.path.join(PROJECT_ROOT, 'codes', 'files', config['name'])
        try:
            # 检查权重文件是否存在
            if os.path.exists(weight_path + '.index'):
                model.load_weights(weight_path)
                print(f"Successfully loaded weights for {config['name']}")
            else:
                print(f"Warning: Weight file not found for {config['name']}")
        except Exception as e:
            print(f"Error loading weights for {config['name']}: {str(e)}")

        return model

    def test_compute_gradients(self):
        """测试计算梯度函数"""
        for config in self.model_configs:
            with self.subTest(model_name=config['name']):
                model = self.load_model(config)

                # 创建测试数据
                batch_size = 4
                x = tf.random.normal([batch_size, config['input_dim']],
                                     mean=0.0, stddev=0.1)  # 减小初始值范围

                # 计算梯度
                grad_p, grad_q = compute_gradients(model, x)

                # 检查输出形状
                self.assertEqual(grad_p.shape, (batch_size, config['input_dim'] // 2))
                self.assertEqual(grad_q.shape, (batch_size, config['input_dim'] // 2))

                # 检查梯度的有限性
                self.assertTrue(tf.reduce_all(tf.math.is_finite(grad_p)))
                self.assertTrue(tf.reduce_all(tf.math.is_finite(grad_q)))

    def test_compute_training_loss(self):
        """测试训练损失计算函数"""
        for config in self.model_configs:
            with self.subTest(model_name=config['name']):
                model = self.load_model(config)

                # 创建测试数据
                batch_size = 4
                batch_data = tf.random.normal([batch_size, config['input_dim']],
                                              mean=0.0, stddev=0.1)
                dq_true = tf.random.normal([batch_size, config['input_dim'] // 2],
                                           mean=0.0, stddev=0.1)
                dp_true = tf.random.normal([batch_size, config['input_dim'] // 2],
                                           mean=0.0, stddev=0.1)
                time_derivatives = (dq_true, dp_true)

                # 计算损失
                loss = compute_training_loss(model, batch_data, time_derivatives)

                # 检查损失值
                self.assertIsInstance(loss, tf.Tensor)
                self.assertEqual(loss.shape, ())
                self.assertTrue(tf.math.is_finite(loss))
                self.assertGreaterEqual(float(loss), 0.0)

    def test_leapfrog(self):
        """测试leapfrog积分器"""
        for config in self.model_configs:
            with self.subTest(model_name=config['name']):
                model = self.load_model(config)

                # 基础测试 - 确保输入是2D张量
                z0 = tf.constant([[1.0, 0.5] * (config['input_dim'] // 2)], dtype=tf.float32)
                t_span = [0.0, 1.0]
                n_steps = 10

                # 执行积分
                t, z = leapfrog(model, z0, t_span, n_steps)

                # 检查输出形状
                self.assertEqual(t.shape, (n_steps + 1,))
                self.assertEqual(z.shape, (n_steps + 1, z0.shape[0], z0.shape[1]))

                # 检查时间步长的单调性
                self.assertTrue(tf.reduce_all(t[1:] > t[:-1]))

                # 检查轨迹的连续性
                diff = tf.reduce_max(tf.abs(z[1:] - z[:-1]))
                self.assertLess(float(diff), 10.0)

                # 为不同分布设置不同的差异阈值
                diff_thresholds = {
                    '1D_Gauss_mix': 1.0,
                    '2D_Neal_funnel': 2.0,
                    '5D_illconditioned_Gaussian': 10.0,
                    'nD_Rosenbrock': 5.0
                }

                # 测试不同形状的输入
                input_shapes = [
                    (tf.constant([[1.0] * config['input_dim']], dtype=tf.float32), "batch_shape"),
                    (tf.reshape(tf.constant([1.0] * config['input_dim'], dtype=tf.float32),
                                [1, config['input_dim']]), "reshaped_shape")
                ]

                for z0_test, shape_type in input_shapes:
                    # 测试整数步长
                    t, z = leapfrog(model, z0_test, t_span, n_steps)

                    # 检查输出形状
                    self.assertEqual(z.shape, (n_steps + 1, 1, config['input_dim']))

                    # 检查轨迹的连续性
                    diff = tf.reduce_max(tf.abs(z[1:] - z[:-1]))
                    self.assertLess(diff, diff_thresholds[config['dist_name']])

                    # 检查能量守恒（对于保守系统）
                    if config['dist_name'] in ['1D_Gauss_mix', '2D_Neal_funnel']:
                        initial_energy = model.compute_hamiltonian(z[0])
                        final_energy = model.compute_hamiltonian(z[-1])
                        energy_diff = tf.abs(final_energy - initial_energy)
                        self.assertLess(float(energy_diff), 0.5)

                # 测试浮点数步长
                n_steps_float = 10.0
                try:
                    t, z = leapfrog(model, z0, t_span, n_steps_float)
                    # 验证结果形状
                    self.assertEqual(z.shape[0], int(n_steps_float) + 1)
                except Exception as e:
                    self.fail(f"Float n_steps={n_steps_float} raised unexpected error: {e}")

                # 测试时间范围边界情况
                t_span_cases = [
                    ([0.0, 1.0], None),  # 正常时间范围
                    ([0.0, 1e3], None),  # 大时间范围
                    ([0.0, 0.1], None)  # 小时间范围
                ]

                for t_span_test, expected_error in t_span_cases:
                    if expected_error is None:
                        try:
                            t, z = leapfrog(model, z0, t_span_test, n_steps)
                            self.assertEqual(t.shape[0], n_steps + 1)
                            self.assertEqual(z.shape[0], n_steps + 1)
                        except Exception as e:
                            self.fail(f"t_span={t_span_test} raised unexpected error: {e}")

    def test_mlp(self):
        """测试MLP模型"""
        for config in self.model_configs:
            with self.subTest(model_name=config['name']):
                # 测试不同的激活函数
                for nonlinearity in ['sine', 'tanh', 'relu']:
                    mlp = MLP(
                        input_dim=config['input_dim'],
                        hidden_dim=self.args.hidden_dim,
                        latent_dim=config['latent_dim'],
                        nonlinearity=nonlinearity
                    )

                    # 测试前向传播
                    x = tf.random.normal([4, config['input_dim']])
                    output = mlp(x)

                    # 检查输出形状
                    self.assertEqual(output.shape, (4, config['latent_dim']))

                    # 检查输出的有限性
                    self.assertTrue(tf.reduce_all(tf.math.is_finite(output)))

                # 测试无效的激活函数
                with self.assertRaises(ValueError):
                    MLP(
                        input_dim=config['input_dim'],
                        hidden_dim=self.args.hidden_dim,
                        latent_dim=config['latent_dim'],
                        nonlinearity='invalid_activation'
                    )

    def test_hnn(self):
        """测试HNN模型"""
        for config in self.model_configs:
            with self.subTest(model_name=config['name']):
                model = self.load_model(config)

                # 测试动能计算
                p = tf.random.normal([4, config['input_dim'] // 2])
                kinetic = model.kinetic_energy(p)
                self.assertEqual(kinetic.shape, (4,))
                self.assertTrue(tf.reduce_all(kinetic >= 0))  # 动能应该非负

                # 测试Hamiltonian计算
                x = tf.random.normal([4, config['input_dim']])
                H = model.compute_hamiltonian(x)
                self.assertEqual(H.shape, (4, 1))
                self.assertTrue(tf.reduce_all(tf.math.is_finite(H)))

                # 测试时间导数计算
                derivatives = model.time_derivative(x)
                self.assertEqual(derivatives.shape, x.shape)
                self.assertTrue(tf.reduce_all(tf.math.is_finite(derivatives)))

                # 测试质量矩阵
                # 默认质量矩阵
                self.assertEqual(model.M.shape, (config['input_dim'] // 2,))
                self.assertTrue(tf.reduce_all(model.M > 0))  # 质量应该为正

                # 自定义质量矩阵
                custom_mass = tf.ones(config['input_dim'] // 2) * 2.0
                model_custom_mass = HNN(
                    input_dim=config['input_dim'],
                    differentiable_model=model.differentiable_model,
                    mass_matrix=custom_mass
                )
                self.assertTrue(tf.reduce_all(tf.equal(model_custom_mass.M, custom_mass)))

                # 验证Hamilton方程
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    H = model.compute_hamiltonian(x)

                dH = tape.gradient(H, x)
                derivatives = model.time_derivative(x)

                # 验证 dq/dt = ∂H/∂p
                dq_dt = derivatives[:, :config['input_dim'] // 2]
                dH_dp = dH[:, config['input_dim'] // 2:]
                self.assertTrue(tf.reduce_all(tf.abs(dq_dt - dH_dp) < 1e-5))

                # 验证 dp/dt = -∂H/∂q
                dp_dt = derivatives[:, config['input_dim'] // 2:]
                dH_dq = dH[:, :config['input_dim'] // 2]
                self.assertTrue(tf.reduce_all(tf.abs(dp_dt + dH_dq) < 1e-5))

    def test_hnn_error_handling(self):
        """测试HNN错误处理"""
        config = self.model_configs[0]
        model = self.load_model(config)

        # 测试维度不匹配
        invalid_input = tf.random.normal([4, config['input_dim'] + 1])
        with self.assertRaises(ValueError):
            model.time_derivative(invalid_input)

        # 测试梯度计算失败的情况
        # 这需要模拟一个会导致梯度计算失败的情况
        class BadModel(tf.keras.Model):
            def call(self, x):
                return tf.zeros_like(x)[:, 0:1]

        bad_model = HNN(
            input_dim=config['input_dim'],
            differentiable_model=BadModel()
        )

        with self.assertRaises(ValueError):
            bad_model.time_derivative(tf.random.normal([4, config['input_dim']]))

    def tearDown(self):
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()