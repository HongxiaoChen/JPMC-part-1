import unittest
import tensorflow as tf
import numpy as np
import os
import sys
from pathlib import Path

# Get the project root path
PROJECT_ROOT = Path(__file__).parent.parent

# Add codes directory to Python path
sys.path.append(str(PROJECT_ROOT))

from codes.utils import compute_training_loss, compute_gradients, leapfrog
from codes.hnn import HNN
from codes.nn_models import MLP
from codes.get_args import get_args


class TestHNNUtils(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
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
        """Load model with specified configuration

        Args:
            config: Dictionary containing model configuration parameters

        Returns:
            Model instance with loaded weights
        """
        self.args.input_dim = config['input_dim']
        self.args.latent_dim = config['latent_dim']
        self.args.dist_name = config['dist_name']

        # Create MLP model
        differentiable_model = MLP(
            input_dim=self.args.input_dim,
            hidden_dim=self.args.hidden_dim,
            latent_dim=self.args.latent_dim,
            nonlinearity=self.args.nonlinearity
        )

        # Create HNN model
        model = HNN(
            input_dim=self.args.input_dim,
            differentiable_model=differentiable_model
        )

        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adam(self.args.learn_rate))

        # Build complete weight file path
        weight_path = os.path.join(PROJECT_ROOT, 'codes', 'files', config['name'])
        try:
            # Check if weight file exists
            if os.path.exists(weight_path + '.index'):
                model.load_weights(weight_path)
                print(f"Successfully loaded weights for {config['name']}")
            else:
                print(f"Warning: Weight file not found for {config['name']}")
        except Exception as e:
            print(f"Error loading weights for {config['name']}: {str(e)}")

        return model

    def test_compute_gradients(self):
        """Test gradient computation function"""
        for config in self.model_configs:
            with self.subTest(model_name=config['name']):
                model = self.load_model(config)

                # Create test data
                batch_size = 4
                x = tf.random.normal([batch_size, config['input_dim']],
                                     mean=0.0, stddev=0.1)  # Reduce initial value range

                # Compute gradients
                grad_p, grad_q = compute_gradients(model, x)

                # Check output shapes
                self.assertEqual(grad_p.shape, (batch_size, config['input_dim'] // 2))
                self.assertEqual(grad_q.shape, (batch_size, config['input_dim'] // 2))

                # Check gradient finiteness
                self.assertTrue(tf.reduce_all(tf.math.is_finite(grad_p)))
                self.assertTrue(tf.reduce_all(tf.math.is_finite(grad_q)))

    def test_compute_training_loss(self):
        """Test training loss computation function"""
        for config in self.model_configs:
            with self.subTest(model_name=config['name']):
                model = self.load_model(config)

                # Create test data
                batch_size = 4
                batch_data = tf.random.normal([batch_size, config['input_dim']],
                                              mean=0.0, stddev=0.1)
                dq_true = tf.random.normal([batch_size, config['input_dim'] // 2],
                                           mean=0.0, stddev=0.1)
                dp_true = tf.random.normal([batch_size, config['input_dim'] // 2],
                                           mean=0.0, stddev=0.1)
                time_derivatives = (dq_true, dp_true)

                # Compute loss
                loss = compute_training_loss(model, batch_data, time_derivatives)

                # Check loss value
                self.assertIsInstance(loss, tf.Tensor)
                self.assertEqual(loss.shape, ())
                self.assertTrue(tf.math.is_finite(loss))
                self.assertGreaterEqual(float(loss), 0.0)

    def test_leapfrog(self):
        """Test leapfrog integrator"""
        for config in self.model_configs:
            with self.subTest(model_name=config['name']):
                model = self.load_model(config)

                # Basic test - ensure input is 2D tensor
                z0 = tf.constant([[1.0, 0.5] * (config['input_dim'] // 2)], dtype=tf.float32)
                t_span = [0.0, 1.0]
                n_steps = 10

                # Perform integration
                t, z = leapfrog(model, z0, t_span, n_steps)

                # Check output shapes
                self.assertEqual(t.shape, (n_steps + 1,))
                self.assertEqual(z.shape, (n_steps + 1, z0.shape[0], z0.shape[1]))

                # Check time step monotonicity
                self.assertTrue(tf.reduce_all(t[1:] > t[:-1]))

                # Check trajectory continuity
                diff = tf.reduce_max(tf.abs(z[1:] - z[:-1]))
                self.assertLess(float(diff), 10.0)

                # Set different thresholds for different distributions
                diff_thresholds = {
                    '1D_Gauss_mix': 1.0,
                    '2D_Neal_funnel': 2.0,
                    '5D_illconditioned_Gaussian': 10.0,
                    'nD_Rosenbrock': 5.0
                }

                # Test different input shapes
                input_shapes = [
                    (tf.constant([[1.0] * config['input_dim']], dtype=tf.float32), "batch_shape"),
                    (tf.reshape(tf.constant([1.0] * config['input_dim'], dtype=tf.float32),
                                [1, config['input_dim']]), "reshaped_shape")
                ]

                for z0_test, shape_type in input_shapes:
                    # Test integer steps
                    t, z = leapfrog(model, z0_test, t_span, n_steps)

                    # Check output shapes
                    self.assertEqual(z.shape, (n_steps + 1, 1, config['input_dim']))

                    # Check trajectory continuity
                    diff = tf.reduce_max(tf.abs(z[1:] - z[:-1]))
                    self.assertLess(diff, diff_thresholds[config['dist_name']])

                    # Check energy conservation (for conservative systems)
                    if config['dist_name'] in ['1D_Gauss_mix', '2D_Neal_funnel']:
                        initial_energy = model.compute_hamiltonian(z[0])
                        final_energy = model.compute_hamiltonian(z[-1])
                        energy_diff = tf.abs(final_energy - initial_energy)
                        self.assertLess(float(energy_diff), 0.5)

                # Test float number of steps
                n_steps_float = 10.0
                try:
                    t, z = leapfrog(model, z0, t_span, n_steps_float)
                    # Validate result shape
                    self.assertEqual(z.shape[0], int(n_steps_float) + 1)
                except Exception as e:
                    self.fail(f"Float n_steps={n_steps_float} raised unexpected error: {e}")

                # Test time span edge cases
                t_span_cases = [
                    ([0.0, 1.0], None),  # Normal time span
                    ([0.0, 1e3], None),  # Large time span
                    ([0.0, 0.1], None)  # Small time span
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
        """Test MLP model"""
        for config in self.model_configs:
            with self.subTest(model_name=config['name']):
                # Test different activation functions
                for nonlinearity in ['sine', 'tanh', 'relu']:
                    mlp = MLP(
                        input_dim=config['input_dim'],
                        hidden_dim=self.args.hidden_dim,
                        latent_dim=config['latent_dim'],
                        nonlinearity=nonlinearity
                    )

                    # Test forward pass
                    x = tf.random.normal([4, config['input_dim']])
                    output = mlp(x)

                    # Check output shape
                    self.assertEqual(output.shape, (4, config['latent_dim']))

                    # Check output finiteness
                    self.assertTrue(tf.reduce_all(tf.math.is_finite(output)))

                # Test invalid activation function
                with self.assertRaises(ValueError):
                    MLP(
                        input_dim=config['input_dim'],
                        hidden_dim=self.args.hidden_dim,
                        latent_dim=config['latent_dim'],
                        nonlinearity='invalid_activation'
                    )

    def test_hnn(self):
        """Test HNN model"""
        for config in self.model_configs:
            with self.subTest(model_name=config['name']):
                model = self.load_model(config)

                # Test kinetic energy computation
                p = tf.random.normal([4, config['input_dim'] // 2])
                kinetic = model.kinetic_energy(p)
                self.assertEqual(kinetic.shape, (4,))
                self.assertTrue(tf.reduce_all(kinetic >= 0))  # Kinetic energy should be non-negative

                # Test Hamiltonian computation
                x = tf.random.normal([4, config['input_dim']])
                H = model.compute_hamiltonian(x)
                self.assertEqual(H.shape, (4, 1))
                self.assertTrue(tf.reduce_all(tf.math.is_finite(H)))

                # Test time derivative computation
                derivatives = model.time_derivative(x)
                self.assertEqual(derivatives.shape, x.shape)
                self.assertTrue(tf.reduce_all(tf.math.is_finite(derivatives)))

                # Test mass matrix
                # Default mass matrix
                self.assertEqual(model.M.shape, (config['input_dim'] // 2,))
                self.assertTrue(tf.reduce_all(model.M > 0))  # Mass should be positive

                # Custom mass matrix
                custom_mass = tf.ones(config['input_dim'] // 2) * 2.0
                model_custom_mass = HNN(
                    input_dim=config['input_dim'],
                    differentiable_model=model.differentiable_model,
                    mass_matrix=custom_mass
                )
                self.assertTrue(tf.reduce_all(tf.equal(model_custom_mass.M, custom_mass)))

                # Validate Hamilton equations
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    H = model.compute_hamiltonian(x)

                dH = tape.gradient(H, x)
                derivatives = model.time_derivative(x)

                # Verify dq/dt = ∂H/∂p
                dq_dt = derivatives[:, :config['input_dim'] // 2]
                dH_dp = dH[:, config['input_dim'] // 2:]
                self.assertTrue(tf.reduce_all(tf.abs(dq_dt - dH_dp) < 1e-5))

                # Verify dp/dt = -∂H/∂q
                dp_dt = derivatives[:, config['input_dim'] // 2:]
                dH_dq = dH[:, :config['input_dim'] // 2]
                self.assertTrue(tf.reduce_all(tf.abs(dp_dt + dH_dq) < 1e-5))

    def test_hnn_error_handling(self):
        """Test HNN error handling"""
        config = self.model_configs[0]
        model = self.load_model(config)

        # Test dimension mismatch
        invalid_input = tf.random.normal([4, config['input_dim'] + 1])
        with self.assertRaises(ValueError):
            model.time_derivative(invalid_input)

        # Test gradient computation failure case
        # This needs to simulate a case that will cause gradient computation to fail
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
        """Clean up test environment"""
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()