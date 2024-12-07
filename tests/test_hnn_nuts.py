import unittest
import tensorflow as tf
import numpy as np
import os
import sys
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class TestHNNNUTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up environment before all tests"""
        os.chdir(os.path.join(PROJECT_ROOT, 'codes'))
        if str(PROJECT_ROOT / 'codes') not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT / 'codes'))

        global get_model, nuts_hnn_sample, build_tree, get_args
        from codes.hnn_nuts_olm import get_model, nuts_hnn_sample, build_tree
        from codes.get_args import get_args

    def load_model(self, config):
        """Load model with specified configuration and weights

        Args:
            config: Dictionary containing model configuration

        Returns:
            Loaded HNN model with weights

        Raises:
            FileNotFoundError: If weight files not found
        """
        args = self.get_modified_args(config)

        # Verify weight files exist
        weight_path_index = PROJECT_ROOT / 'codes' / 'files' / f"{config['weight_file']}.index"
        weight_path_data = PROJECT_ROOT / 'codes' / 'files' / f"{config['weight_file']}.data-00000-of-00001"

        if not weight_path_index.exists():
            raise FileNotFoundError(f"Weight index file not found: {weight_path_index}")
        if not weight_path_data.exists():
            raise FileNotFoundError(f"Weight data file not found: {weight_path_data}")

        # Create and load model
        model = get_model(args)
        model.load_weights(str(PROJECT_ROOT / 'codes' / 'files' / config['weight_file']))

        return model

    def setUp(self):
        """Set up test environment"""
        self.test_configs = [
            {
                'name': 'nD_Rosenbrock',
                'input_dim': 6,
                'latent_dim': 100,
                'weight_file': 'nD_Rosenbrock100',
            },
            {
                'name': '2D_Neal_funnel',
                'input_dim': 4,
                'latent_dim': 2,
                'weight_file': '2D_Neal_funnel250',
            },
            {
                'name': '5D_illconditioned_Gaussian',
                'input_dim': 10,
                'latent_dim': 5,
                'weight_file': '5D_illconditioned_Gaussian250',
            },
            {
                'name': 'nD_Rosenbrock',
                'input_dim': 20,
                'latent_dim': 10,
                'weight_file': '10D_Rosenbrock250',
            }
        ]

    def get_modified_args(self, config):
        """Get modified arguments

        Args:
            config: Dictionary containing model configuration parameters

        Returns:
            Modified args object
        """
        args = get_args()
        args.dist_name = config['name']
        args.input_dim = config['input_dim']
        args.latent_dim = config['latent_dim']
        # NUTS specific parameters
        args.total_samples = 10
        args.burn_in = 3
        args.nuts_step_size = 0.1
        args.hnn_error_threshold = 1.0  # Increase threshold
        args.leapfrog_error_threshold = 1000.0
        args.n_cooldown = 5
        args.num_chains = 2
        args.save_dir = str(PROJECT_ROOT / 'codes' / 'files' / config['weight_file'])
        args.hidden_dim = 100  # Add hidden layer dimension
        args.nonlinearity = 'sine'  # Add activation function
        args.traditional_only = False  # Add traditional_only attribute
        return args

    def test_nuts_sampling_modes(self):
        """Test both modes of NUTS: HNN and Traditional"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                args = self.get_modified_args(config)
                model = self.load_model(config)

                # Test HNN mode
                samples_hnn, acc_hnn, errors_hnn, grads_hnn = nuts_hnn_sample(
                    model, args, traditional_only=False)

                # Test traditional mode
                samples_trad, acc_trad, errors_trad, grads_trad = nuts_hnn_sample(
                    model, args, traditional_only=True)

                # Verify output dimensions
                expected_shape = (args.num_chains, args.total_samples, args.input_dim)
                self.assertEqual(samples_hnn.shape, expected_shape)
                self.assertEqual(samples_trad.shape, expected_shape)

                # Verify acceptance rates
                self.assertGreater(tf.reduce_mean(acc_hnn), 0.1)
                self.assertGreater(tf.reduce_mean(acc_trad), 0.1)

                # Verify gradient computation counts
                self.assertGreater(grads_trad, 0)
                # Compare gradient counts when in HNN mode
                self.assertLess(grads_hnn, grads_trad)
    def test_energy_conservation(self):
        """Test energy conservation for both HNN and traditional NUTS"""
        # Define energy thresholds for different distributions
        energy_thresholds = {
            'nD_Rosenbrock': 10.0,
            '2D_Neal_funnel': 2.0,
            '5D_illconditioned_Gaussian': 5.0
        }

        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                args = self.get_modified_args(config)
                model = self.load_model(config)
                threshold = energy_thresholds.get(config['name'], 2.0)  # Default threshold

                # Test both HNN and traditional modes
                for traditional_only in [False, True]:
                    mode_name = "Traditional" if traditional_only else "HNN"

                    # Run sampling
                    samples, _, _, _ = nuts_hnn_sample(model, args, traditional_only=traditional_only)

                    # Calculate energies for all samples in first chain
                    energies = []
                    for sample in samples[0]:  # Use first chain
                        energy = float(model.compute_hamiltonian(tf.expand_dims(sample, 0)))
                        energies.append(energy)

                    # Check consecutive energy differences
                    for i in range(len(energies) - 1):
                        energy_diff = abs(energies[i + 1] - energies[i])
                        self.assertLess(
                            energy_diff,
                            threshold,
                            f"{mode_name} mode: Energy difference {energy_diff} exceeds threshold {threshold} "
                            f"for distribution {config['name']}"
                        )

    def test_gradient_counting(self):
        """Test gradient computation counts"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                args = self.get_modified_args(config)
                model = self.load_model(config)

                # Compare gradient counts between modes
                _, _, _, grads_hnn = nuts_hnn_sample(model, args, traditional_only=False)
                _, _, _, grads_trad = nuts_hnn_sample(model, args, traditional_only=True)

                # Verify gradient of grads_hnn < grads_trad
                self.assertGreater(grads_trad, grads_hnn)

    def test_build_tree(self):
        """Comprehensive test for build_tree function"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                args = self.get_modified_args(config)
                model = self.load_model(config)
                state_dim = args.input_dim // 2

                # Test data initialization
                q = tf.zeros(state_dim)
                p = tf.ones(state_dim)
                u = tf.constant(0.5)
                H0 = model.compute_hamiltonian(tf.expand_dims(tf.concat([q, p], 0), 0))

                # Test both HNN and traditional modes
                for traditional_only in [False, True]:
                    mode_name = "Traditional" if traditional_only else "HNN"

                    # Test different tree depths
                    for depth in [0, 1, 2]:
                        # Test both directions
                        for direction in [-1, 1]:
                            results = build_tree(
                                q, p, u, tf.constant(direction), depth,
                                args.nuts_step_size, H0, model,
                                state_dim, args.hnn_error_threshold,
                                args.leapfrog_error_threshold, False, args,
                                traditional_only=traditional_only
                            )

                            # Basic checks
                            self.assertEqual(len(results), 13,
                                             f"Expected 13 return values from build_tree in {mode_name} mode")

                            # Structure checks
                            q_minus, p_minus, q_plus, p_plus, q_prime, p_prime, n_prime, \
                                s_prime, alpha, n_alpha, error, use_leapfrog, grad_count = results

                            # Dimension checks
                            self.assertEqual(q_minus.shape, (state_dim,))
                            self.assertEqual(p_minus.shape, (state_dim,))
                            self.assertEqual(q_plus.shape, (state_dim,))
                            self.assertEqual(p_plus.shape, (state_dim,))
                            self.assertEqual(q_prime.shape, (state_dim,))
                            self.assertEqual(p_prime.shape, (state_dim,))

                            # Value range checks
                            self.assertGreaterEqual(float(n_prime), 0)
                            self.assertGreaterEqual(float(alpha), 0)
                            self.assertLessEqual(float(alpha), 1)
                            self.assertGreaterEqual(float(n_alpha), 0)
                            self.assertGreaterEqual(grad_count, 0)

                            # Mode-specific checks
                            if traditional_only:
                                self.assertTrue(use_leapfrog)

                            # Energy conservation check at depth 0
                            if depth == 0:
                                initial_state = tf.concat([q, p], 0)
                                final_state = tf.concat([q_prime, p_prime], 0)
                                initial_H = float(model.compute_hamiltonian(
                                    tf.expand_dims(initial_state, 0)))
                                final_H = float(model.compute_hamiltonian(
                                    tf.expand_dims(final_state, 0)))
                                energy_diff = abs(final_H - initial_H)

                                # Different thresholds for different distributions
                                threshold = 2.0 if config['name'] == 'nD_Rosenbrock' else 1.0
                                self.assertLess(energy_diff, threshold,
                                                f"{mode_name} mode: Energy difference {energy_diff} "
                                                f"exceeds threshold at depth 0")

                            # Check tree growth direction
                            if direction == -1:
                                # For leftward growth, q_minus should differ from initial q
                                self.assertTrue(tf.reduce_any(tf.not_equal(q_minus, q)))
                            else:
                                # For rightward growth, q_plus should differ from initial q
                                self.assertTrue(tf.reduce_any(tf.not_equal(q_plus, q)))

                            # Check trajectory validity
                            self.assertTrue(tf.reduce_all(tf.math.is_finite(q_prime)))
                            self.assertTrue(tf.reduce_all(tf.math.is_finite(p_prime)))

    def test_nuts_error_handling(self):
        """Test error handling"""
        config = self.test_configs[0]
        args = self.get_modified_args(config)
        model = self.load_model(config)

        # Test invalid step size
        args.nuts_step_size = -0.1
        with self.assertRaises(ValueError):
            nuts_hnn_sample(model, args)

        # Restore step size and test invalid sample count
        args.nuts_step_size = 0.1
        args.total_samples = 0
        with self.assertRaises(ValueError):
            nuts_hnn_sample(model, args)

        # Test invalid burn-in period
        args.total_samples = 10
        args.burn_in = -1
        with self.assertRaises(ValueError):
            nuts_hnn_sample(model, args)

        # Test burn-in greater than total samples
        args.burn_in = 15  # Greater than total_samples
        with self.assertRaises(ValueError):
            nuts_hnn_sample(model, args)

        # Test invalid chain count
        args.burn_in = 3
        args.num_chains = 0
        with self.assertRaises(ValueError):
            nuts_hnn_sample(model, args)

        # Test invalid cooldown period
        args.num_chains = 2
        args.n_cooldown = -1
        with self.assertRaises(ValueError):
            nuts_hnn_sample(model, args)

    def test_tensorflow_specific_errors(self):
        """Test TensorFlow specific errors"""
        config = self.test_configs[0]
        args = self.get_modified_args(config)
        model = self.load_model(config)

        # Test cases that might trigger TensorFlow errors
        args.input_dim = -1  # This may cause TensorFlow dimension error
        with self.assertRaises(tf.errors.InvalidArgumentError):
            nuts_hnn_sample(model, args)

    def tearDown(self):
        """Clean up test environment"""
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()