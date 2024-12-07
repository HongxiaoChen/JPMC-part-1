import unittest
import tensorflow as tf
import numpy as np
import os
import sys
from pathlib import Path
from codes.functions import functions

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class TestHNNHMC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up environment before all tests"""
        # Change to codes directory
        os.chdir(os.path.join(PROJECT_ROOT, 'codes'))
        # Add codes directory to Python path
        if str(PROJECT_ROOT / 'codes') not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT / 'codes'))

        # Import required modules after changing working directory
        global HNNSampler, get_args, MLP, HNN
        from codes.hnn_hmc import HNNSampler
        from codes.get_args import get_args
        from codes.nn_models import MLP
        from codes.hnn import HNN

    def setUp(self):
        """Set up test environment"""
        # Configure test cases
        self.test_configs = [
            {
                'name': 'nD_Rosenbrock',
                'input_dim': 6,
                'latent_dim': 100,
                'weight_file': 'nD_Rosenbrock100',  # Remove extension
                'settings': [
                    {'step_size': 0.01, 'trajectory_length': 1},
                    {'step_size': 0.05, 'trajectory_length': 5},
                ]
            },
            {
                'name': '2D_Neal_funnel',
                'input_dim': 4,
                'latent_dim': 2,
                'weight_file': '2D_Neal_funnel250',  # Remove extension
                'settings': [
                    {'step_size': 0.01, 'trajectory_length': 1},
                    {'step_size': 0.05, 'trajectory_length': 5},
                ]
            },
            {
                'name': '5D_illconditioned_Gaussian',
                'input_dim': 10,
                'latent_dim': 5,
                'weight_file': '5D_illconditioned_Gaussian250',  # Remove extension
                'settings': [
                    {'step_size': 0.01, 'trajectory_length': 1},
                    {'step_size': 0.05, 'trajectory_length': 5},
                ]
            },
            {
                'name': 'nD_Rosenbrock',
                'input_dim': 20,
                'latent_dim': 10,
                'weight_file': '10D_Rosenbrock250',  # Remove extension
                'settings': [
                    {'step_size': 0.01, 'trajectory_length': 1},
                    {'step_size': 0.05, 'trajectory_length': 5},
                ]
            }
        ]

    def get_modified_args(self, config, setting):
        """Get modified arguments

        Args:
            config: Dictionary containing model configuration
            setting: Dictionary containing HMC settings

        Returns:
            Modified args object
        """
        args = get_args()
        args.dist_name = config['name']
        args.input_dim = config['input_dim']
        args.latent_dim = config['latent_dim']
        args.hmc_step_size = setting['step_size']
        args.trajectory_length = setting['trajectory_length']
        # Use smaller sample size to speed up testing
        args.hmc_samples = 100
        args.num_burnin = 50
        args.num_chains = 2
        # Build complete path using PROJECT_ROOT
        args.save_dir = str(PROJECT_ROOT / 'codes' / 'files' / config['weight_file'])
        return args

    def test_hnn_hmc_initialization(self):
        """Test HNN-HMC initialization"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                # Check weight files (verify .index file exists)
                weight_path_index = PROJECT_ROOT / 'codes' / 'files' / f"{config['weight_file']}.index"
                weight_path_data = PROJECT_ROOT / 'codes' / 'files' / f"{config['weight_file']}.data-00000-of-00001"

                print(f"Checking weight files:")
                print(f"Index file: {weight_path_index}")
                print(f"Data file: {weight_path_data}")

                # Verify both files exist
                self.assertTrue(weight_path_index.exists(),
                                f"Weight index file not found: {weight_path_index}")
                self.assertTrue(weight_path_data.exists(),
                                f"Weight data file not found: {weight_path_data}")

                args = self.get_modified_args(config, config['settings'][0])
                sampler = HNNSampler(args)

                # Check initialized parameters
                self.assertEqual(sampler.state_dim, args.input_dim // 2)
                self.assertEqual(sampler.total_dim, args.input_dim)
                self.assertEqual(sampler.step_size, args.hmc_step_size)
                self.assertEqual(sampler.trajectory_length, args.trajectory_length)
                self.assertEqual(len(sampler.trajectories), 0)

                # Verify HNN model structure
                self.assertEqual(sampler.hnn_model.__class__.__name__, 'HNN')
                self.assertEqual(sampler.hnn_model.differentiable_model.__class__.__name__, 'MLP')
                self.assertEqual(sampler.hnn_model.dim, args.input_dim // 2)

    def test_hnn_hmc_sampling(self):
        """Test HNN-HMC sampling"""
        for config in self.test_configs:
            for setting in config['settings']:
                with self.subTest(distribution=config['name'],
                                  step_size=setting['step_size'],
                                  trajectory_length=setting['trajectory_length']):
                    args = self.get_modified_args(config, setting)
                    sampler = HNNSampler(args)

                    # Perform sampling
                    samples, acceptance = sampler.sample()

                    # Check output dimensions
                    expected_shape = (args.num_chains, args.hmc_samples, args.input_dim)
                    self.assertEqual(samples.shape, expected_shape)

                    # Check acceptance dimensions
                    expected_acceptance_shape = (args.num_chains,
                                                 args.hmc_samples - args.num_burnin)
                    self.assertEqual(acceptance.shape, expected_acceptance_shape)

                    # Check sample finiteness
                    self.assertTrue(tf.reduce_all(tf.math.is_finite(samples)))

                    # Check acceptance rates within valid range
                    self.assertTrue(tf.reduce_all(acceptance >= 0))
                    self.assertTrue(tf.reduce_all(acceptance <= 1))

                    # Check mean acceptance rate
                    mean_acceptance = tf.reduce_mean(acceptance)
                    self.assertGreater(float(mean_acceptance), 0.1)

    def test_trajectory_storage(self):
        """Test trajectory storage functionality"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                args = self.get_modified_args(config, config['settings'][0])
                sampler = HNNSampler(args)

                # Perform sampling
                samples, _ = sampler.sample()

                # Get stored trajectories
                trajectories = sampler.get_trajectories()

                # Check trajectory count
                expected_trajectory_count = args.num_chains * (args.hmc_samples - args.num_burnin)
                self.assertEqual(len(trajectories), expected_trajectory_count)

                # Check trajectory dimensions
                n_steps = args.trajectory_length * int(1 / args.hmc_step_size)
                expected_trajectory_shape = (n_steps + 1, 1, args.input_dim)
                self.assertEqual(trajectories[0].shape, expected_trajectory_shape)

    def test_energy_conservation(self):
        """Test energy conservation"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                args = self.get_modified_args(config, config['settings'][0])
                sampler = HNNSampler(args)

                # Perform sampling
                sampler.sample()

                # Get trajectories
                trajectories = sampler.get_trajectories()

                # Check energy change for each trajectory
                for traj in trajectories:
                    initial_state = tf.convert_to_tensor(traj[0], dtype=tf.float32)
                    final_state = tf.convert_to_tensor(traj[-1], dtype=tf.float32)

                    initial_H = float(sampler.hnn_model.compute_hamiltonian(initial_state))
                    final_H = float(sampler.hnn_model.compute_hamiltonian(final_state))

                    # Energy change should be small when computed with HNN
                    energy_diff = abs(final_H - initial_H)
                    self.assertLess(energy_diff, 1.0)

    def test_error_handling(self):
        """Test error handling"""
        config = self.test_configs[0]
        args = self.get_modified_args(config, config['settings'][0])

        # Test invalid step_size
        args.hmc_step_size = -0.1
        with self.assertRaises(ValueError):
            HNNSampler(args)

        # Test invalid trajectory_length
        args.hmc_step_size = 0.01
        args.trajectory_length = -1
        with self.assertRaises(ValueError):
            HNNSampler(args)

        # Test invalid dimension
        args.trajectory_length = 5
        args.input_dim = -1
        with self.assertRaises(ValueError):
            HNNSampler(args)

        # Test invalid latent_dim
        args.input_dim = 6
        args.latent_dim = -1
        with self.assertRaises(ValueError):
            HNNSampler(args)

        # Test non-existent weight file
        args.latent_dim = 100
        args.save_dir = "nonexistent_path"
        with self.assertRaises(tf.errors.NotFoundError):
            HNNSampler(args)

    def tearDown(self):
        """Clean up test environment"""
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()