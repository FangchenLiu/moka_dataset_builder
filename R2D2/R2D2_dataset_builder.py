from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from r2d2.trajectory_utils.misc import load_trajectory
from r2d2.data_loading.trajectory_sampler import crawler


class R2D2(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'exterior_image_1_left': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Exterior camera 1 left viewpoint',
                        ),
                        'exterior_image_1_right': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Exterior camera 1 right viewpoint'
                        ),
                        'exterior_image_2_left': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Exterior camera 2 left viewpoint'
                        ),
                        'exterior_image_2_right': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Exterior camera 2 right viewpoint'
                        ),
                        'wrist_image_left': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB left viewpoint',
                        ),
                        'wrist_image_right': tfds.features.Image(
                            shape=(256, 256, 3),
                            dytype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB right viewpoint'
                        ),
                        'cartesian_position': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Robot Cartesian state',
                        ),
                        'gripper_position': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float32,
                            doc='Gripper position statae',
                        ),
                        'joint_position': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Joint position state'
                        )
                    }),
                    'action': tfds.features.FeaturesDict({
                        'cartesian_position': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Commanded Cartesian position'
                        ),
                        'cartesian_velocity': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Commanded Cartesian velocity'
                        ),
                        'gripper_position': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float32,
                            doc='Commanded gripper position'
                        ),
                        'gripper_velocity': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float32,
                            doc='Commanded gripper velocity'
                        ),
                        'joint_position': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Commanded joint position'
                        ),
                        'joint_velocity': tdfs features.Tensor(
                            shape=(7,)
                            dtype=np.float32,
                            doc='Commanded joint velocity'
                        )
                    }),
                    tfds.features.Tensor(
                        shape=(10,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/train/episode_*.npy'),
            'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    # Remove Extra Transitions #
    timestep_list = np.array(timestep_list)
    if (num_samples_per_traj is not None) and (len(timestep_list) > num_samples_per_traj):
        ind_to_keep = np.random.choice(len(timestep_list), size=num_samples_per_traj, replace=False)
        timestep_list = timestep_list[ind_to_keep]


    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _resize_and_encode(image, size):
            return tf.io.encode_jpeg(tf.cat(tf.round(tf.image.resize(image, size, method="bicubic")),
                tf.uint8))

        def _parse_example(episode_path):
            FRAMESKIP = 1
            IMAGE_SIZE = (256, 256)

            h5_filepath = os.path.join(path, 'trajectory.h5')
            recording_folderpath = os.path.join(path, 'recordings', 'MP4')
            
            traj = load_trajectory(h5_filepath, recording_folderpath=recording_folderpath)
            traj = traj[::frameskip]

            data = [flatten(t) for t in traj]
            assert all(t.keys() == traj_flat[0].keys() for t in data)

            for t in range(len(data)):
                for key in traj_flat[0].keys():
                    traj_flat[t][key] = _resize_and_encode(traj_flat[i][key], IMAGE_SIZE)
                

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                obs = step['observation']
                action = step['action']
                language_instruction = 'Execute a task.'
                # compute Kona language embedding
                language_embedding = self._embed([language_instruction])[0].numpy()

                episode.append({
                    'observation': {
                        'exterior_image_1_left': obs['image']['29838012_left'],
                        'exterior_image_1_right': obs['image']['29838012_right'],
                        'exterior_image_2_left': obs['image']['23404442_left']
                        'exterior_image_2_right': obs['image']['23404442_right'],
                        'wrist_image_left': obs['image']['19824535_left'],
                        'wrist_image_right': obs['image']['19824535_right'],
                        'cartesian_position': obs['robot_state']['cartesian_position'],
                        'joint_position': obs['robot_state']['joint_position'],
                        'gripper_position': obs['robot_state']['gripper_position'],
                    },
                    'action': {
                        'cartesian_position': action['cartesian_position'],
                        'cartesian_velocity': action['cartesian_velocity'],
                        'gripper_position': action['gripper_position'],
                        'gripper_velocity': action['gripper_velocity'],
                        'joint_position': action['joint_position'],
                        'joint_velocity': action['joint_velocity'],
                    },
                    'discount': 1.0,
                    'reward': float(i == (len(data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': h5_filepath,
                    'recording_folderpath': recording_folderpath
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = crawler(path)
        episode_paths = [p for p in episode_paths if os.path.exists(p + '/trajectory.h5') and \
                os.path.exists(p + '/recordings/MP4')]

        # for smallish datasets, use single-thread parsing
        #for sample in episode_paths:
        #    yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        beam = tfds.core.lazy_imports.apache_beam
        return (
                 beam.Create(episode_paths)
                 | beam.Map(_parse_example)
        )


