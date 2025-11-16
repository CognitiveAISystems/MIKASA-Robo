from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class MikasaDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for mikasa dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Base camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(25,),
                            dtype=np.float32,
                            doc='NOT USED',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, for control type: pd_ee_delta_pose (see ManiSkill docs)',
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
                    # 'language_embedding': tfds.features.Tensor(
                    #     shape=(512,),
                    #     dtype=np.float32,
                    #     doc='Kona language embedding. '
                    #         'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    # ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {,
            'train' : ADD DATASET PATH HERE FOLLOWING EXAMPLE!!!
            # 'train': self._generate_examples(path='MIKASA-Robo/unbatched/RememberColor9-v0/train_data_*.npz'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            data = np.load(episode_path)     # this is a list of dicts in our case

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            traj_len = data['success'].shape[0]

            # for i in range(traj_len):
                # compute Kona language embedding
                # language_embedding = self._embed([step['language_instruction']])[0].numpy()

            for i in range(traj_len):
                episode.append({
                    'observation': {
                        'image': data['rgb'][i][:, :, :3],
                        'wrist_image': data['rgb'][i][:, :, 3:],
                        'state': data['joints'][i],
                    },
                    'action': data['action'][i],
                    'discount': 1.0,
                    'reward': data['reward'][i],#float(i == (len(data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (traj_len - 1),
                    'is_terminal': data['done'][i],#i == (traj_len - 1),
                    'language_instruction': ADD INSTRUCTION HERE!!!
                    # 'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

