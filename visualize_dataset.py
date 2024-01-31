import argparse
import tqdm
import importlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress debug warning messages
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import wandb
import tensorflow as tf

WANDB_ENTITY = 'fliu'
WANDB_PROJECT = 'vis_rlds'

if WANDB_ENTITY is not None:
    render_wandb = True
    wandb.init(entity=WANDB_ENTITY,
               project=WANDB_PROJECT)
else:
    render_wandb = False


# create TF dataset
dataset_name = 'table_wiping'
print(f"Visualizing data from dataset: {dataset_name}")
module = importlib.import_module(dataset_name)
ds = tfds.load(dataset_name, data_dir='/hdd/data/cvp_rlds', split='train')
# ds = ds.filter(lambda episode: tf.strings.regex_full_match(episode['episode_metadata']['recording_folderpath'], '.*RAIL.*'))
# ds = ds.shuffle(100)

# visualize episodes
for i, episode in enumerate(ds.take(184)):
    images = []
    images_1 = []
    images_2 = []
    for step in episode['steps']:
        images.append(step['observation']['image_0'].numpy())
        images_1.append(step['observation']['image_1'].numpy())
        images_2.append(step['observation']['image_2'].numpy())
    
    image_strip_1 = np.concatenate(images[-20:][::4], axis=1)
    image_strip_2 = np.concatenate(images_1[-20:][::4], axis=1)
    image_strip_3 = np.concatenate(images_2[-20:][::4], axis=1)

    image_strip_4 = np.concatenate(images[:20][::4], axis=1)
    image_strip_5 = np.concatenate(images_1[:20][::4], axis=1)
    image_strip_6 = np.concatenate(images_2[:20][::4], axis=1)

    caption = step['language_instruction'].numpy().decode() + ' (temp. downsampled 4x)'

    if render_wandb:
        wandb.log({f'image_end_{i}': wandb.Image(image_strip_1, caption=caption)})
        wandb.log({f'image_end_{i}': wandb.Image(image_strip_2, caption=caption)})
        wandb.log({f'image_end_{i}': wandb.Image(image_strip_3, caption=caption)})
        wandb.log({f'image_beg_{i}': wandb.Image(image_strip_4, caption=caption)})
        wandb.log({f'image_beg_{i}': wandb.Image(image_strip_5, caption=caption)})
        wandb.log({f'image_beg_{i}': wandb.Image(image_strip_6, caption=caption)})
    else:
        plt.figure()
        plt.imshow(image_strip_1)
        plt.title(caption)
exit(0)

# visualize action and state statistics
actions, states = [], []
for episode in tqdm.tqdm(ds.take(500)):
    for step in episode['steps']:
        actions.append(step['action'].numpy())
        states.append(step['observation']['cartesian_position'].numpy())
actions = np.array(actions)
states = np.array(states)
action_mean = actions.mean(0)
state_mean = states.mean(0)

def vis_stats(vector, vector_mean, tag):
    assert len(vector.shape) == 2
    assert len(vector_mean.shape) == 1
    assert vector.shape[1] == vector_mean.shape[0]

    n_elems = vector.shape[1]
    fig = plt.figure(tag, figsize=(5*n_elems, 5))
    for elem in range(n_elems):
        plt.subplot(1, n_elems, elem+1)
        plt.hist(vector[:, elem], bins=20)
        plt.title(vector_mean[elem])

    if render_wandb:
        wandb.log({tag: wandb.Image(fig)})

vis_stats(actions, action_mean, 'action_stats')
vis_stats(states, state_mean, 'state_stats')

if not render_wandb:
    plt.show()


