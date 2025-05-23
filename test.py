import tensorflow as tf

episode_paths = tf.io.gfile.glob("gs://gresearch/robotics/droid_raw/1.0.1/*/success/*/*/metadata_*.json")
for p in episode_paths:
    episode_id = p[:-5].split("/")[-1].split("_")[-1]
    import IPython; IPython.embed()