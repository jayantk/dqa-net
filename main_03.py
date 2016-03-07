import json
from copy import deepcopy
from pprint import pprint
import os

import tensorflow as tf

from configs.get_config import get_config
from models.attention_model_03 import AttentionModel
from read_data_03 import read_data
from configs.version_03 import configs

flags = tf.app.flags

# File directories
flags.DEFINE_string("log_dir", "log", "Log directory [log]")
flags.DEFINE_string("save_dir", "save", "Save directory [save]")
flags.DEFINE_string("eval_dir", "eval", "Eval value storing directory [eval]")
flags.DEFINE_string("data_dir", "data/s3", "Data directory [data/s3]")
flags.DEFINE_string("fold_path", "data/s3/fold1.json", "fold json path [data/s3/fond1.json]")

# Training parameters
flags.DEFINE_integer("batch_size", 100, "Batch size for the network [100]")
flags.DEFINE_integer("hidden_size", 100, "Hidden size [50]")
flags.DEFINE_integer("image_size", 4096, "Image size [4096]")
flags.DEFINE_integer("num_layers", 3, "Number of layers [3]")
flags.DEFINE_integer("rnn_num_layers", 2, "Number of rnn layers [2]")
flags.DEFINE_float("init_mean", 0, "Initial weight mean [0]")
flags.DEFINE_float("init_std", 0.1, "Initial weight std [0.1]")
flags.DEFINE_float("init_lr", 0.01, "Initial learning rate [0.01]")
flags.DEFINE_integer("anneal_period", 100, "Anneal period [100]")
flags.DEFINE_float("anneal_ratio", 0.5, "Anneal ratio [0.5")
flags.DEFINE_integer("num_epochs", 200, "Total number of epochs for training [200]")
flags.DEFINE_boolean("linear_start", False, "Start training with linear model? [False]")
flags.DEFINE_float("max_grad_norm", 40, "Max grad norm; above this number is clipped [40]")
flags.DEFINE_float("keep_prob", 0.5, "Keep probability of dropout [0.5]")

# Training and testing options
flags.DEFINE_boolean("train", False, "Train? Test if False [False]")
flags.DEFINE_integer("val_num_batches", 5, "Val num batches [5]")
flags.DEFINE_boolean("load", False, "Load from saved model? [False]")
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_boolean("gpu", False, 'Enable GPU? (Linux only) [False]')
flags.DEFINE_integer("val_period", 5, "Val period (for display purpose only) [5]")
flags.DEFINE_integer("save_period", 10, "Save period [10]")
flags.DEFINE_integer("config", -1, "Config number to load. -1 to use currently defined config. [-1]")

# Debugging
flags.DEFINE_boolean("draft", False, "Draft? (quick build) [False]")

# App-specific training parameters
# TODO : Any other parameters

# App-specific options
# TODO : Any other options

FLAGS = flags.FLAGS


def main(_):
    meta_data_path = os.path.join(FLAGS.data_dir, "meta_data.json")
    meta_data = json.load(open(meta_data_path, "r"))
    if not os.path.exists(FLAGS.eval_dir):
        os.mkdir(FLAGS.eval_dir)
    if FLAGS.train:
        train_ds = read_data(FLAGS, 'train')
        val_ds = read_data(FLAGS, 'val')
        FLAGS.train_num_batches = train_ds.num_batches
        FLAGS.val_num_batches = min(FLAGS.val_num_batches, train_ds.num_batches, val_ds.num_batches)
        if not os.path.exists(FLAGS.save_dir):
            os.mkdir(FLAGS.save_dir)
    else:
        test_ds = read_data(FLAGS, 'test')
        FLAGS.test_num_batches = test_ds.num_batches

    # Other parameters
    FLAGS.max_sent_size = meta_data['max_sent_size']
    FLAGS.max_label_size = meta_data['max_label_size']
    FLAGS.max_num_rels = meta_data['max_num_rels']
    FLAGS.pred_size = meta_data['pred_size']
    FLAGS.num_choices = meta_data['num_choices']
    FLAGS.vocab_size = meta_data['vocab_size']
    FLAGS.main_name = __name__

    config_name = "Config%s" % str(FLAGS.config).zfill(2)
    config_priority = 1
    config_dict = configs[FLAGS.config]() if FLAGS.config >= 0 else {}

    # For quick draft build (deubgging).
    if FLAGS.draft:
        FLAGS.train_num_batches = 1
        FLAGS.val_num_batches = 1
        FLAGS.test_num_batches = 1
        FLAGS.num_epochs = 1
        FLAGS.eval_period = 1
        FLAGS.save_period = 1
        # TODO : Add any other parameter that induces a lot of computations
        FLAGS.num_layers = 1
        config_priority = 0

    config = get_config(FLAGS.__flags, config_dict, config_name, priority=config_priority)
    pprint(config._asdict)

    graph = tf.Graph()
    model = AttentionModel(graph, config)
    with tf.Session(graph=graph) as sess:
        sess.run(tf.initialize_all_variables())
        if config.train:
            writer = tf.train.SummaryWriter(config.log_dir, sess.graph_def)
            if config.load:
                model.load(sess)
            model.train(sess, writer, train_ds, val_ds)
        else:
            model.load(sess)
            eval_tensors = [model.yp]
            model.eval(sess, test_ds, eval_tensors=eval_tensors)

if __name__ == "__main__":
    tf.app.run()
