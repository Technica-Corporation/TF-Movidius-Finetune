import tensorflow as tf
from nets import nets_factory
slim = tf.contrib.slim

ckpt_path_full = './models/checkpoints/mobilenet/mobilenet_224/mobilenet_v1_1.0_224.ckpt'
ckpt_path_last = '/tmp/model.ckpt'
key_scope = 'final_training_ops'
#Define model
global_step = slim.create_global_step()
network_fn = nets_factory.get_network_fn('mobilenet_v1', num_classes=1001, is_training=False)
placeholder = tf.placeholder("float", name='input', shape=[None, 224, 224, 3])
logits, endpoints = network_fn(placeholder)
with tf.name_scope('final_training_ops'):
    with tf.name_scope('weights'):
        initial_value = tf.truncated_normal([1001, 200], stddev=0.001)
        layer_weights = tf.Variable(initial_value, name='final_weights')
    with tf.name_scope('biases'):
        layer_biases = tf.Variable(tf.zeros([200]), name='final_biases')
    with tf.name_scope('Wx_plus_b'):
        final_tensor = tf.matmul(logits, layer_weights) + layer_biases

variables_to_restore = []
variables_to_restore_last = ['final_training_ops/weights/final_weights:0']
for var in slim.get_variables_to_restore():
    excluded = False
    if var.op.name.startswith(key_scope):
        excluded = True
        break
    if not excluded:
        variables_to_restore.append(var)
init_fn_full = slim.assign_from_checkpoint_fn(ckpt_path_full, variables_to_restore)
init_fn_last = slim.assign_from_checkpoint_fn(ckpt_path_last, [layer_weights, layer_biases])
#restorer = tf.train.Saver(var_list = [layer_weights, layer_biases])
saver = tf.train.Saver()
with tf.Session() as sess:
    init_fn_full(sess)
    init_fn_last(sess)
    #restorer.restore(sess, ckpt_path_last)
    saver.save(sess, './test.ckpt')