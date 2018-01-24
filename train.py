import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from nets import nets_factory
from data import get_split, load_batch, load_labels_into_dict
import os
import time
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
slim = tf.contrib.slim


items_to_descriptions = {'image': 'A 3-channel RGB coloured image', 'label': 'Img label'}

#================ DATASET INFORMATION ======================
#State dataset directory where the tfrecord files are located
tf.app.flags.DEFINE_string('dataset_dir', 'None', 'dataset directory where the tfrecord files are located')
tf.app.flags.DEFINE_string('train_dir', '/tmp/tflog/', 'direcotry where to create log files')
tf.app.flags.DEFINE_string('checkpoint_path', None, 'location of checkpoint file')
tf.app.flags.DEFINE_integer('image_size', 128, 'image size to resize input images to')
tf.app.flags.DEFINE_integer('num_classes', None, 'number of classes to predict')
tf.app.flags.DEFINE_string('labels_file', None, 'path to labels file')
tf.app.flags.DEFINE_string('file_pattern', 'falldata_%s_*.tfrecord', 'file pattern of TFRecord files')
tf.app.flags.DEFINE_string('file_pattern_for_counting', 'falldata', 'identify tfrecord files')
tf.app.flags.DEFINE_string('preprocessing', 'inception', 'define preprocessing function to use defiend in preprocessing_factory')
#================ TRAINING INFORMATION ======================
tf.app.flags.DEFINE_integer('num_preprocessing_threads', 4, 'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'number of epochs to train')
tf.app.flags.DEFINE_integer('batch_size', 8, 'batch size')
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None, 'checkpoint scopes to exclude')
tf.app.flags.DEFINE_string('model_name', 'mobilenet_v1', 'The name of the architecture to train.')
tf.app.flags.DEFINE_integer('log_every_n_steps', 100, 'The frequency with which logs are print.')
tf.app.flags.DEFINE_boolean('ignore_missing_vars', False, 'When restoring a checkpoint would ignore missing variables.')
tf.app.flags.DEFINE_boolean('save_every_n_epoch', None, 'Whether to save ckpt file every epoch')
tf.app.flags.DEFINE_string('trainable_scopes', None, 'Comma-separated list of scopes to filter the set of variables to train.' 'By default, None would train all the variables.')
tf.app.flags.DEFINE_integer(
        'num_readers', 4,
        'The number of parallel readers that read data from the dataset.')
######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
        'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
        'optimizer', 'rmsprop',
        'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
        '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
        'adadelta_rho', 0.95,
        'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
        'adagrad_initial_accumulator_value', 0.1,
        'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
        'adam_beta1', 0.9,
        'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
        'adam_beta2', 0.999,
        'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                                                    'The learning rate power.')

tf.app.flags.DEFINE_float(
        'ftrl_initial_accumulator_value', 0.1,
        'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
        'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
        'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
        'momentum', 0.9,
        'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
        'learning_rate_decay_type',
        'exponential',
        'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
        ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
        'end_learning_rate', 0.0001,
        'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
        'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
        'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
        'num_epochs_per_decay', 2.0,
        'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
        'sync_replicas', False,
        'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
        'replicas_to_aggregate', 1,
        'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
        'moving_average_decay', None,
        'The decay to use for the moving average.'
        'If left as None, then moving averages are not used.')

FLAGS = tf.app.flags.FLAGS

def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
        num_samples_per_epoch: The number of samples in each epoch of training.
        global_step: The global_step tensor.

    Returns:
        A `Tensor` representing the learning rate.

    Raises:
        ValueError: if
    """
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                                        FLAGS.num_epochs_per_decay)
    if FLAGS.sync_replicas:
        decay_steps /= FLAGS.replicas_to_aggregate

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                                                            global_step,
                                                                            decay_steps,
                                                                            FLAGS.learning_rate_decay_factor,
                                                                            staircase=True,
                                                                            name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                                                         global_step,
                                                                         decay_steps,
                                                                         FLAGS.end_learning_rate,
                                                                         power=1.0,
                                                                         cycle=False,
                                                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                                         FLAGS.learning_rate_decay_type)

def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

    Args:
        learning_rate: A scalar or `Tensor` learning rate.

    Returns:
        An instance of an optimizer.

    Raises:
        ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
                learning_rate,
                rho=FLAGS.adadelta_rho,
                epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
                learning_rate,
                initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
                learning_rate,
                beta1=FLAGS.adam_beta1,
                beta2=FLAGS.adam_beta2,
                epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
                learning_rate,
                learning_rate_power=FLAGS.ftrl_learning_rate_power,
                initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
                l1_regularization_strength=FLAGS.ftrl_l1,
                l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
                learning_rate,
                momentum=FLAGS.momentum,
                name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
                learning_rate,
                decay=FLAGS.rmsprop_decay,
                momentum=FLAGS.rmsprop_momentum,
                epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def _get_variables_to_train():
    """Returns a list of variables to train.

    Returns:
        A list of variables to train by the optimizer.
    """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    if len(variables_to_train) == 0:
        return None
    print(variables_to_train)
    return variables_to_train

#Defines functions to load checkpoint
def _get_init_fn():
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
        An init function run by the supervisor.
    """
    if FLAGS.checkpoint_path is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        tf.logging.info('Ignoring --checkpoint_path because a checkpoint already exists in {}'.format(FLAGS.train_dir))
        tf.logging.warning('Warning --checkpoint_exclude_scopes used when restoring from fine-tuned model {}'.format(FLAGS.checkpoint_exclude_scopes))
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
    elif tf.train.checkpoint_exists(FLAGS.checkpoint_path):
        checkpoint_path = FLAGS.checkpoint_path
    else:
        raise ValueError('No valid checkpoint found in --train_dir or --checkpoint_path: {}, {}'.format(FLAGS.train_dir, FLAGS.checkpoint_path))
    #exclusions = []
    #if FLAGS.checkpoint_exclude_scopes:
    #   exclusions = [scope.strip() for scope in FLAGS.checkpoint_exclude_scopes.split(',')]
    variables_to_restore = slim.get_variables_to_restore(exclude=FLAGS.checkpoint_exclude_scopes)
    tf.logging.info("Restoring variables")
    tf.logging.info(variables_to_restore)
    print(variables_to_restore)
    sys.exit(1)
    '''
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                tf.logging.info('Excluding {}'.format(var))
                break
        if not excluded:
            variables_to_restore.append(var)
    '''
    tf.logging.info('Fine-tuning from {}'.format(checkpoint_path))

    return slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=FLAGS.ignore_missing_vars)


#Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
#def train_step(sess, train_op, global_step, metrics_op, log=False):
def _train_step(sess, train_op, global_step, log=False):
    '''
    Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
    '''
    #Check the time for each sess run
    start_time = time.time()
    total_loss, global_step_count = sess.run([train_op, global_step])
    time_elapsed = time.time() - start_time
    if log:
        logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)
    return total_loss, global_step_count

def _add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.

    Args:
        result_tensor: The new final node that produces results.
        ground_truth_tensor: The node we feed ground truth data
        into.

    Returns:
        Tuple of (evaluation step, prediction).
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(
                    prediction, tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


def main(_):
    #Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)

    #======================= TRAINING PROCESS =========================
    #Now we start to construct the graph and build our model
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level

        #####################
        #    Data Loading   #
        #####################
        train_image_size = FLAGS.image_size or network_fn.default_image_size
        dataset = get_split('train', FLAGS.dataset_dir, FLAGS.num_classes, FLAGS.labels_file, file_pattern=FLAGS.file_pattern, file_pattern_for_counting=FLAGS.file_pattern_for_counting)
        image, raw_i, label = load_batch(dataset, FLAGS.preprocessing, FLAGS.batch_size, train_image_size, FLAGS.num_readers)
        images, labels = tf.train.batch([image, label],
                                        batch_size=FLAGS.batch_size,
                                        num_threads=FLAGS.num_preprocessing_threads,
                                        capacity=5 * FLAGS.batch_size,
                                        allow_smaller_final_batch=True)
        if not FLAGS.num_classes:
            logging.error('--num_classes invalid: {}, {}'.format(FLAGS.num_classes))
        num_batches_per_epoch = int(dataset.num_samples / FLAGS.batch_size)
        num_steps_per_epoch = num_batches_per_epoch

        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(FLAGS.model_name, num_classes=(dataset.num_classes), weight_decay=FLAGS.weight_decay, is_training=True)


        logits, end_points = network_fn(images)
        final_tensor = tf.identity(end_points['Predictions'], name="final_result")
        tf.summary.histogram('activations', final_tensor)
        #Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!) ?
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
        with tf.name_scope('cross_entropy_loss'):
            total_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        #Performs the equivalent to tf.nn.sparse__entropy_with_logits but enhanced with checks
            #cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            #loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits = logits)
            #cross_entropy_mean = tf.reduce_mean(loss)
            #Optionally calculate a weighted loss
            #if FLAGS.class_weight:
            #   loss = tf.losses.compute_weighted_loss(loss, weights=FLAGS.class_weight)
            #total_loss = tf.losses.get_total_loss() #obtain the regularization losses as well
        #Create the global step for monitoring the learning_rate and training.
        global_step = tf.train.get_or_create_global_step()
        learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
        optimizer = _configure_optimizer(learning_rate)
        variables_to_train = _get_variables_to_train()
        train_op = slim.learning.create_train_op(total_loss, optimizer, variables_to_train=variables_to_train)
        #predictions = tf.argmax(final_tensor, 1)
        #probabilities = end_points['Predictions']
        #accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        #metrics_op = tf.group(accuracy_update, probabilities)
        #accuracy, prediction = _add_evaluation_step(final_tensor, one_hot_labels)
        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update, probabilities)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        my_summary_op = tf.summary.merge_all()

        variables_to_restore = slim.get_variables_to_restore(exclude = FLAGS.checkpoint_exclude_scopes.split(','))
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)
        sv = tf.train.Supervisor(logdir = FLAGS.train_dir, summary_op = None, init_fn=restore_fn())
        #Run the managed session
        with sv.managed_session() as sess:
            for step in range(num_steps_per_epoch * FLAGS.num_epochs):
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, FLAGS.num_epochs)
                    if FLAGS.save_every_n_epoch:
                        if (step/num_batches_per_epoch)%FLAGS.save_every_n_epoch==0:
                            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)
                if step % FLAGS.log_every_n_steps == 0:
                    # optionally, print your logits and predictions for a sanity check that things are going fine.
                    '''
                    logits_value, probabilities_value, predictions_value, labels_value = sess.run([logits, probabilities, predictions, labels])
                    print('logits: {}'.format(logits_value))
                    print('Probabilities: {}'.format(probabilities_value))
                    print('predictions: {}'.format(predictions_value))
                    print('Labels: {}:'.format(labels_value))
                    '''
                    loss, _ = _train_step(sess, train_op, sv.global_step, log=True)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)
                else:
                    loss, _ = _train_step(sess, train_op, sv.global_step)
            #Log the final training loss and train accuracy
            logging.info('Final Loss: %s', loss)
            logging.info('Final Train Accuracy: %s', sess.run(accuracy))
            logging.info('Finished training! Saving model to disk now {}'.format(sv.save_path))
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)


if __name__ == '__main__':
  tf.app.run()
