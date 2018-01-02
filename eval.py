import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from preprocessing import inception_preprocessing
from nets import nets_factory
import time
import os
from data import get_split, load_batch
#from train_flowers import get_split, load_batch
#import matplotlib.pyplot as plt
#plt.style.use('ggplot')
slim = tf.contrib.slim

#================ DATASET INFORMATION ======================
#State dataset directory where the tfrecord files are located
tf.app.flags.DEFINE_string('log_dir', '/tmp/tflog/', 'direcotry where to find model files')
tf.app.flags.DEFINE_string('log_eval', './evallog', 'direcotry where to create log files')
tf.app.flags.DEFINE_string('dataset_dir', '/home/local/TECHNICALABS/alu/data/falldetect/data/processed/tfrecord/', 'directory to where Validation TFRecord files are')
tf.app.flags.DEFINE_integer('eval_batch_size', 32, 'batch size on how many examples to evalaue at a time')
tf.app.flags.DEFINE_integer('num_classes', 2, 'number of classes')
tf.app.flags.DEFINE_string('file_pattern', 'falldata_%s_*.tfrecord', 'file pattern of TFRecord files')
tf.app.flags.DEFINE_string('file_pattern_for_counting', 'falldata', 'identify tfrecord files')
tf.app.flags.DEFINE_string('labels_file', None, 'path to labels file')
tf.app.flags.DEFINE_integer('image_size', 224, 'image size ISxIS')
tf.app.flags.DEFINE_integer('batch_size', 1, 'batch size')
tf.app.flags.DEFINE_string('model_name', 'mobilenet_v1', 'name of model architecture defined in nets factory')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'number of epochs to evaluate for')
FLAGS = tf.app.flags.FLAGS


#Create a evaluation step function
def eval_step(sess, metrics, global_step, global_step_op, accuracy):
	'''
	Simply takes in a session, runs the metrics op and some logging information.
	'''
	start_time = time.time()
	_, global_step_count, accuracy_value = sess.run([metrics, global_step_op, accuracy])
	time_elapsed = time.time() - start_time
	#Log some information
	logging.info('Global Step %s: Streaming Accuracy: %.4f (%.2f sec/step)', global_step_count, accuracy_value, time_elapsed)
	return accuracy_value


def main(_):
	checkpoint_file = tf.train.latest_checkpoint(FLAGS.log_dir)
	#Create log_dir for evaluation information
	if not os.path.exists(FLAGS.log_eval):
		os.mkdir(FLAGS.log_eval)

	#Just construct the graph from scratch again
	with tf.Graph().as_default() as graph:
		tf.logging.set_verbosity(tf.logging.INFO)
		#Get the dataset first and load one batch of validation images and labels tensors. Set is_training as False so as to use the evaluation preprocessing
		dataset = get_split('validation', FLAGS.dataset_dir, FLAGS.num_classes, FLAGS.labels_file, FLAGS.file_pattern, FLAGS.file_pattern_for_counting)
		images, raw_images, labels = load_batch(dataset, FLAGS.batch_size, FLAGS.image_size, is_training = False)
		#Create some information about the training steps
		num_batches_per_epoch = int(dataset.num_samples / FLAGS.batch_size)
		num_steps_per_epoch = num_batches_per_epoch
		print(num_steps_per_epoch)
		network_fn = nets_factory.get_network_fn(FLAGS.model_name, num_classes=(dataset.num_classes), is_training=False)
		logits, end_points = network_fn(images)
		variables_to_restore = slim.get_variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		#FIXME: better way to define restore init fn?
		def restore_fn(sess):
			return saver.restore(sess, checkpoint_file)

		#Just define the metrics to track without the loss or whatsoever
		predictions = tf.argmax(end_points['Predictions'], 1)
		accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
		metrics_op = tf.group(accuracy_update)

		#Create the global step and an increment op for monitoring
		global_step = get_or_create_global_step()
		global_step_op = tf.assign(global_step, global_step + 1) #no apply_gradient method so manually increasing the global_step

		#Define some scalar quantities to monitor
		tf.summary.scalar('Validation_Accuracy', accuracy)
		my_summary_op = tf.summary.merge_all()

		#Get your supervisor
		sv = tf.train.Supervisor(logdir = FLAGS.log_eval, summary_op = None, saver = None, init_fn = restore_fn)

		#Now we are ready to run in one session
		with sv.managed_session() as sess:
			for step in range(num_steps_per_epoch * FLAGS.num_epochs):
				sess.run(sv.global_step)
				#print vital information every start of the epoch as always
				if step % num_batches_per_epoch == 0:
					logging.info('Epoch: %s/%s', step / num_batches_per_epoch + 1, FLAGS.num_epochs)
					logging.info('Current Streaming Accuracy: %.4f', sess.run(accuracy))
				#Compute summaries every 10 steps and continue evaluating
				if step % 10 == 0:
					eval_step(sess, metrics = metrics_op, global_step = sv.global_step, global_step_op=global_step_op, accuracy=accuracy)
					summaries = sess.run(my_summary_op)
					sv.summary_computed(sess, summaries)

				#Otherwise just run as per normal
				else:
					eval_step(sess, global_step_op = global_step_op, metrics = metrics_op, global_step = sv.global_step, accuracy=accuracy)
			#At the end of all the evaluation, show the final accuracy
			logging.info('Final Streaming Accuracy: %.4f', sess.run(accuracy))
			logging.info('Model evaluation has completed! Visit TensorBoard for more information regarding your evaluation.')
if __name__ == '__main__':
	tf.app.run()