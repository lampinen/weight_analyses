from __future__ import print_function
from __future__ import division 

from collections import Counter

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
#import matplotlib.pyplot as plot

###### configuration ###########################################################

config = {
    "num_runs": 10,
    "batch_size": 10,
    "base_learning_rate": 0.0005,
    "base_lr_decay": 0.9,
    "base_lr_decays_every": 1,
    "base_lr_min": 0.00001,
    "base_training_epochs": 60,
    "output_path": "./results/",
    "nobias": False, # no biases
    "layer_sizes": [256, 128, 64, 128, 256]
}

###### MNIST data loading and manipulation #####################################
# downloaded from https://pjreddie.com/projects/mnist-in-csv/

train_data = np.loadtxt("../SWIL/MNIST_data/mnist_train.csv", delimiter = ",")
test_data = np.loadtxt("../SWIL/MNIST_data/mnist_test.csv", delimiter = ",")

def process_data(dataset):
    """Get data split into dict with labels and images"""
    labels = dataset[:, 0]
    images = dataset[:, 1:]/255.
    data = {"labels": labels, "images": images}
    return data

train_data = process_data(train_data)
test_data = process_data(test_data)

###### Build model func ########################################################

def softmax(x, T=1):
    """Compute the softmax function at temperature T"""
    if T != 1:
        x /= T
    x -= np.amax(x)
    x = np.exp(x)
    x /= np.sum(x)
    if not(np.any(x)): # handle underflow
        x = np.ones_like(x)/len(x) 
    return x

def to_unit_rows(x):
    """Converts row vectors of a matrix to unit vectors"""
    return x/np.expand_dims(np.sqrt(np.sum(x**2, axis=1)), -1)

def _display_image(x):
    x = np.reshape(x, [28, 28])
    plot.figure()
    plot.imshow(x, vmin=0, vmax=1)

class MNIST_autoenc(object):
    """MNIST autoencoder architecture, with or without replay buffer"""

    def __init__(self, layer_sizes):
        """Create a MNIST_autoenc model. 
           layer_sizes: list of the hidden layer sizes of the model
        """
        self.base_lr = config["base_learning_rate"]

        self.input_ph = tf.placeholder(tf.float32, [None, 784])
        self.lr_ph = tf.placeholder(tf.float32)

        self.bottleneck_size = min(layer_sizes)

	# small weight initializer
	weight_init = tf.contrib.layers.variance_scaling_initializer(factor=0.2, mode='FAN_AVG')
	

        net = self.input_ph
	bottleneck_layer_i = len(layer_sizes)//2
        for i, h_size in enumerate(layer_sizes):
	    if config["nobias"]:
	      net = slim.layers.fully_connected(net, h_size, activation_fn=tf.nn.relu,
						weights_initializer=weight_init,
						biases_initializer=None)
	    else:
	      net = slim.layers.fully_connected(net, h_size, activation_fn=tf.nn.relu,
						weights_initializer=weight_init)
            if i == bottleneck_layer_i: 
                self.bottleneck_rep = net
	if config["nobias"]:
	    self.output = slim.layers.fully_connected(net, 784, activation_fn=tf.nn.sigmoid,
						      weights_initializer=weight_init,
						      biases_initializer=None)
	else:
	    self.output = slim.layers.fully_connected(net, 784, activation_fn=tf.nn.sigmoid,
						      weights_initializer=weight_init)
						  
        self.loss = tf.nn.l2_loss(self.output-self.input_ph)

        self.optimizer = tf.train.GradientDescentOptimizer(self.lr_ph)
        self.train = self.optimizer.minimize(tf.reduce_mean(self.loss))

        self.first_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fully_connected/weights')[0]
        self.first_biases = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fully_connected/biases')[0]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, dataset, nepochs=100, log_file_prefix=None, test_dataset=None):
        """Train the model on a dataset"""
        if log_file_prefix is not None:
            if test_dataset is not None:
                with open(config["output_path"] + log_file_prefix + "base_test_losses.csv", "w") as fout:
                    fout.write("epoch, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i\n" % tuple(range(10)))
		    losses = self.eval(test_dataset)
		    fout.write(("0, ") + "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % tuple(losses))
	    with open(config["output_path"] + log_file_prefix + "base_train_losses.csv", "w") as fout:
		fout.write("epoch, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i\n" % tuple(range(10)))
		losses = self.eval(dataset)
		fout.write(("0, ") + "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % tuple(losses))

        batch_size = config["batch_size"]
        for epoch in range(1, nepochs + 1):
            order = np.random.permutation(len(dataset["labels"]))
            for batch_i in xrange(len(dataset["labels"])//batch_size):
                this_batch_images = dataset["images"][order[batch_i*batch_size:(batch_i+1)*batch_size], :]
                self.sess.run(self.train, feed_dict={
                        self.input_ph: this_batch_images,
                        self.lr_ph: self.base_lr 
                    })

            # eval
            if log_file_prefix is not None:
                if test_dataset is not None:
                    with open(config["output_path"] + log_file_prefix + "base_test_losses.csv", "a") as fout:
                        losses = self.eval(test_dataset)
			fout.write(("%i, " % epoch) + "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % tuple(losses))
		with open(config["output_path"] + log_file_prefix + "base_train_losses.csv", "a") as fout:
		    losses = self.eval(dataset)
		    fout.write(("%i, " % epoch) + "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % tuple(losses))

	    # update lr
	    if epoch > 0 and epoch % config["base_lr_decays_every"] == 0 and self.base_lr > config["base_lr_min"]: 
		self.base_lr *= config["base_lr_decay"]
         

    def get_reps(self, images):
        """Gets bottleneck reps for the given images"""
        batch_size = config["batch_size"]
        reps = np.zeros([len(images), self.bottleneck_size])
        for batch_i in xrange((len(images)//batch_size) + 1):
            this_batch_images = images[batch_i*batch_size:(batch_i+1)*batch_size, :]
            reps[batch_i*batch_size:(batch_i+1)*batch_size, :] = self.sess.run(
                self.bottleneck_rep, feed_dict={
                    self.input_ph: this_batch_images 
                })
        return reps

    def get_loss(self, images):
        """Gets losses for the given images"""
        batch_size = config["batch_size"]
        loss = np.zeros([len(images)])
        for batch_i in xrange((len(images)//batch_size) + 1):
            this_batch_images = images[batch_i*batch_size:(batch_i+1)*batch_size, :]
            loss[batch_i*batch_size:(batch_i+1)*batch_size] = self.sess.run(
                self.loss, feed_dict={
                    self.input_ph: this_batch_images
                })
        return loss

    def eval(self, dataset):
        """Evaluates model on the given dataset. Returns list of losses where
        losses[i] is the average loss on digit i"""
        losses = self.get_loss(dataset["images"])
        losses_summarized = [np.sum(losses[dataset["labels"] == i])/np.sum(dataset["labels"] == i) for i in range(10)]
        return losses_summarized

    def save_first_weights(self, filename_prefix):
        weights, biases = self.sess.run([self.first_weights, self.first_biases])
        np.savetxt(filename_prefix + "first_layer_weights.csv", weights, delimiter=',')
        np.savetxt(filename_prefix + "first_layer_biases.csv", biases, delimiter=',')

    def display_output(self, image):
        """Runs an image and shows comparison"""
        output_image = self.sess.run(self.output, feed_dict={
                self.input_ph: np.expand_dims(image, 0) 
            })

        _display_image(image)
        _display_image(output_image)
        plot.show()



###### Run stuff ###############################################################

for run in range(config["num_runs"]):
    filename_prefix = "run%i_" %(run)
    print(filename_prefix)
    np.random.seed(run)
    tf.set_random_seed(run)

    model = MNIST_autoenc(layer_sizes=config["layer_sizes"])

    np.random.shuffle(train_data)
    model.save_first_weights(filename_prefix + "pre_")

    model.train(train_data, config["base_training_epochs"],
                log_file_prefix=filename_prefix, test_dataset=test_data)

    model.save_first_weights(filename_prefix + "post_")
    tf.reset_default_graph()
