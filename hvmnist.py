import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from tensorflow.examples.tutorials.mnist import input_data



# Mnist data
data=input_data.read_data_sets('data/MNIST/',one_hot=True)
print(data)
#First Convolutional Layer
filter_size1=5
num_filters1=16

#Second Convolutional Layer
filter_size2=5
num_filters2=36

#Fully Connected layer
fc_size=128

data.test.cls=np.argmax(data.test.labels,axis=1)

# describing our Mnist Data
img_size=28

img_size_flat=img_size*img_size

img_shape=(img_size,img_size)

num_channels=1

num_classes=10

def plot_images(images,cls_true,cls_pred=None):
	assert len(images)==len(cls_true)==9


	fig,axes=plt.subplots(3,3)
	fig.subplots_adjust(hspace=0.3,wspace=0.3)


	for i,ax in enumerate(axes.flat):
		ax.imshow(images[i].reshape(img_shape),cmap='binary')


		if cls_pred is None:
			xlabel="true:{0}".format(cls_true[i])
		else:
			xlabel="true:{0},Pred:{1}".format(cls_true[i],cls_pred[i])


		ax.set_xlabel(xlabel)

		ax.set_xticks([])
		ax.set_yticks([])
	plt.ion()
	plt.show()


images=data.test.images[0:9]
cls_true=data.test.cls[0:9]
plot_images(images,cls_true=cls_true)



def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape,stddev=0.05))

def new_biases(length):
	return tf.Variable(tf.constant(0.05,shape=[length]))


def new_conv_layer(input,num_input_channel,filter_size,num_filters,use_pooling=True):

	shape=[filter_size,filter_size,num_input_channel,num_filters]

	weights=new_weights(shape=shape)

	biases=new_biases(length=num_filters)

	layer=tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME')

	layer+=biases


	if use_pooling:
		layer=tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


	layer=tf.nn.relu(layer)


	return layer,weights



def flatten_layer(layer):
	layer_shape=layer.get_shape()

	num_features=layer_shape[1:4].num_elements()

	layer_flat=tf.reshape(layer,[-1,num_features])


	return layer_flat,num_features




def new_fc_layer(input,num_inputs,num_outputs,use_relu=True):

	weights=new_weights(shape=[num_inputs,num_outputs])

	biases=new_biases(length=num_outputs)


	layer=tf.matmul(input,weights)+biases

	if use_relu:
		layer=tf.nn.relu(layer)

	return layer



x=tf.placeholder(tf.float32,shape=[None,img_size_flat],name='x')
y_true=tf.placeholder(tf.float32,shape=[None,10],name='y_true')
y_true_cls=tf.argmax(y_true,dimension=1)
x_image=tf.reshape(x,[-1,img_size,img_size,num_channels])



layer_conv1,weights_conv1=new_conv_layer(x_image,num_channels,filter_size1,num_filters1,use_pooling=True)

layer_conv2,weights_conv2=new_conv_layer(layer_conv1,num_filters1,filter_size2,num_filters2,use_pooling=True)

layer_flat,num_features=flatten_layer(layer_conv2)

print(layer_flat,"\n",num_features)

layer_fc1=new_fc_layer(layer_flat,num_features,fc_size,use_relu=True)

layer_fc2=new_fc_layer(layer_fc1,fc_size,num_classes,use_relu=False)


y_pred=tf.nn.softmax(layer_fc2)
y_pred_cls=tf.argmax(y_pred,dimension=1)

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true))
optimizer=tf.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)
correct_prediction=tf.equal(y_pred_cls,y_true_cls)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess=tf.Session()
sess.run(tf.global_variables_initializer())

test_batch_size=256

def plot_example_errors(cls_pred,correct):
	incorrect=(correct==False)

	images=data.test.images[incorrect]
	cls_pred=cls_pred[incorrect]
	cls_true=data.test.cls[incorrect]

	plot_images(images[0:9],cls_true[0:9],cls_pred[0:9])






def test_accuracy():
	num_test=len(data.test.images)

	cls_pred=np.zeros(shape=num_test,dtype=np.int)

	i=0
	while i<num_test:
		j=min(i+test_batch_size,num_test)

		images=data.test.images[i:j,:]
		labels=data.test.labels[i:j,:]
		feed_dict={x:images,y_true:labels}
		cls_pred[i:j]=sess.run(y_pred_cls,feed_dict=feed_dict)
		cls_true=data.test.cls

		correct=(cls_true==cls_pred)
		correct_sum=correct.sum()

		acc=float(correct_sum)/num_test
		print("Accuracy: {0}".format(acc))

	plot_example_errors(cls_pred,correct)

def plot_conv_layer(layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = sess.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids =int(math.ceil(math.sqrt(num_filters)))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.ion()
    plt.show()
    plt.savefig("layer.png")



def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.
    
    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = sess.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int(math.ceil(math.sqrt(num_filters)))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.ion()
    plt.show()
    plt.savefig("weights.png")





train_batch_size=64

for i in range(200):
	batch=data.train.next_batch(train_batch_size)
	
	if i%100==0:
		train_accuracy=sess.run(accuracy,feed_dict={x:batch[0],y_true:batch[1]})
		print("step %d, training accuracy %g"%(i, train_accuracy))


	sess.run(optimizer,feed_dict={x:batch[0],y_true:batch[1]})


#test_accuracy()

image1 = data.test.images[0]
def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()
plot_image(image1)

plot_conv_weights(weights=weights_conv1)

plot_conv_layer(layer=layer_conv1, image=image1)



