import tensorflow as tf
import cv2
import numpy as np

ckpt_path = 'model/model.ckpt-262131'
images_list_path = 'dataset/train.txt'
images_path = 'dataset/train'
threshold = .4
MEANS = (123.68, 116.78, 103.94)
model_input_shape = (513, 513)


def preprocess_image(image, means=MEANS):
    image = cv2.resize(image, model_input_shape)
    image = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    channels = cv2.split(image)
    for i in range(len(channels)):
        channels[i] -= means[i]
    image = cv2.merge(channels)
    image = np.expand_dims(image, axis=0)
    return image


images = []
image_names = []
image_shapes = []
with open(images_list_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        image_names.append(line.strip())
        image = cv2.imread(images_path+'/images/'+line.strip() + '.jpg')
        image_shapes.append((image.shape[0], image.shape[1]))
        image = preprocess_image(image)
        images.append(image)


saver = tf.train.import_meta_graph(ckpt_path+'.meta')
with tf.Session() as sess:
    saver.restore(sess, ckpt_path)
    graph = tf.get_default_graph()
    in_tensor = graph.get_tensor_by_name('IteratorGetNext:0')
    out_tensor_name = graph.get_tensor_by_name('softmax_tensor:0')
    counter = 0
    for image, image_name, image_shape in zip(images, image_names, image_shapes):
        counter += 1
        print('processing ' + str(counter) + ' of ' + str(len(images)))
        out_tensor = sess.run(out_tensor_name, feed_dict={in_tensor: image})
        out_image = cv2.resize(np.uint8(np.squeeze(out_tensor[:, :, :, 1] > threshold)), image_shape)
        cv2.imwrite(images_path + '/model_pred/' + image_name + '.png', out_image)

