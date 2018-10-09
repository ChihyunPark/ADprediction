import numpy as np
import tensorflow as tf
from sklearn import datasets


class Autoencoder:
    def __init__(self, input_dim, hidden_dim, epoch=250, batch_size=10, learning_rate=0.001):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])

        with tf.name_scope('encode'):
            weights = tf.Variable(tf.random_normal([input_dim, hidden_dim], dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([hidden_dim]), name='biases')
            encoded = tf.nn.tanh(tf.matmul(x, weights) + biases)
        with tf.name_scope('decode'):
            weights = tf.Variable(tf.random_normal([hidden_dim, input_dim], dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([input_dim]), name='biases')
            decoded = tf.matmul(encoded, weights) + biases

        self.x = x
        self.encoded = encoded
        self.decoded = decoded
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.x, self.decoded))))

        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()

    def get_batch(self, X, size):
        a = np.random.choice(len(X), size, replace=False)
        return X[a]

    def train(self, data):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epoch):
                for j in range(np.shape(data)[0] // self.batch_size):
                    batch_data = self.get_batch(data, self.batch_size)
                    l, _ = sess.run([self.loss, self.train_op], feed_dict={self.x: batch_data})
                if i % 10 == 0:
                    print('epoch {0}: loss = {1}'.format(i, l))
                    self.saver.save(sess, './model.ckpt')
            self.saver.save(sess, './model.ckpt')

    def test(self, data):
        with tf.Session() as sess:
            self.saver.restore(sess, "./model.ckpt")
            hidden, reconstructed = sess.run([self.encoded, self.decoded], feed_dict={self.x: data})
            #print('input', data)
            #print('compressed', hidden)
            #print('reconstructed', reconstructed)
            return reconstructed, hidden




def main():
    hidden_dim = 2
    data = datasets.load_iris().data

    input_dim = len(data[0])
    ae = Autoencoder(input_dim, hidden_dim, 500, 10, 0.1)
    ae.train(data)

    reconstructed, hidden = ae.test([[8, 4, 6, 2]])
    print(reconstructed)
    print(hidden)

if __name__ == '__main__':
    main()
