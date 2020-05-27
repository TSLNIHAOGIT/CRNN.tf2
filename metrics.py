# import tensorflow as tf
# from tensorflow import keras
from typing import List
import numpy as np

def compute_accuracy(ground_truth: List[str], predictions: List[str]) -> np.float32:
    accuracy = []
    for index, label in enumerate(ground_truth):
        prediction = predictions[index]
        total_count = len(label)
        correct_count = 0
        try:
            for i, tmp in enumerate(label):
                if tmp == prediction[i]:
                    correct_count += 1
        except IndexError:
            continue
        finally:
            try:
                accuracy.append(correct_count / total_count)
            except ZeroDivisionError:
                if len(prediction) == 0:
                    accuracy.append(1)
                else:
                    accuracy.append(0)

    accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    return accuracy


# class WordAccuracy(keras.metrics.Metric):
#     """
#     Calculate the word accuracy between y_true and y_pred.
#     """
#     def __init__(self, name='word_accuracy', **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.total = self.add_weight(name='total', dtype=tf.int32,
#                                      initializer=tf.zeros_initializer())
#         self.count = self.add_weight(name='count', dtype=tf.int32,
#                                      initializer=tf.zeros_initializer())
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         print('y',y_true,y_pred)
#         y_true = tf.sparse.to_dense(y_true)
#         """
#         Maybe have more fast implementation.
#         """
#         b = tf.shape(y_true)[0]
#         max_width = tf.maximum(tf.shape(y_true)[1], tf.shape(y_pred)[1])
#         logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
#         decoded, _ = tf.nn.ctc_greedy_decoder(
#             inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
#             sequence_length=logit_length)
#
#         # y_true = tf.sparse.reset_shape(y_true, [b, max_width])
#         # y_pred = tf.sparse.reset_shape(decoded[0], [b, max_width])
#         # y_true = tf.sparse.to_dense(y_true, default_value=-1)
#         # y_pred = tf.sparse.to_dense(y_pred, default_value=-1)
#
#         y_true = tf.reshape(y_true, (b, max_width))
#         print('decode',decoded[0])
#         y_pred = tf.reshape(tf.sparse.to_dense(decoded[0]), [b, max_width])
#         # y_true = tf.sparse.to_dense(y_true, default_value=-1)
#         # y_pred = tf.sparse.to_dense(y_pred, default_value=-1)
#         print('y_pred',y_pred)
#
#
#
#
#         y_true = tf.cast(y_true, tf.int32)
#         y_pred = tf.cast(y_pred, tf.int32)
#         values = tf.math.reduce_any(tf.math.not_equal(y_true, y_pred), axis=1)
#         values = tf.cast(values, tf.int32)
#         values = tf.reduce_sum(values)
#         self.total.assign_add(b)
#         self.count.assign_add(b - values)
#
#     def result(self):
#         return self.count / self.total
#
#     def reset_states(self):
#         self.count.assign(0)
#         self.total.assign(0)


if __name__=='__main__':
    # ground_truth =['RAK' '06ERNE0' 'U430WE' 'RDQK5' 'MA' '07SJW6IOQ' 'W0KXG0O'
    #      'DY2Z' 'JCYQO4' '7' 'S' '9UWG8Q' '66JW9LF' 'NF0CL4C0F' '0NT'
    #      'EP' 'QVEP009AS' '8DK' 'I0ZDO' 'NDC' 'KMJMT3' '0' 'F7Y365'
    #      'LI58SK43U']

    # predictions=[
    #     'RA', '0RNE0', 'U4W', 'QK', 'M', '7SWO', 'WK0', 'DZ', 'JCQ', '7', 'S', '9UQ', '6W', 'NC4C0', '0N', 'EP', 'V09', '8D', '0ZD', 'ND', 'MJM3', '0', 'FY35', 'L5S4']
    # ground_truth = ['RAK']
    # predictions = ['RAK']

    ground_truth=[b'D13' b'6A3O' b'PVJ' b'0FLKUW6' b'1VL170MJ' b'FYD3' b'D4ACESZDEX' b'C99'
     b'6' b'OK' b'K86SDG' b'MFAV' b'D' b'I59W' b'H' b'X' b'ZVOS8TB'
     b'IW5Z0CI4J' b'Y' b'NVL2M66' b'DL' b'WMA0EGM' b'BTMKV5J' b'AYTW6QOKZ'
     b'KUOX2' b'D8J6YT' b'9H40HKF' b'8D' b'G65' b'61NJJ9CJ3D']
    ground_truth=[each.decode('utf8') for each in ground_truth]
    print('ground_truth',ground_truth)
    predictions=[
        'D3', '6O', 'P', '0K', '117M', 'F3', 'ACEZEX', 'C9', '6', 'K', 'K6G', 'MV', 'D', 'I5', 'H', 'X', 'ZOS', 'IWIJ', 'Y', 'LM6', 'D', 'M0M', 'MK', 'AW6O', 'UX', '8J', '9HH', 'D', '6', '1JJJCJ']

    acc=compute_accuracy(ground_truth, predictions)
    print(acc)