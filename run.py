# import
import tensorflow as tf
from config import *
from data import Data
import matplotlib.pyplot as plt
import csv
import os


# set meta file
best_step = 0
meta_name = 'chk_step_%d.ckpt' % best_step

# write csv
csv_path = 'output.csv'
if os.path.exists(csv_path):
    os.remove(csv_path)
    print('[run] removed the current csv')

csv_file = open(csv_path, 'w')
csv_writer = csv.writer(csv_file, lineterminator='\n')

header = ['ImageID', 'Label']
csv_writer.writerow(header)

row_index = 0

# get batches
def get_batches(data_len, batch_size):
    batch_starts = range(0, data_len, batch_size)
    batch_ends = [batch_start + batch_size for batch_start in batch_starts]
    return zip(batch_starts, batch_ends)

# load config
with tf.name_scope('config'):
    cfig = get_config()

# load data
with tf.name_scope('data'):
    data = Data()
    data.read_test_data(cfig[eKey.test_path])
    test_data  = data.get_test_data()

# run
with tf.Session(config=tf.ConfigProto()) as sess:
    saver = tf.train.import_meta_graph(cfig[eKey.checkpoint_dir] + meta_name + '.meta')
    saver.restore(sess, cfig[eKey.checkpoint_dir] + meta_name)

    data = tf.get_collection('data')[0]
    pred = tf.get_collection('pred')[0]

    batches = get_batches(test_data.shape[0], 100) # 100 data per batch
    for start, end in batches:
        print('[run] run batch: start = {0}; end = {1}'.format(start,end))
        rPred = sess.run(pred, feed_dict={data: test_data[start:end]})

        for i in range(rPred.shape[0]):
            row_index = row_index + 1
            items = [str(row_index), str(rPred[i])]
            csv_writer.writerow(items)

'''
# show some results
for i in range(test_data.shape[0]):
    plt.imshow(np.squeeze(test_data[i]), cmap='gray')
    plt.title('prediction = %d' %prediction[i])
    plt.pause(0.2)
'''
    
print('The end.')    
