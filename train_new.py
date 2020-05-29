import os
import time
import argparse

import tensorflow as tf
from tensorflow import keras

from dataset import OCRDataLoader,parse_mjsynth,Decoder
from model import crnn
from losses import CTCLoss
from metrics import compute_accuracy #WordAccuracy




parser = argparse.ArgumentParser()
# parser.add_argument("-ta", "--train_annotation_paths", type=str,default='../example/images',
#                     required=True, nargs="+",
#                     help="The path of training data annnotation file.")
parser.add_argument("-va", "--val_annotation_paths", type=str, nargs="+", 
                    help="The path of val data annotation file.")
# parser.add_argument("-tf", "--train_parse_funcs", type=str,default='mjsynth', required=True,
#                     nargs="+", help="The parse functions of annotaion files.")
parser.add_argument("-vf", "--val_parse_funcs", type=str, nargs="+", 
                    help="The parse functions of annotaion files.")
# parser.add_argument("-t", "--table_path", type=str, required=True,default='example/tables' ,
#                     help="The path of table file.")
parser.add_argument("-w", "--image_width", type=int, default=100, 
                    help="Image width(>=16).")
parser.add_argument("-b", "--batch_size", type=int, default=30,
                    help="Batch size.")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001,
                    help="Learning rate.")
parser.add_argument("-e", "--epochs", type=int, default=100,
                    help="Num of epochs to train.")

args = parser.parse_args()
print('args',args)

train_dl = OCRDataLoader(
    # args.train_annotation_paths,
    # args.train_parse_funcs,

[
    r'E:\tsl_file\python_project\all_datas\annotation_crnn.txt'
    # r'E:\tsl_file\python_project\CRNN.tf2\example\annotation.txt',
 ],
[
    'example'
],


    args.image_width,
    # args.table_path,
r'E:\tsl_file\python_project\CRNN.tf2\example\table.txt',

    args.batch_size,
    True)

with open(r'E:\tsl_file\python_project\CRNN.tf2\example\table.txt', "r") as f:
    inv_table = [char.strip() for char in f]
decoder = Decoder(inv_table)


print("Num of training samples: {}".format(len(train_dl)))
localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
# saved_model_path = ("saved_models/{}/".format(localtime) +
#     "{epoch:03d}_{word_accuracy:.4f}.h5")
if args.val_annotation_paths:
    val_dl = OCRDataLoader(
        args.val_annotation_paths, 
        args.val_parse_funcs, 
        args.image_width,
        args.table_path,
        args.batch_size)
    print("Num of val samples: {}".format(len(val_dl)))
    # saved_model_path = ("saved_models/{}/".format(localtime) +
    #     "{epoch:03d}_{word_accuracy:.4f}_{val_word_accuracy:.4f}.h5")
else:
    val_dl = lambda: None

print("Start at {}".format(localtime))
# os.makedirs("saved_models/{}".format(localtime))


print('train_dl.num_classes',train_dl.num_classes)
model = crnn(train_dl.num_classes)
# model.build(input_shape=())
# print('model.summary={}'.format(model.summary()))

print('start compile')
custom_loss=CTCLoss()
print('custom_loss={}'.format(custom_loss))
# compute_accuracy=WordAccuracy()



start_learning_rate = args.learning_rate
learning_rate = tf.Variable(start_learning_rate, dtype=tf.float32)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)




# model.compile(
#               optimizer=keras.optimizers.Adam(lr=args.learning_rate),
#               loss=custom_loss,
#               metrics=[WordAccuracy()]
# )
#
#
# model.summary()
#
# callbacks = [
#     # keras.callbacks.ModelCheckpoint(saved_model_path),
#     keras.callbacks.TensorBoard(log_dir="../example/logs/{}".format(localtime),
#                                 histogram_freq=1)
# ]
# print('start fit')
#


# for index, each in enumerate(dataset):
#     #index=0-29,就是repeat=30
#     #每张图片大小是32*100   shape=(label_counts,h,w,1)
#     #each (<tf.Tensor: id=2706, shape=(4, 32, 100, 1),
#     print('index={},\n,each={}'.format(index,each))
#
# model.fit(train_dl(), epochs=args.epochs,
#           callbacks=callbacks,
#           validation_data=val_dl())
#


import math
import numpy as np

print('train_dl',type(train_dl),train_dl,len(train_dl))
dataset=train_dl()

# nums=tf.data.experimental.cardinality(dataset)

nums=train_dl.size
print('nums',nums)


BATCH_SIZE=args.batch_size
##所有的数据：未划分训练和测试一共可以分为几个batch
N_BATCH=math.ceil(nums/BATCH_SIZE)#总数据条数/batch_size
print('N_BATCH',N_BATCH)



val_size=int(0.2*N_BATCH)
print('val_size',val_size)
# train_size=nums-val_size





# print('dataset test',dataset)
# for (batch, (inp, targ, ground_truth)) in enumerate(dataset):
#     print('dataset==',inp,targ,ground_truth)

##因为dataset是已经经过batch处理了，返回的都是一个个的batch
dataset_val=dataset.take(val_size)

# # print('dataset test',dataset)
# for (batch, (inp, targ, ground_truth)) in enumerate(dataset_val):
#     print('dataset==',batch,targ,ground_truth)



dataset=dataset.skip(val_size)
# print('dataset test',dataset)
# for (batch, (inp, targ, ground_truth)) in enumerate(dataset):
#     print('batch==',batch)
#     print('dataset==',inp,targ,ground_truth
#           )







logdir = "./logs/"
writer = tf.summary.create_file_writer(logdir)


checkpoint_dir = r'E:\tsl_file\python_project\all_models\crnn_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir , checkpoint_name='ckpt', max_to_keep=5)

status=checkpoint.restore(manager.latest_checkpoint)
print('status',status)

start_step=optimizer.iterations.numpy()
print('start_step',start_step,N_BATCH)#start_step 495 4
if start_step==0:
    learning_rate.assign(start_learning_rate)
else:
    lr = max(0.00001, start_learning_rate * math.pow(0.99, (start_step+1)/N_BATCH+1))#epoch
    learning_rate.assign(lr)


# print('optimizer.iterations.numpy()',optimizer.iterations.numpy())#optimizer.iterations.numpy() 1062
# lr = max(0.00001, args.learning_rate * math.pow(0.99, optimizer.iterations.numpy()//30))
# learning_rate.assign(lr)

# if True:
with writer.as_default():
    for epoch in range(args.epochs):
        start = time.time()

        total_loss = 0

        print('start_train')
        # print('start')

        for (batch, (inp, targ,ground_truth)) in enumerate(dataset):
            print('epoch={},batch={}'.format(epoch,batch))
            # step = epoch * N_BATCH + batch

            # print('batch',batch)
            # print('inp shape',inp.shape)#inp shape (64, 32, 100, 1, 1)
            # loss = 0
            # global_step.assign_add(1)

            # results = np.zeros((BATCH_SIZE, targ.shape[1] - 1), np.int32)

            with tf.GradientTape() as tape:
                y_pred_logits = model(inp)
                '''
                targ=SparseTensor(indices=tf.Tensor(
                [[ 0  0][[ 0  0]
                 [ 0  1][0 1],shape=(172, 2),values=tf.Tensor(
                [44 36  2 40 41),shape=(172,),dense_shape=tf.Tensor([30 10], shape=(2,)
                 [ 0  1]y_pred_logits shape=(30, 24, 63)
                
                '''

                # print('targ={},\n,y_pred_logits={}'.format(targ,y_pred_logits))
                batch_loss=custom_loss(targ,y_pred_logits)


            total_loss += batch_loss


            gradients = tape.gradient(batch_loss, model.trainable_variables)

            gradients, _ = tf.clip_by_global_norm(gradients, 15)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            step = optimizer.iterations
            print('step={}'.format(step))

            if (step + 1) % N_BATCH == 0:
                # 这里应该模拟的是一个epoch学习率才会进行一次变动
                lr = max(0.00001, start_learning_rate * math.pow(0.99, (step + 1) / N_BATCH))  # epoch
                learning_rate.assign(lr)

            if (step + 1)%min(30,N_BATCH)==0:
                '''
                每30次迭代之后，添加损失函数到tensorboard,并保存一次模型
                '''
                ground_truth=[each.decode('utf8') for each in ground_truth.numpy()]
                # print('y true',targ)#dense_shape=tf.Tensor([30 10],
                print('ground_truth=',ground_truth)
                decoded=decoder.decode(y_pred_logits, method='beam_search')
                print('decoded',decoded)#len is batch_size

                acc = compute_accuracy(ground_truth, decoded)
                print('acc',acc)


                ##在验证集上测试准确率
                acc_val=[]
                for (batch, (inp_val, targ_val, ground_truth_val)) in enumerate(dataset_val):
                    y_pred_logits_val = model(inp_val)
                    ground_truth_val = [each.decode('utf8') for each in ground_truth_val.numpy()]
                    decoded_val = decoder.decode(y_pred_logits_val, method='beam_search')
                    acc_val.append( compute_accuracy(ground_truth_val, decoded_val))
                print('acc_val',np.mean(acc_val),acc_val)
                for i in range(3):
                    print("real_val:{:s}  pred_val:{:s} acc_val:{:f}".format(ground_truth_val[i], decoded_val[i],
                                                                 compute_accuracy([ground_truth_val[i]], [decoded_val[i]])))







                #
                tf.summary.scalar('loss', batch_loss, step=step)
                tf.summary.scalar('accuracy', acc, step=step)
                tf.summary.scalar('lr', learning_rate.numpy(), step=step)
                writer.flush()


                print('Epoch {} Batch {} Loss {:.4f}  '.format(epoch,batch,batch_loss.numpy()))

            # if optimizer.iterations.numpy() % 10 == 0:
                for i in range(3):
                    print("real:{:s}  pred:{:s} acc:{:f}".format(ground_truth[i], decoded[i],
                                                                 compute_accuracy([ground_truth[i]], [decoded[i]])))

                # checkpoint.save(file_prefix=checkpoint_prefix)
                # checkpoint.save(file_prefix=checkpoint_prefix)
                path = manager.save(checkpoint_number=step)
                print("model saved to %s" % path)

        # print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


'''

E:\tsl_file\python_project\TF_2C\python.exe E:/tsl_file/python_project/CRNN.tf2/train.py
2020-05-18 13:19:40.106240: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_100.dll
args Namespace(batch_size=256, epochs=20, image_width=100, learning_rate=0.001, val_annotation_paths=None, val_parse_funcs=None)
anot E:\tsl_file\python_project\CRNN.tf2\example\annotation.txt ** example
content [['images/1_Paintbrushes_55044.jpg', '55044'], ['images/2_Reimbursing_64165.jpg', '64165'], ['images/3_Creationisms_17934.jpg', '17934']]
img_paths =['E:\\tsl_file\\python_project\\CRNN.tf2\\example\\images/1_Paintbrushes_55044.jpg', 'E:\\tsl_file\\python_project\\CRNN.tf2\\example\\images/2_Reimbursing_64165.jpg', 'E:\\tsl_file\\python_project\\CRNN.tf2\\example\\images/3_Creationisms_17934.jpg']
 labels=['55044', '64165', '17934']
2020-05-18 13:19:59.788489: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
2020-05-18 13:20:00.389906: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce MX130 major: 5 minor: 0 memoryClockRate(GHz): 1.189
pciBusID: 0000:01:00.0
2020-05-18 13:20:00.403735: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2020-05-18 13:20:00.421960: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-05-18 13:20:00.618718: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2020-05-18 13:20:01.131654: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce MX130 major: 5 minor: 0 memoryClockRate(GHz): 1.189
pciBusID: 0000:01:00.0
2020-05-18 13:20:01.132012: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2020-05-18 13:20:01.132927: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-05-18 13:20:17.506140: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-18 13:20:17.506411: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
2020-05-18 13:20:17.506572: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
2020-05-18 13:20:17.675573: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1384 MB memory) -> physical GPU (device: 0, name: GeForce MX130, pci bus id: 0000:01:00.0, compute capability: 5.0)
Num of training samples: 3
Start at 2020-05-18-13-20-20
start compile
y Tensor("y_true:0", shape=(None, None, None), dtype=float32) Tensor("y_pred:0", shape=(None, None, 63), dtype=float32)
decode SparseTensor(indices=Tensor("CTCGreedyDecoder:0", shape=(None, 2), dtype=int64), values=Tensor("CTCGreedyDecoder:1", shape=(None,), dtype=int64), dense_shape=Tensor("CTCGreedyDecoder:2", shape=(2,), dtype=int64))
y_pred Tensor("Reshape_1:0", shape=(None, None), dtype=int64)
Traceback (most recent call last):
  File "E:/tsl_file/python_project/CRNN.tf2/train.py", line 79, in <module>
    loss=CTCLoss(), metrics=[WordAccuracy()])
  File "E:\tsl_file\python_project\TF_2C\lib\site-packages\tensorflow_core\python\training\tracking\base.py", line 457, in _method_wrapper
    result = method(self, *args, **kwargs)
  File "E:\tsl_file\python_project\TF_2C\lib\site-packages\tensorflow_core\python\keras\engine\training.py", line 373, in compile
    self._compile_weights_loss_and_weighted_metrics()
  File "E:\tsl_file\python_project\TF_2C\lib\site-packages\tensorflow_core\python\training\tracking\base.py", line 457, in _method_wrapper
    result = method(self, *args, **kwargs)
  File "E:\tsl_file\python_project\TF_2C\lib\site-packages\tensorflow_core\python\keras\engine\training.py", line 1653, in _compile_weights_loss_and_weighted_metrics
    self.total_loss = self._prepare_total_loss(masks)
  File "E:\tsl_file\python_project\TF_2C\lib\site-packages\tensorflow_core\python\keras\engine\training.py", line 1713, in _prepare_total_loss
    per_sample_losses = loss_fn.call(y_true, y_pred)
  File "E:\tsl_file\python_project\CRNN.tf2\losses.py", line 21, in call
    blank_index=self.blank_index)
  File "E:\tsl_file\python_project\TF_2C\lib\site-packages\tensorflow_core\python\ops\ctc_ops.py", line 689, in ctc_loss_v2
    name=name)
  File "E:\tsl_file\python_project\TF_2C\lib\site-packages\tensorflow_core\python\ops\ctc_ops.py", line 764, in ctc_loss_dense
    label_length = ops.convert_to_tensor(label_length, name="label_length")
  File "E:\tsl_file\python_project\TF_2C\lib\site-packages\tensorflow_core\python\framework\ops.py", line 1184, in convert_to_tensor
    return convert_to_tensor_v2(value, dtype, preferred_dtype, name)
  File "E:\tsl_file\python_project\TF_2C\lib\site-packages\tensorflow_core\python\framework\ops.py", line 1242, in convert_to_tensor_v2
    as_ref=False)
  File "E:\tsl_file\python_project\TF_2C\lib\site-packages\tensorflow_core\python\framework\ops.py", line 1296, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "E:\tsl_file\python_project\TF_2C\lib\site-packages\tensorflow_core\python\framework\constant_op.py", line 286, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "E:\tsl_file\python_project\TF_2C\lib\site-packages\tensorflow_core\python\framework\constant_op.py", line 227, in constant
    allow_broadcast=True)
  File "E:\tsl_file\python_project\TF_2C\lib\site-packages\tensorflow_core\python\framework\constant_op.py", line 265, in _constant_impl
    allow_broadcast=allow_broadcast))
  File "E:\tsl_file\python_project\TF_2C\lib\site-packages\tensorflow_core\python\framework\tensor_util.py", line 437, in make_tensor_proto
    raise ValueError("None values not supported.")
ValueError: None values not supported.

Process finished with exit code 1

'''