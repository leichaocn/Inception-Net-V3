#构建42层的Inception V3网络，估算其forward和backward计算的耗时。

# region 载入库,定义常用函数,配置基础参数
import tensorflow as tf
from datetime import datetime
import math
import time

slim = tf.contrib.slim
#引用truncate_normal(0.01),即表示得到一个标准差为0.01的正太分布 tf.truncated_normal_initializer(0.0, 0.01)
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


#循环100个batch
num_batches=20
batch_size = 10
height, width = 299, 299
#为对前向计算的时间进行测试，所以生成32张随机图片，作为一个batch。
inputs = tf.random_uniform((batch_size, height, width, 3))

# endregion

# region 计算图生成函数
def inception_v3_base(inputs, scope=None):
  # 定义一个字典，用来保存一些关键节点。
  end_points = {}

  with tf.variable_scope(scope, 'InceptionV3', [inputs]):
    #用arg_scope对其中的三类函数设置缺省值。
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='VALID'):
      # 299 x 299 x 3
      net = slim.conv2d(inputs, 32, [3, 3], stride=2, scope='Conv2d_1a_3x3')
      # 149 x 149 x 32
      net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_2a_3x3')
      # 147 x 147 x 32
      net = slim.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv2d_2b_3x3')
      # 147 x 147 x 64
      net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
      # 73 x 73 x 64
      net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_3b_1x1')
      # 73 x 73 x 80.
      net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_4a_3x3')
      # 71 x 71 x 192.
      net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_5a_3x3')
      # 35 x 35 x 192.

    # Inception blocks
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
      # 第一个Inception模块组（blocks），包括3个module，其中Mixed_5b约=Mixed_5c=Mixed_5d
      # mixed,输出shape为: 35 x 35 x 256.
      with tf.variable_scope('Mixed_5b'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          # 输出shape为: 35 x 35 x 64.
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
          # 输出shape为: 35 x 35 x 64.
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
          # 输出shape为: 35 x 35 x 96.
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
          # 输出shape为: 35 x 35 x 32.
        #每个branch的rank=4，即shape=[batches,height,width,channels],
        #tf.concat([],num) 表示如果num是0，表示按batches相加；1，按height；2，按width；3，按channels。
        #tf.concat([],4)的结果就是，64+64+96+32=256，而4个branch的batches，height，width本身都相同。
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # mixed_1,输出shape为: 35 x 35 x 288.
      with tf.variable_scope('Mixed_5c'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
          branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv_1_0c_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')#仅仅这里的深度64与mix_5b不同
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # mixed_2,输出shape为: 35 x 35 x 288.
      with tf.variable_scope('Mixed_5d'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
      
      # 第二个Inception模块组（blocks），包括5个module，其中Mixed_6a，Mix_6b约=Mixed_6c=Mixed_6d=Mixed_6e
      # mixed_3: 输入35*35*288,输出17 x 17 x 768.
      with tf.variable_scope('Mixed_6a'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 384, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1')
          # 输出17*17*384
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1')
          # 输出17*17*96
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
          # 输出shape为: 17 x 17 x 256.(大概是因为补０的原因，待查)
        net = tf.concat([branch_0, branch_1, branch_2], 3)

      # mixed4: 17 x 17 x 768.
      with tf.variable_scope('Mixed_6b'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # mixed_5: 17 x 17 x 768.
      with tf.variable_scope('Mixed_6c'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')#160与6b的128不同
          branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')#160与6b的128不同
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')#160与6b的128不同
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')#160与6b的128不同
          branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')#160与6b的128不同
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')#160与6b的128不同
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
      # mixed_6: 17 x 17 x 768.
      with tf.variable_scope('Mixed_6d'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # mixed_7: 17 x 17 x 768.
      with tf.variable_scope('Mixed_6e'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
      end_points['Mixed_6e'] = net

      # 第三个Inception模块组（blocks），包括3个module，其中有Mixed_7a，Mix_7b=Mixed_7c
      # mixed_8: 8 x 8 x 1280.
      with tf.variable_scope('Mixed_7a'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
          branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
        net = tf.concat([branch_0, branch_1, branch_2], 3)
      # mixed_9: 8 x 8 x 2048.
      with tf.variable_scope('Mixed_7b'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat([
              slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = tf.concat([
              slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # mixed_10: 8 x 8 x 2048.
      with tf.variable_scope('Mixed_7c'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat([
              slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')], 3)
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = tf.concat([
              slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
      return net, end_points


def inception_v3(inputs,
                 num_classes=1000, #最后输出的分类
                 is_training=True, #true即表示在训练，就会启用dropout和BN
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax, #最后默认的分类层
                 spatial_squeeze=True, #true表示默认是要对维度进行压缩，由1*1*1000变成1000
                 reuse=None,#是否会对网络和Variable重复使用，None表示否 
                 scope='InceptionV3'):

  with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      #inception_v3_base()即我们前面定义的三个Inception模块组网络结构。
      net, end_points = inception_v3_base(inputs, scope=scope)

      # Auxiliary Head logits，输入：Mixed_6e的输出，即前两个模块组的输出。输出：aux_logits存入end_points['AuxLogits']
      # 结构是：1个平均池化，3个卷积，最后是1个空间降维。
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='SAME'):
        aux_logits = end_points['Mixed_6e']
        with tf.variable_scope('AuxLogits'):
          #池化 
          aux_logits = slim.avg_pool2d(
              aux_logits, [5, 5], stride=3, padding='VALID',
              scope='AvgPool_1a_5x5')
          #此时aux_logits的shape是5*5*768
          #卷积 
          aux_logits = slim.conv2d(aux_logits, 128, [1, 1],
                                   scope='Conv2d_1b_1x1')

          #卷积 
          aux_logits = slim.conv2d(
              aux_logits, 768, [5,5],
              weights_initializer=trunc_normal(0.01),
              padding='VALID', scope='Conv2d_2a_5x5')
          #此时aux_logits的shape是1*1*768
          #卷积 
          aux_logits = slim.conv2d(
              aux_logits, num_classes, [1, 1], activation_fn=None,
              normalizer_fn=None, weights_initializer=trunc_normal(0.001),
              scope='Conv2d_2b_1x1')
          #此时aux_logits的shape是1*1*1000，用tf.squeeze去掉两个为1的维度。[batch_size,1,1,1000] 
          #空间降维
          if spatial_squeeze:
            #把[1,2]表示从aux_logits的shape的最左边数起（第一个维度是0位置），第2个和第3个位置均是1维，用squeeze压缩掉。
            aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
          end_points['AuxLogits'] = aux_logits

      # Logits层。 输入：Mixed_7c的输出，即上面三个模块组的最终输出。输出：prelogits和logits存入end_points
      # 结构为1个平均池化，1个dropout（结果同时存入end_points），1个卷积，1个空间降维（结果logits同时存入end_points）
      with tf.variable_scope('Logits'):
        #池化。池化前的shape为8*8*2048
        net = slim.avg_pool2d(net, [8, 8], padding='VALID',
                              scope='AvgPool_1a_8x8')
        # 池化后，net为1 x 1 x 2048
        # dropout
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        end_points['PreLogits'] = net
        # 2048
        # 卷积
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
        #空间降维
        if spatial_squeeze:
          #把[1,2]表示从logits的shape的最左边数起（第一个维度是batch_size所处在的0位置），第2个和第3个位置均是1维，用squeeze压缩掉。
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        # 1000
      end_points['Logits'] = logits#这里的logits是最终返回的logits
      
      #Softmax层，输入：上一层求得的logits。输出：把预测结果存入end_points
      #prediction_fn()即为slim.sotfmax()
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points #这里返回的logits，是softmax层的输入。经过softmax层的结果存在end_points['Predictions']里。
# endregion

# region 参数配置函数
def inception_v3_arg_scope(weight_decay=0.00004,
                           stddev=0.1,
                           batch_norm_var_collection='moving_vars'):
  # 这里定义字典batch_norm_params，在下面的对slim.conv2d的参数定义中赋给了normalizer_params。
  batch_norm_params = {
      'decay': 0.9997,
      'epsilon': 0.001,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      }
  }

  with slim.arg_scope([slim.conv2d, slim.fully_connected],#事实上，slim.fully_connected（）并未用到。
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=trunc_normal(stddev),#trunc_normal()在程序开始处已经用lambda定义了。
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params) as sc:
        # 在卷积slim.conv2d中，引入了slim.batch_norm，并赋予了参数batch_norm_params
      return sc
# endregion
  
# region 计算图执行函数
# 进行前向计算,求出每个batch所耗费的时间。
# 传入的session是已经启动了的会话，传入的target则需要graph通过运行session计算出的目标，infor_string仅是作为输出的字符串。
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print ('%s: 第 %d 步（batch）, 该步消耗时间为 %.3f 秒' %
                       (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print ('%s: %s 经过 %d 步（batch）, 每步（batch）消耗时间是 %.3f +/- %.3f 秒' %
           (datetime.now(), info_string, num_batches, mn, sd))
# endregion

# region 构建计算图
# 这个inception_v3_arg_scope()函数是我们前面定义的，
# 规定了InceptionV3网络内BN，卷积，全连接的一些缺省参数。
# 传入到slim.arg_scope()里，用于给整个网络的参数进行缺省初始化。
with slim.arg_scope(inception_v3_arg_scope()):
  # inception_v3()函数即前面我们定义的整个InceptionV3网络结构。
  # 训练时，is_training置为True（缺省值也时True），会启用dropout和batch_norm
  logits, end_points = inception_v3(inputs, is_training=False)
init = tf.global_variables_initializer()
# endregion

# region 执行计算图
sess = tf.Session()
sess.run(init)
# 执行伪预测,评估前向时间.
time_tensorflow_run(sess, logits, "Forward")

# endregion
