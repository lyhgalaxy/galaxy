from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools
import tensorflow as tf


slim = tf.contrib.slim

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth', 'ratio'])
DepthwiseConv = namedtuple('DepthwiseConv', ['kernel', 'stride', 'depth', 'ratio'])
InvResBlock = namedtuple('InvResBlock', ['kernel', 'stride', 'depth', 'ratio', 'repeate'])


div=4   
    
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=64//div, ratio=1),  # 32*32
    DepthwiseConv(kernel=[3, 3], stride=1, depth=64//div, ratio=1),  # 32*32

    InvResBlock(kernel=[3, 3], stride=2, depth=64//div, ratio=2, repeate=3),  # 16*16
    InvResBlock(kernel=[3, 3], stride=2, depth=128//div, ratio=4, repeate=1), # 8*8
    InvResBlock(kernel=[3, 3], stride=1, depth=128//div, ratio=2, repeate=4),
    InvResBlock(kernel=[3, 3], stride=2, depth=128//div, ratio=4, repeate=1),
    InvResBlock(kernel=[3, 3], stride=1, depth=128//div, ratio=2, repeate=2),
    ]    
    
    
    
    
def inverted_block(net, input_filters, output_filters, expand_ratio, stride, scope=None):
    with tf.name_scope(scope):
        res_block = slim.conv2d(inputs=net, num_outputs=input_filters * expand_ratio, kernel_size=[1, 1])
        # depthwise conv2d
        res_block = slim.separable_conv2d(inputs=res_block, num_outputs=None, kernel_size=[3, 3], stride=stride, depth_multiplier=1.0, normalizer_fn=slim.batch_norm)
        res_block = slim.conv2d(inputs=res_block, num_outputs=output_filters, kernel_size=[1, 1], activation_fn=None)
        # stride 2 blocks
        if stride == 2:
            return res_block
        # stride 1 block
        else:
            if input_filters != output_filters:
                net = slim.conv2d(inputs=net, num_outputs=output_filters, kernel_size=[1, 1], activation_fn=None)
            return tf.add(res_block, net)

            
def mobilenet_v2_base(inputs,
                      final_endpoint='Conv2d_6_InvResBlock',
                      min_depth=8,
                      conv_defs=None,
                      scope=None):

  depth = lambda d: max(int(d), min_depth)
  end_points = {}

  if conv_defs is None:
    conv_defs = _CONV_DEFS

  with tf.variable_scope(scope, 'MobileFaceNet', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):

      net = inputs
      for i, conv_def in enumerate(conv_defs):
        end_point_base = 'Conv2d_%d' % i
        print(end_point_base)
        

        if isinstance(conv_def, Conv):
          end_point = end_point_base
          net = slim.conv2d(net, depth(conv_def.depth), conv_def.kernel,
                            stride=conv_def.stride,
                            normalizer_fn=slim.batch_norm,
                            scope=end_point)
          end_points[end_point] = net
          if end_point == final_endpoint:
            return net, end_points

        elif isinstance(conv_def, DepthwiseConv):
            end_point = 'DepthwiseConv'
            print(end_point)
            # depthwise conv2d
            net = slim.separable_conv2d(inputs=net, num_outputs=None, kernel_size=conv_def.kernel, stride=conv_def.stride,
                                        depth_multiplier=1.0, normalizer_fn=slim.batch_norm)
            net = slim.conv2d(inputs=net, num_outputs=conv_def.depth, kernel_size=[1, 1], activation_fn=None)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

        elif isinstance(conv_def, InvResBlock):
          end_point = end_point_base + '_InvResBlock'
          print(end_point)
          # inverted bottleneck blocks
          input_filters = net.shape[3].value
          # first layer needs to consider stride
          net = inverted_block(net, input_filters, depth(conv_def.depth), conv_def.ratio, conv_def.stride, end_point+'_0')
          for index in range(1, conv_def.repeate):
              suffix = '_' + str(index)
              net = inverted_block(net, input_filters, depth(conv_def.depth), conv_def.ratio, 1, end_point+suffix)

          end_points[end_point] = net
          if end_point == final_endpoint:
            return net, end_points

        else:
          raise ValueError('Unknown convolution type %s for layer %d'
                           % (conv_def.ltype, i))
  raise ValueError('Unknown final endpoint %s' % final_endpoint)

  

def mobilenet_v2(inputs,
                 bottleneck_layer_size=128,
                 is_training=False,
                 min_depth=8,
                 conv_defs=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='MobileFaceNet',
                 global_pool=False):
 
  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
    raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                     len(input_shape))

  with tf.variable_scope(scope, 'MobileFaceNet', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
      net, end_points = mobilenet_v2_base(inputs, scope=scope, min_depth=min_depth, conv_defs=conv_defs)   
      with tf.variable_scope('Logits'):
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
          end_points['global_pool'] = net
        else:
          # Pooling with a fixed kernel size.
          kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
          # Global depthwise conv2d
          net = slim.separable_conv2d(inputs=net, num_outputs=None, kernel_size=kernel_size, stride=1,
                                      depth_multiplier=1.0, activation_fn=None, padding='VALID')
          net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=[1, 1], stride=1, activation_fn=None, padding='VALID')
          end_points['GDConv'] = net

        # if not bottleneck_layer_size:
          # return net, end_points
        # # # logits = slim.conv2d(net, bottleneck_layer_size, kernel_size=[1, 1], stride=1, activation_fn=None, scope='LinearConv1x1')
        # logits = net
        # if spatial_squeeze:
          # logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
      # end_points['Logits'] = logits      
  return net, end_points


mobilenet_v2.default_image_size = 112


def wrapped_partial(func, *args, **kwargs):
  partial_func = functools.partial(func, *args, **kwargs)
  functools.update_wrapper(partial_func, func)
  return partial_func

  
def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out

  
def prelu(input, name=''):
    alphas = tf.get_variable(name=name + 'prelu_alphas',initializer=tf.constant(0.25,dtype=tf.float32,shape=[input.get_shape()[-1]]))
    pos = tf.nn.relu(input)
    neg = alphas * (input - abs(input)) * 0.5
    return pos + neg
 

def leaky_relu(input, name='leaky_relu'):
    tf.maximum(input, input*0.01, name=name) #leak 为传入的参数，可以设为0.01 等

    
def elu(input, name=''): 
    
    # tf.nn.leaky_relu
    pass
    
    
 

def mobilenet_v2_arg_scope(is_training=True,
                           weight_decay=0.00005,
                           regularize_depthwise=False):
 
  batch_norm_params = {
      'is_training': is_training,
      'center': True,
      'scale': True,
      'fused': True,
      'decay': 0.995,
      'epsilon': 2e-5,
      # force in-place updates of mean and variance estimates
      'updates_collections': None,
      # Moving averages ends up in the trainable variables collection
      'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
  }

  # Set weight_decay for weights in Conv and InvResBlock layers.
  #weights_init = tf.truncated_normal_initializer(stddev=stddev)
  weights_init = tf.contrib.layers.xavier_initializer(uniform=False)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  if regularize_depthwise:
    depthwise_regularizer = regularizer
  else:
    depthwise_regularizer = None
    
    
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.leaky_relu,
                      normalizer_fn=slim.batch_norm): #tf.keras.layers.PReLU                 
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
        with slim.arg_scope([slim.separable_conv2d],
                            weights_regularizer=depthwise_regularizer) as sc:
          return sc


def mobilenet_v2_out(net, end_points):
    age_logits = slim.fully_connected(net, 101, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                      weights_regularizer=slim.l2_regularizer(1e-5),
                                      scope='logits/age', reuse=False)
                                      
                                      
    end_points['FullNet'] = age_logits
    pre_logits= tf.squeeze(age_logits, [1, 2], name='SpatialSqueeze')
    end_points['AgeLogitsFlatten'] = pre_logits
    # gender_logits = slim.fully_connected(net, 2, activation_fn=None,
                                         # weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                         # weights_regularizer=slim.l2_regularizer(1e-5),
                                         # scope='logits/gender', reuse=False)
    # end_points['GenderLogitsFlatten'] = gender_logits
    return pre_logits, end_points
          
          
          
          
          
          
def inference(images, bottleneck_layer_size=128,channel_div=1, phase_train=False,
              weight_decay=0.00005, reuse=False):
    
    arg_scope = mobilenet_v2_arg_scope(is_training=phase_train, weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        net, end_points = mobilenet_v2(images, bottleneck_layer_size=bottleneck_layer_size, is_training=phase_train, reuse=reuse)
      
        return mobilenet_v2_out(net, end_points)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
