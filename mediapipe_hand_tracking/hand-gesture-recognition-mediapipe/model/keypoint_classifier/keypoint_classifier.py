#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='mediapipe_hand_tracking/hand-gesture-recognition-mediapipe/model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads) #创建一个tf.lite.Interpreter对象，并指定模型路径和线程数参数。

        self.interpreter.allocate_tensors() #为输入和输出张量分配内存。
        self.input_details = self.interpreter.get_input_details()   #获取输入张量的详细信息
        self.output_details = self.interpreter.get_output_details() #获取输出张量的详细信息

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index'] #获取输入张量的索引。
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))    #输入手部关键点列表赋值给输入张量
        self.interpreter.invoke()   #运行解释器进行推断。

        output_details_tensor_index = self.output_details[0]['index']   #获取输出张量的索引。

        result = self.interpreter.get_tensor(output_details_tensor_index)   #获取输出张量的值。

        result_index = np.argmax(np.squeeze(result))    #对输出结果进行处理，返回最大值的索引值。

        return result_index #返回最终结果。
