#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:30:03 2026

@author: huzhen
"""
from keras.metrics import Metric
import tensorflow as tf
from keras import ops as K


class MaskedAccuracy(Metric):
    def __init__(self, pad_token_id, name="masked_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_token_id = pad_token_id
        self.correct_count = self.add_weight(name="correct_count", initializer="zeros")
        self.total_valid_tokens = self.add_weight(name="total_valid_tokens", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred_labels = K.argmax(y_pred, axis=-1)#(-1,m)
        y_true = tf.cast(y_true, dtype=pred_labels.dtype)#(-1,m)
        mask = K.not_equal(y_true, self.pad_token_id)
        correct_predictions = tf.equal(pred_labels, y_true) & mask
        self.correct_count.assign_add(tf.reduce_sum(K.cast(correct_predictions, tf.float32)))
        self.total_valid_tokens.assign_add(tf.reduce_sum(tf.cast(mask, tf.float32)))

    def result(self):
        return self.correct_count / tf.maximum(self.total_valid_tokens, 1e-7)

    def reset_state(self):
        self.correct_count.assign(0)
        self.total_valid_tokens.assign(0)