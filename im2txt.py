#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:33:26 2020

@author: AkashTyagi
"""
import os
import math
import tensorflow as tf
from collections import defaultdict

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

class Captioning:
    def __init__(self):
        g = tf.Graph()
        with g.as_default():
            print(os.getcwd())
            model = inference_wrapper.InferenceWrapper()
            restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                        "./checkpoints/5M_iterations/model.ckpt-5000000")
        g.finalize()
        # Create the vocabulary.
        self.vocab = vocabulary.Vocabulary("./vocab/word_counts.txt")
        
        self.sess = tf.Session(graph=g)

        # Load the model from checkpoint.
        restore_fn(self.sess)
        self.generator = caption_generator.CaptionGenerator(model, self.vocab)


    def image(self, filename):
        result_set = defaultdict(list)
        count = 0
        with tf.gfile.GFile(filename, "rb") as f:
            image = f.read()
        captions = self.generator.beam_search(self.sess, image)
        print("Captions for image %s:" % os.path.basename(filename))
        for i, caption in enumerate(captions):
            # Ignore begin and end words.
            prob = math.exp(caption.logprob)
            if prob > 0:
                count +=1
                sentence = [self.vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                result_set[str(count)]={"Sentence":sentence,"confidence":prob}
                print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
        #os.unlink(filename)
        return result_set
                
    def close_session(self):
        self.sess.close()

if __name__ == "__main__":
    obj = Captioning();
    obj.image("./images/cycling.jpg")
    
