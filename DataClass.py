# coding: utf-8
# In[ ]:
import numpy as np
import numpy
import gc
import psutil
import os

#shuffle可以选择，
class DataSet(object):

    def __init__(self,images):
        self._images = images
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = images.shape[0] #Refers to the number of samples of all training data
 

    def crossValidation(self,k_fold=1):  
        Test_data = {}
        Train_data = {}
        images = self._images[:]
        
        index_num_example = np.arange(self._num_examples)
        np.random.shuffle(index_num_example)
        images = images[index_num_example]

        if k_fold==1:
           Train_data[0]=images[0:4*images.shape[0]//5]
           Test_data[0]=images[4*images.shape[0]//5:]
           return Train_data,Test_data

        else:    
            test_num = self._num_examples//k_fold			   
            for i in range(k_fold-1): #Remove the last fold and deal with it separately
                test_arange = np.arange(test_num*i,test_num*(i+1))
                idx_all = np.arange(self._num_examples)
                train_arange = np.delete(idx_all, test_arange)
                Test_data[i] = images[test_arange]
                Train_data[i] = images[train_arange] 
                Test_data[k_fold-1]= images[(k_fold-1)*test_num:]
                Train_data[k_fold-1] = images[0:test_num*(k_fold-1)]
            Test_data[k_fold-1]= images[(k_fold-1)*test_num:]
            Train_data[k_fold-1] = images[0:test_num*(k_fold-1)]
            return Train_data,Test_data     
    
    
    def next_batch(self, batch_size,shuffle=True):
        start = self._index_in_epoch  #self._index_in_epoch  How many samples are used for all calls, which is equivalent to a global variable #start The first batch is 0. The rest is the same as self._index_in_epoch. If there is more than one epoch, it will be re-assigned below.。
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)  
            numpy.random.shuffle(perm0)
            self._images = self._images[perm0]
        # Go to the next epoch
        #从这里到下一个else，所做的是一个epoch快运行完了，但是不够一个batch，将这个epoch的结尾和下一个epoch的开头拼接起来，共同组成一个batch——size的数据。

        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start  # 一个epoch 最后不够一个batch还剩下几个
            images_rest_part = self._images[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self._images[perm]
              # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end] 
            return numpy.concatenate((images_rest_part, images_new_part), axis=0)
            #新的epoch，和上一个epoch的结尾凑成一个batch
        else:
            self._index_in_epoch += batch_size  #每调用这个函数一次，_index_in_epoch就加上一个batch——size的，它相当于一个全局变量，上不封顶
            end = self._index_in_epoch
            return self._images[start:end]
			


# In[ ]:
