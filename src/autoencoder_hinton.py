#coding=utf-8
"""
deep auto-encoder using RBM

"""
import os
import sys
import time
import gzip
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from scipy import io
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from rbm import RBM

try:
    import PIL.Image as Image
except ImportError:
    import Image
    
import pylab

#===============================================================================
# DAE -- deep auto-encoder 
#===============================================================================
from scipy.io import loadmat, savemat
class autoencoder_hinton(object):
    
    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 layers_sizes=[129, 500, 54, 500, 129],
                 output_folder='out',
                 p_dict=None,
                 sigma_dict=None,
                 file_path = '',
                 logger=None):
        
        self.sigmoid_layers = []        
        self.params = []
        self.finetune_train_flag = False
        self.n_layers = len(layers_sizes)
        self.n_layers_rmb = int(len(layers_sizes)/2)
        self.numpy_rng = numpy_rng
        self.output_folder = output_folder
        self.x = T.matrix(name='x')
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))        
        if logger == None:
            self.logger = mylogger(output_folder + '/log.log')
        else:
            self.logger = logger
        
        assert self.n_layers > 0
        if p_dict==None:
            self.p_list = [0]*self.n_layers_rmb
            self.sigma_list = [0]*self.n_layers_rmb
            self.p = 0
            self.sigma = 0
        elif p_dict!=None and sigma_dict==None:
            assert len(p_dict['p_list']) == self.n_layers_rmb
            self.p_list = p_dict['p_list']
            self.sigma_list = [0]*self.n_layers_rmb
            self.p = p_dict['p']
            self.sigma = 0
        elif p_dict!=None and sigma_dict!=None:           
            assert len(p_dict['p_list']) == self.n_layers_rmb
            assert len(sigma_dict['sigma_list']) == len(p_dict['p_list'])
            self.p_list = p_dict['p_list']
            self.sigma_list = sigma_dict['sigma_list']
            self.p = p_dict['p']
            self.sigma = sigma_dict['sigma']
        
        w = [0]*self.n_layers_rmb; hbias = [0]*self.n_layers_rmb; vbias = [0]*self.n_layers_rmb;
#         set_trace() 
        w[0], hbias[0], vbias[0] = self.load_hinton_rbm(file_path+'/mnistvh.mat', 'vishid', 'hidrecbiases', 'visbiases') #rbm1
#         w[1], hbias[1], vbias[1] = self.load_hinton_rbm(file_path+'/mnisthp.mat', 'hidpen', 'penrecbiases', 'hidgenbiases') #rbm2
#         w[2], hbias[2], vbias[2] = self.load_hinton_rbm(file_path+'/mnisthp2.mat', 'hidpen2', 'penrecbiases2', 'hidgenbiases2') #rbm3
#         w[3], hbias[3], vbias[3] = self.load_hinton_rbm(file_path+'/mnistpo.mat', 'hidtop', 'toprecbiases', 'topgenbiases') #rbm4
        
        sigmoid_layer1 = HiddenLayer(rng=self.numpy_rng, input=self.x, 
                                    n_in=w[0].get_value(borrow=True).shape[0], n_out=w[0].get_value(borrow=True).shape[1],
                                    W=w[0], b=hbias[0], activation=T.nnet.sigmoid)        
#         sigmoid_layer2 = HiddenLayer(rng=self.numpy_rng, input=sigmoid_layer1.output, 
#                                     n_in=w[1].get_value(borrow=True).shape[0], n_out=w[1].get_value(borrow=True).shape[1],
#                                     W=w[1], b=hbias[1], activation=T.nnet.sigmoid)
#         sigmoid_layer3 = HiddenLayer(rng=self.numpy_rng, input=sigmoid_layer2.output, 
#                                     n_in=w[1].get_value(borrow=True).shape[1], n_out=w[1].get_value(borrow=True).shape[0],                                    
#                                     W=theano.shared(numpy.asarray(w[1].T.eval(), dtype=theano.config.floatX), borrow=True),
#                                     b=vbias[1], activation=T.nnet.sigmoid) #sigmoid layer, inverse
        sigmoid_layer2 = HiddenLayer(rng=self.numpy_rng, input=sigmoid_layer1.output, 
                                    n_in=w[0].get_value(borrow=True).shape[1], n_out=w[0].get_value(borrow=True).shape[0],                                    
                                    W=theano.shared(numpy.asarray(w[0].T.eval(), dtype=theano.config.floatX), borrow=True),
                                    b=vbias[0], activation=None) #output layer, inverse
        
#         m_idx = self.n_layers_rmb-1;
#         for i in xrange(m_idx):
#             if i == 0:
#                 layer_input = self.x
#             else:
#                 layer_input = self.sigmoid_layers[-1].output
#             sigmoid_layer = HiddenLayer(rng=self.numpy_rng, input=layer_input, 
#                                     n_in=w[i].get_value(borrow=True).shape[0], n_out=w[i].get_value(borrow=True).shape[1],
#                                     W=w[i], b=hbias[i], activation=T.nnet.sigmoid) #sigmoid layer
#             self.sigmoid_layers.append(sigmoid_layer)
#             
#         sigmoid_layer = HiddenLayer(rng=self.numpy_rng, input=self.sigmoid_layers[-1].output, 
#                                     n_in=w[m_idx].get_value(borrow=True).shape[0], n_out=w[m_idx].get_value(borrow=True).shape[1],
#                                     W=w[m_idx], b=hbias[m_idx], activation=T.nnet.sigmoid) #coding layer
#         self.sigmoid_layers.append(sigmoid_layer)
#         
#         for i in xrange(m_idx,0,-1):
#             sigmoid_layer = HiddenLayer(rng=self.numpy_rng, input=self.sigmoid_layers[-1].output, 
#                                     n_in=w[i].get_value(borrow=True).shape[1], n_out=w[i].get_value(borrow=True).shape[0],                                    
#                                     W=theano.shared(numpy.asarray(w[i].T.eval(), dtype=theano.config.floatX), borrow=True),
#                                     b=vbias[i], activation=T.nnet.sigmoid) #sigmoid layer, inverse
#             self.sigmoid_layers.append(sigmoid_layer)
#         
#         sigmoid_layer = HiddenLayer(rng=self.numpy_rng, input=self.sigmoid_layers[-1].output, 
#                                     n_in=w[0].get_value(borrow=True).shape[1], n_out=w[0].get_value(borrow=True).shape[0],                                    
#                                     W=theano.shared(numpy.asarray(w[0].T.eval(), dtype=theano.config.floatX), borrow=True),
#                                     b=vbias[0], activation=None) #output layer, inverse
#         self.sigmoid_layers.append(sigmoid_layer)

        self.sigmoid_layers.append(sigmoid_layer1)
        self.sigmoid_layers.append(sigmoid_layer2)
#         self.sigmoid_layers.append(sigmoid_layer3)
#         self.sigmoid_layers.append(sigmoid_layer4)
        for sigmoid_layer in self.sigmoid_layers:            
            self.params.extend(sigmoid_layer.params)
        set_trace()
        self.save_model_mat('%s/DAE_pre%s.mat'%(self.output_folder, self.get_model_file()))
        
#         finetune_file = '%s/DAE%s.mat'%(output_folder, self.get_model_file())
#         if os.path.isfile(finetune_file): #trained
#             self.finetune_train_flag = True
            
            
    def load_hinton_rbm(self, file_name, w_name, hbias_name, vbias_name, shared=1):
        
        dataset = io.loadmat(file_name) #rbm
        w = dataset[w_name]
        hbias = dataset[hbias_name]
        hbias = hbias.reshape(hbias.shape[1])
        vbias = dataset[vbias_name]
        vbias = vbias.reshape(vbias.shape[1])
        if shared == 1:
            w_shared = theano.shared(numpy.asarray(w, dtype=theano.config.floatX), borrow=True)
            hbias_shared = theano.shared(numpy.asarray(hbias, dtype=theano.config.floatX), borrow=True)
            vbias_shared = theano.shared(numpy.asarray(vbias, dtype=theano.config.floatX), borrow=True)
            ret_val = [w_shared, hbias_shared, vbias_shared]
        else:
            ret_val = [w, hbias, vbias]
        return ret_val
                
                
    def log(self, msg):
        self.logger.log(msg)
    
    
    def save_RBM_model(self, layer_idx, file_name=None):
        rbm_layer = self.rbm_layers[layer_idx]
        w = rbm_layer.W.get_value(borrow=True)
        hbias = rbm_layer.hbias.get_value(borrow=True)
        vbias = rbm_layer.vbias.get_value(borrow=True)
        if file_name == None:
            file_name = '%s/L%g.mat'%(self.output_folder, layer_idx)
        savemat(file_name, {'w':w, 'hbias':hbias, 'vbias':vbias})
        self.log(file_name+' saved')

    

    def load_RMB_model(self, layer_idx, shared=1):
        file_name = '%s/L%g.mat'%(self.output_folder, layer_idx)
        self.log("Load mat file: "+file_name)
        dataset = io.loadmat(file_name)
        w = dataset['w']
        hbias = dataset['hbias']
        hbias = hbias.reshape(hbias.shape[1])
        vbias = dataset['vbias']
        vbias = vbias.reshape(vbias.shape[1])
        if shared == 1:
            w_shared = theano.shared(numpy.asarray(w, dtype=theano.config.floatX), borrow=True)
            hbias_shared = theano.shared(numpy.asarray(hbias, dtype=theano.config.floatX), borrow=True)
            vbias_shared = theano.shared(numpy.asarray(vbias, dtype=theano.config.floatX), borrow=True)
            ret_val = [w_shared, hbias_shared, vbias_shared]
        else:
            ret_val = [w, hbias, vbias]
        return ret_val
    
         
    def init_sigmoid_layers(self):
        #sigmoid layers
        self.sigmoid_layers = []        
        self.params = []
        for i in xrange(self.n_layers-1):            
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output
                
            if i<(self.n_layers-1)/2:
                rbm_layer = self.rbm_layers[i]
                n_in = rbm_layer.n_visible
                n_out = rbm_layer.n_hidden
                W = rbm_layer.W
                b = rbm_layer.hbias
            else:#inverse the layer
                rbm_layer = self.rbm_layers[self.n_layers-2-i]
                n_in = rbm_layer.n_hidden
                n_out = rbm_layer.n_visible
                W = theano.shared(numpy.asarray(rbm_layer.W.T.eval(), dtype=theano.config.floatX), borrow=True)
                b = rbm_layer.vbias
                
            sigmoid_layer = HiddenLayer(rng=self.numpy_rng,
                                        input=layer_input,
                                        n_in=n_in,
                                        n_out=n_out,
                                        W=W,
                                        b=b,
                                        activation=T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)                                
#         set_trace()
        self.errors = (self.sigmoid_layers[-1].output - self.x) ** 2
                
        self.save_model_mat('%s/DAE_pre%s.mat'%(self.output_folder, self.get_model_file()))
    
    def save_model(self, file_name=None):
        if file_name == None:
            file_name = '%s/DAE%s.mat'%(self.output_folder, self.get_model_file())
        self.save_model_mat(file_name)
    
    def get_model_file(self):
                
        if self.p!=0:
            file_str = '_p%g_s%g'%(self.p, self.sigma)
        else:
            file_str = ''
        for i in range(len(self.p_list)):
            if self.p_list[i] != 0:
                file_str = '%s_(L%g_p%g_s%g)'%(file_str, i , self.p_list[i], self.sigma_list[i])
        return file_str.strip()
    
    
    def save_model_mat(self, file_name):
        params = []
        for item in self.params:            
            param = item.get_value(borrow=True)            
            params.append(param)
#         set_trace()
        savemat(file_name, {'params':params})
        self.log(file_name+' saved')
               
        
    def get_gaussian_cost(self, x, p=0, sigma=1, mu=0.5):
        x_mean = T.mean(x, axis=0)
        cost = (1 / (sigma * T.sqrt(2 * numpy.pi)) * T.exp(-(x_mean - mu) ** 2 / (2 * sigma ** 2)))
        return p * cost
    
    
    def get_square_cost(self, x, p):        
        x_mean = T.mean(x, axis=0)
        cost = -(x_mean-0.5)**2 + 0.5        
        return p * cost
    
    
    def get_cost(self, p=0, sigma=1):
        # the last layer        
        z = self.sigmoid_layers[-1].output
        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        p_idx = len(self.sigmoid_layers)/2 - 1 #penalty layer, the middle layer
        
#         if p == 0:
#             cost = T.mean(L)
#         elif (p != 0) and (sigma == 0):# for square penalty
#             square_cost = self.get_square_cost(self.sigmoid_layers[p_idx].output, p)
#             cost = T.mean(L) + T.mean(square_cost)
#         elif(p != 0) and (sigma != 0):# for Gaussian penalty                        
#             gaussian_cost = self.get_gaussian_cost(self.sigmoid_layers[p_idx].output, p, sigma)
#             cost = T.mean(L) + T.mean(gaussian_cost)
                        
        return T.mean(L)
    
    def get_cost_updates(self, learning_rate=0.1, p=0, sigma=1):
        
        cost = self.get_cost(p, sigma)
        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
#         set_trace()
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        return (cost, updates)
 
    def pretraining_functions(self, train_set, batch_size):
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')       
        pretrain_fns = []
        for i in range(len(self.rbm_layers)):
            # get the cost and the updates list
            cost, updates = self.rbm_layers[i].get_cost_updates(learning_rate, persistent=None, k=1)
            fn = theano.function(
                inputs=[index,
                        learning_rate],
                outputs=cost,
                updates=updates,
                givens={self.x: train_set[index * batch_size: (index + 1) * batch_size]}

            )            
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns
    
    def build_finetune_functions(self, datasets, batch_size):
        
#         self.init_sigmoid_layers()#firstly, initial the sigmoid layer
        train_set, valid_set, test_set = datasets        
        n_valid_batches = valid_set.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set.get_value(borrow=True).shape[0] / batch_size
        
        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.scalar('lr')
        
        sae_cost, sae_updates = self.get_cost_updates(learning_rate=learning_rate,
                                                      p=self.p,
                                                      sigma=self.sigma)
    #     set_trace()
        train_fn = theano.function(
                inputs=[index,
                        learning_rate],
                outputs=sae_cost,
                updates=sae_updates,
                givens={self.x: train_set[index * batch_size: (index + 1) * batch_size]}
            )
        test_fn = theano.function(
                inputs=[index],
                outputs=sae_cost,
                givens={self.x: test_set[index * batch_size: (index + 1) * batch_size]}
            )
        valid_fn = theano.function(
                inputs=[index],
                outputs=sae_cost,
                givens={self.x: valid_set[index * batch_size: (index + 1) * batch_size]}
            )
        
        # Create a function that scans the entire validation set
        def valid_model():
            return [test_fn(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_model():
            return [valid_fn(i) for i in xrange(n_test_batches)]
        
        return train_fn, valid_model, test_model
    
def train_autoencoder(datasets,layers_sizes,output_folder,p_dict,sigma_dict):
    
    train_set, valid_set, test_set = datasets
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    pre_epochs=20
    fine_epochs=100
    finetune_lr = 0.001
    batch_size = 100
    down_epoch = 100
    n_train_batches = train_set.get_value(borrow=True).shape[0] / batch_size
    logger = mylogger(output_folder + '/log.log')
    logger.log('layer size: '+''.join(['%d '%i for i in layers_sizes]).strip())
    file_path = '../../Autoencoder_hinton'
    rng = numpy.random.RandomState(123)
    dae = autoencoder_hinton(numpy_rng=rng, layers_sizes=layers_sizes, output_folder=output_folder, logger=logger, file_path=file_path)
    start_time = strict_time() 

    p = p_dict['p']
    sigma = sigma_dict['sigma']
    logger.log('fine tuning...')
    if dae.finetune_train_flag == True:
        logger.log('SAE trained')
        return    
    logger.log('learning rate:%g, p:%g, sigma:%g' % (finetune_lr, p, sigma))
    train_fn, valid_model, test_model = dae.build_finetune_functions(datasets, batch_size)#must be here! after pre_train
    
    for epoch in xrange(1, fine_epochs + 1):
        epoch_time_s = strict_time()
        c = []
        for minibatch_index in xrange(n_train_batches):
            err = train_fn(minibatch_index, finetune_lr)
            c.append(err)
        logger.log('Training epoch %d, cost %.5f, took %f seconds ' % (epoch, numpy.mean(c), (strict_time() - epoch_time_s)))
        if epoch % down_epoch == 0:
            finetune_lr = 0.8 * finetune_lr
            logger.log('learning rate: %g' % (finetune_lr))
#         if epoch in epoch_list:
#             dae.save_model('%s/SAE_p%g_s%g_(e_%g).mat'%(output_folder, p, sigma, epoch))
    dae.save_model()
    logger.log('ran for %.2f m ' % ((strict_time() - start_time) / 60.))

  
def strict_time():
    if sys.platform == "win32":
        return strict_time()
    else:
        return time.time()
    
from data_mat import load_TIMIT, load_model_mat, save_model_mat
from logger import log_init, mylogger
from tools import set_trace

def train_spectrum_coding():
        
    output_folder = '2400_rmb'
    data_file = 'rbm_TIMIT_dr2_(N1)_split.mat'

#     if output_folder.find('300') != -1:
#         data_file = '300bps/TIMIT_train_split.mat'
#     elif output_folder.find('600') != -1:
#         data_file = '600bps/TIMIT_train_dr1_dr4_split.mat'
#     elif output_folder.find('1200') != -1:
#         data_file = '1200bps/TIMIT_train_dr1_dr2_split.mat'
#     elif output_folder.find('2400') != -1:
#         data_file = '2400bps/TIMIT_train_dr1_dr2_split.mat'
        
    datasets = load_TIMIT(data_file)
    train_set, valid_set, test_set = datasets
    p_list = [0]
    sigma_list = [0]
    input_dim = train_set.get_value(borrow=True).shape[1]    
#     layers_sizes = [input_dim,2000,1000,500,54,500,1000,2000,input_dim]
#     layers_sizes = [input_dim,2000,2000,54,2000,2000,input_dim]
    layers_sizes = [input_dim,500,input_dim]
#     layers_sizes = [input_dim,500,250,125,70,125,250,500,input_dim]
    for p in p_list:
        for sigma in sigma_list:
            p_dict = {'p_list': [0], 'p': p}
            sigma_dict = {'sigma_list':[0], 'sigma':sigma}
            train_autoencoder(datasets,layers_sizes,output_folder,p_dict,sigma_dict)
    
        
if __name__ == '__main__':
    
    train_spectrum_coding()
    
