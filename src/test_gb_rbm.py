
import numpy, os
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from data_mat import load_TIMIT
from tools import set_trace
from scipy.io import loadmat, savemat
from gb_rbm import GBRBM
from logger import log_init, mylogger

def train_gb_rbm(batch_size=100,epochs=50):
    output_folder = 'gb_rbm'
    data_file = 'rbm_TIMIT_dr2_(N1)_split.mat'
    
    datasets = load_TIMIT(data_file)
    train_set, valid_set, test_set = datasets
    numpy_rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        
    input_dim = train_set.get_value(borrow=True).shape[1]
    layers_sizes = [input_dim,70,input_dim]
    input_x = T.matrix(name='x')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    logger = mylogger(output_folder + '/log.log')
    gb_rbm_layer = GBRBM(numpy_rng=numpy_rng,
                              theano_rng=theano_rng,
                              input=input_x,
                              n_visible=layers_sizes[0],
                              n_hidden=layers_sizes[1])
    
    index = T.lscalar('index')  
    momentum = T.scalar('momentum')
    learning_rate = T.scalar('lr') 
    # number of mini-batches
    n_batches = train_set.get_value(borrow=True).shape[0] / batch_size
    # start and end index of this mini-batch
    batch_begin = index * batch_size
    batch_end = batch_begin + batch_size
        
    r_cost, fe_cost, updates = gb_rbm_layer.get_cost_updates(batch_size, learning_rate,
                                                            momentum, weight_cost=0.0002,
                                                            persistent=None, k = 1)

            # compile the theano function
    fn = theano.function(inputs=[index,
                              theano.Param(learning_rate, default=0.0001),
                              theano.Param(momentum, default=0.5)],
                              outputs= [r_cost, fe_cost],
                              updates=updates,
                              givens={input_x: train_set[batch_begin:batch_end]})
    r_c, fe_c = [], []  # keep record of reconstruction and free-energy cost
    for epoch in range(epochs):
        for batch_index in xrange(n_batches):  # loop over mini-batches
            [reconstruction_cost, free_energy_cost] = fn(index=batch_index)
            r_c.append(reconstruction_cost)
            fe_c.append(free_energy_cost)
        logger.log('pre-training, epoch %d, r_cost %f, fe_cost %f' % (epoch, numpy.mean(r_c), numpy.mean(fe_c)))
    
        params = []
        for item in gb_rbm_layer.params:
            params.append(item.get_value(borrow=True))
    savemat(output_folder+'/gb_rbm.mat', {'params':params})
        
if __name__ == '__main__':
    train_gb_rbm()
