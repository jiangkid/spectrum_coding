clear;

addpath('F:\workspace\common');
path_base = '..\src\gb_rbm\';
data_file = '../data/rbm_TIMIT_dr2_(N1)_split.mat';
load(data_file)
[test_N,col]=size(test_set);
item_str = [path_base,'gb_rbm.mat'];
load(item_str);
batch_size = 100;

w1=[params{1}; params{2}];
w2=[params{1}'; params{3}];
for i=1:test_N/batch_size
    data = test_set((i-1)*batch_size+1:i*batch_size,:);
    data = [data  ones(batch_size,1)];
    w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(batch_size,1)];
%     dataout = 1./(1 + exp(-w1probs*w2));
    dataout = w1probs*w2;
    subplot(2,2,1);histogram(data(:,1:end-1),25,'Normalization','probability');
    subplot(2,2,2);histogram(w1probs(:,1:end-1),25,'Normalization','probability');
    subplot(2,2,3);histogram(dataout(:,1:end-1),25,'Normalization','probability');%xlim([0,1]);
end