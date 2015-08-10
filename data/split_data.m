clear;
load('TIMIT_train_(N8).mat');
addpath('F:\workspace\common');
[power_norm, power_mu, power_sigma] = rbm_normalizeData(data_power);
save('rbm_TIMIT_train_(N8)_mu_sigma.mat', 'power_mu', 'power_sigma');
m = size(power_norm,1);
k = randperm(m);
train_set = power_norm(k(1:50000),:);
test_set = power_norm(k(50001:54000),:);
valid_set = power_norm(k(54001:58000),:);
save('rbm_TIMIT_train_(N8)_split.mat','train_set','test_set','valid_set');