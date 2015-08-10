function [ nn ] = load_model_SAE( file_name,flag)
%LOAD_MODEL Summary of this function goes here
%   Detailed explanation goes here
load(file_name);% w, b, b_prime
% % ========define of network=======
paramSize = size(params,2);
net_size = [size(params{1},1)];
for i=1:paramSize/2
    net_size = [net_size, size(params{2*i},2)];
end

nn = nnsetup_v3(net_size);
nn.Nframe = net_size(1)/121;
if nargin == 2 && flag == 1
    nn.output = 'linear';
end
nn.learningRate = 0.1;
nn.threshold = 0.5;
for i=1:paramSize/2
    nn.W{i}  = [params{i*2}',params{i*2-1}'];
end

end

