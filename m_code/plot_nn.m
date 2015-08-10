function plot_nn( nn )
%PLOT_NN Summary of this function goes here
%   Detailed explanation goes here
for layer = 1:nn.n
    subplot(3,3,layer)
    histogram(nn.a{layer}(:,2:end),25,'Normalization','probability');    
    title(layer);
end

end

