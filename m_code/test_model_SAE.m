clear;
% close all;
% addpath(genpath('F:\matlab\DeepLearning\DeepLearnToolbox\trunk'));
% addpath('F:\matlab\DeepLearning\speech coding');
addpath('F:\workspace\common');
path_base = '..\src\mfs_300\';
data_file = '../data/mfs_train_(N8).mat';
% if strfind(path_base,'300')
%     data_file = '../data/300bps/TIMIT_train_split.mat';
% elseif strfind(path_base,'600')
%     data_file = '../data/600bps/TIMIT_train_dr1_dr4_split.mat';
% elseif strfind(path_base,'1200')
%     data_file = '../data/1200bps/TIMIT_train_dr1_dr2_split.mat';
% elseif strfind(path_base,'2400')
%     data_file = '../data/TIMIT_train_(N8)_split.mat';
% end
load(data_file)
test_data = test_set(1:100,:);

% epoch_list = [10];
p_list = [3];
% for s_idx = 1:length(sigma_list)
% item_str = [path_base,'SAE_p0_s1_(L1_p0_s1)_(L2_p0_s1_(L1_p0_s1))_(c500).mat'];
% item_str = [path_base,'SAE_pre_(L2_p10_s0.3).mat'];
% for p_idx = 1:length(p_list)
% % for e = 1:10
% item_str = [path_base,'SAE_p',mat2str(p_list(p_idx)),'_s0_(L3_p1_s).mat'];%
% % item_str = [path_base,'SAE_(e_',mat2str(e),').mat'];
% nn = load_model_SAE(item_str);
% nn = nnff(nn,test_data,test_data);
% layer_idx = round(nn.n/2);
% figure;histogram(nn.a{layer_idx}(:,2:end),25,'Normalization','probability');xlim([0,1]);
% % title(p_list(p_idx))
% % title(e)
% end
p = 3;
% item_str = [path_base,'SAE_p',mat2str(p),'_s0_(L3_p1_s0).mat'];%
item_str = [path_base,'DAE.mat'];
nn = load_model_SAE(item_str,1);
nn = nnff_v3(nn,test_data,test_data);
% nn = nn_codec_v3(nn,test_data);
figure;set(gcf,'Position',[100 300 800 500]);
plot_nn(nn);
% nn = nn_codec_v3(nn,test_data);
% figure;set(gcf,'Position',[100 300 800 500]);
% plot_nn(nn);
% end