clear;
addpath('f:\matlab\common');
addpath('f:\matlab\common\voicebox');
addpath('F:\workspace\common');
% addpath('F:\matlab\DeepLearning\speech coding');
addpath('F:\matlab\MFCC_Coding\harmonic_comm');
addpath('F:\matlab\MFCC_Coding\Ogre Toolbox');

path_base = '..\src\rbm_300\';
% data_file = '../data/TIMIT_train_(N8)_mu_sigma.mat';
data_file = '../data/rbm_TIMIT_train_(N8)_mu_sigma.mat';
% if strfind(path_base,'300')
%     data_file = '../data/300bps/TIMIT_train_mu_sigma.mat';
% elseif strfind(path_base,'600')
%     data_file = '../data/600bps/TIMIT_train_mu_sigma.mat';
% elseif strfind(path_base,'1200')
%     data_file = '../data/1200bps/TIMIT_train_mu_sigma.mat';
% elseif strfind(path_base,'2400')
%     data_file = '../data/TIMIT_train_(N8)_mu_sigma.mat';
% end
load(data_file)
rng('default');rng(0);

p_list = [0.03];
sigma_list = [0];
results = cell(length(p_list),length(sigma_list));
results_pesq = cell(length(p_list), length(sigma_list));
results_segsnr = cell(length(p_list), length(sigma_list));
for p_idx = 1:length(p_list)
for s_idx = 1:length(sigma_list)
    item_str = [path_base,'DAE_p',mat2str(p_list(p_idx)),'_s',mat2str(sigma_list(s_idx)),'.mat'];
%     item_str = [path_base,'SAE_p',mat2str(p_list(p_idx)),'_s',mat2str(sigma_list(s_idx)),'_(L1_p1_s0).mat'];
%     item_str = [path_base,'DAE.mat'];
    if(exist(item_str,'file')==0) continue;end
    nn_SAE = load_model_SAE(item_str,1);
    nn_SAE.mu = power_mu;
    nn_SAE.sigma = power_sigma;
%     v1 = 0; v2 = 0;
%     results{p_idx,s_idx} = [v1, v2];
%     [float_score, binary_score] = speech_lsd_test(nn_SAE);
%     results{p_idx, s_idx} = [float_score, binary_score];
%     nn_SAE.threshold = 0.42;
    [float_score, binary_score]  = spectrum_codec_test(nn_SAE);
    results_pesq{p_idx, s_idx} = [float_score.pesq, binary_score.pesq];
%     results_segsnr{p_idx, s_idx} = [float_score.segsnr, binary_score.segsnr];
end
end
diary off;
% save('results_SAE.mat','results');
