%获取语音数据
clear;
addpath('f:\matlab\common\voicebox');
data_train = 'E:\TIMIT_wav_8k\train';
dataDir1 = 'E:\TIMIT_wav_8k\train\dr1';
dataDir2 = 'E:\TIMIT_wav_8k\train\dr2';
dataDir3 = 'E:\TIMIT_wav_8k\train\dr3';
dataDir4 = 'E:\TIMIT_wav_8k\train\dr4';

% allfiles = char(find_wav(dataDir1),find_wav(dataDir2),find_wav(dataDir3),find_wav(dataDir4));
% allfiles = char(find_wav(dataDir1),find_wav(dataDir2));
allfiles = find_wav(dataDir2);
% allfiles = find_wav(data_train);

n=240;                          % DFT window length
% inc = n/2; % 50% overlap
% inc = 3*n/4; % 25% overlap
inc = n; % 0% overlap

win=sqrt(hamming(n,'periodic'));     % omit sqrt if OV=4
win=win/sqrt(sum(win(1:inc:n).^2));      % normalize window

data_phase = [];%phase
data_power = [];%power in dB
for idx = 1:size(allfiles,1)
    fileName = allfiles(idx,:);
    [y, fs] = audioread(fileName);
    z=enframe(y,win,inc);
    f = rfft(z,n,2);
    mag = abs(f);
    pha = angle(f);
    power_dB = mag2db(mag);
    data_phase = [data_phase;pha];
    data_power = [data_power;power_dB];
end
% data_power = combineData(data_power, 8);
load('../data/TIMIT_train_(N1)_mu_sigma.mat');
[power_norm] = normalizeData(data_power,power_mu, power_sigma);
save('../data/TIMIT_dr2_(N1).mat','data_phase','data_power','power_norm', 'power_mu', 'power_sigma');
