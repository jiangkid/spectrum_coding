function [ ret_val ] = spectrum_codec( speech_in, nn )
%MFCC_CODEC Summary of this function goes here
%   Detailed explanation goes here
mu = nn.mu;
sigma = nn.sigma;
n=240;                          % DFT window length
% inc = n/2; % 50% overlap
% inc = 3*n/4; % 25% overlap
inc = n; % 0% overlap

win=sqrt(hamming(n,'periodic'));     % omit sqrt if OV=4
win=win/sqrt(sum(win(1:inc:n).^2));      % normalize window

fs = 8000;
%fft
z=enframe(speech_in,win,inc);
f = rfft(z,n,2);
mag_org = abs(f);
pha = angle(f);
power_dB = mag2db(mag_org);

data_power = power_dB;
[data_m,data_dim] = size(data_power);
N = nn.Nframe;
m = fix(data_m/N);
comb_data = zeros(m, data_dim*N);
for i = 1:m
    comb_data(i,:) = reshape(data_power((i-1)*N+1:i*N,:)', 1, data_dim*N);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(nn.output,'sigm')
    [comb_data_norm,~,~] = normalizeData(comb_data,mu,sigma);
elseif strcmp(nn.output, 'linear')
    [comb_data_norm,~,~] = rbm_normalizeData(comb_data,mu,sigma);
end
if nn.binary == 1
nn_out = nn_codec_v3(nn, comb_data_norm);
else
nn_out = nnff_v3(nn, comb_data_norm,comb_data_norm);
end
% histogram(nn_out.a{3}(:,2:end));
%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(nn.output,'sigm')
    [comb_data_r] = normalizeData_r(nn_out.a{end},mu,sigma);
elseif strcmp(nn.output, 'linear')
    [comb_data_r,~,~] = rbm_normalizeData(nn_out.a{end},mu,sigma,1);    
end
% [comb_data_r] = normalizeData_r(comb_data_norm,mu,sigma);
data_power_r = zeros(m*N, data_dim);
for i = 1:m
    data_power_r((i-1)*N+1:i*N,:) = reshape(comb_data_r(i,:),data_dim,N)';
end
mag_r = db2mag(data_power_r);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%线性插值
mag = mag_r;
[frameNum, col] = size(mag);
interpNum = 3;%插值个数
mag_interp = zeros(frameNum+(frameNum-1)*interpNum, n/2+1);
mag_interp(1:(interpNum+1):end,:) = mag;
for frameIdx = 1:(frameNum-1)
    mag_interp((frameIdx*(interpNum+1)-(interpNum-1)):frameIdx*(interpNum+1),:) = interp1([1,interpNum+2], mag(frameIdx:frameIdx+1,:), (2:interpNum+1)); %interp1 行线性插值
end
mag_r = mag_interp';

%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isfield(nn,'LSD')&& (nn.LSD == 1)     
    ret_val = calc_LSD(data_power_r, data_power);
else
    [outSpeech,~] = LSEE(mag_r,win,4);
    ret_val = outSpeech'; %return the speech
    if (length(speech_in)/length(outSpeech) > 1.2) || (length(speech_in)/length(outSpeech) < 0.8)
        error('length error');
    end
end

end

