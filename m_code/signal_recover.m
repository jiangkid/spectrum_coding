%–≈∫≈ª÷∏¥/MFCC”Ô“Ù÷ÿΩ®≤‚ ‘
clear;
addpath('f:\matlab\common\voicebox');
addpath('F:\matlab\MFCC_Coding\Ogre Toolbox');
filePath = 'E:\TIMIT_wav_8k\test';
% filePath = 'F:\”Ô“Ùø‚\ZTE_speech\clean';
allfiles = find_wav(filePath);
fileNum = size(allfiles, 1);
num = 100;
fileIdx = randperm(fileNum, num);%ÀÊª˙—°‘Ò100∏ˆ
PESQ_all = zeros(num, 1);
parfor i = 1:num
    fileName = allfiles(fileIdx(i),:);    
    PESQ_all(i) = mag2signal(fileName,240);    
end
[value,idx] = max(PESQ_all);
fprintf('%s, PESQ: %.2f\n', allfiles(idx,:), value);
fprintf('mean PESQ: %.3f\n', mean(PESQ_all));
rmpath('f:\matlab\common\voicebox');
rmpath('F:\matlab\MFCC_Coding\Ogre Toolbox');