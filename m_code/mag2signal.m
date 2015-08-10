%calc signal magnitude, and recover
function pesq_mos = mag2signal(fileName,n)
[inSpeech, fs] = audioread(fileName); % read the wavefile
win = hamming(n,'periodic');
z=enframe(inSpeech,win,3*n/4);
mag_org = abs(rfft(z'));
%线性插值
mag = mag_org';
[frameNum, col] = size(mag);
interpNum = 3;%插值个数
mag_interp = zeros(frameNum+(frameNum-1)*interpNum, n/2+1);
mag_interp(1:(interpNum+1):end,:) = mag;
for frameIdx = 1:(frameNum-1)
    mag_interp((frameIdx*(interpNum+1)-(interpNum-1)):frameIdx*(interpNum+1),:) = interp1([1,interpNum+2], mag(frameIdx:frameIdx+1,:), (2:interpNum+1)); %interp1 行线性插值
end
mag_interp = mag_interp';
[sig_r,] = LSEE(mag_interp,win,16/3);
if (length(inSpeech)/length(sig_r) > 1.1) || (length(inSpeech)/length(sig_r) < 0.9)
    error('length error');
end
pesq_mos = pesq(inSpeech, sig_r', fs);
end