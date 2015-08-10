function [ LSD ] = calc_LSD( power1, power2 )
%calculate log spectral distortion (LSD) between power1 and power2
%power1, power2 are in dB
LSD = mean(sqrt(mean((power1-power2).^2,2)));

end
