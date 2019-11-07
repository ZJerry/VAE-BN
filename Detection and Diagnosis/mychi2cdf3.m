function P_L = mychi2cdf3(Mahdis,D,cls)
[A,B] = size(Mahdis);
% for j = 1:A
for k = 1:B
    P_L(k) = chi2cdf(Mahdis(cls,k),D);
end
% end