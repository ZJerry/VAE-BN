function P_L = mychi2cdf2(Mahdis,D)
[A,B] = size(Mahdis);
for j = 1:A
    for k = 1:B
        P_L(j,k) = chi2cdf(Mahdis(j,k),D);
    end
end