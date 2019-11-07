function P_L = mychi2cdf(Mahdis,D)
for i = 1:D
    [A,B] = size(Mahdis{i});
    for j = 1:A
        for k = 1:B
            P_L{i}(j,k) = chi2cdf(Mahdis{i}(j,k),1);
        end
    end
end