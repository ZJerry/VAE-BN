function Y_cat = to_catnum(Y)
[N, classnum] = size(Y);
for i = 1:N
   [~,Y_cat(i)] = max(Y(i,:));
end