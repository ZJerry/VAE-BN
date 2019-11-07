function S = mysum2(BIPv)
[~,D] = size(BIPv);
for i = 1:D
%    [A,B] = size(BIPv{i});
   S{i} = sum(BIPv{i}(:));
end
