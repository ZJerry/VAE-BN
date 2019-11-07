function Snorm = mysum_norm3(BIPv)
[C,D] = size(BIPv);
if sum(sum(BIPv)) == 0
    Snorm = 1/(C*D)*ones(C,D);
else
    Snorm = BIPv/(sum(sum(BIPv)));
end
% for i = 1:D
% %    [A,B] = size(BIPv{i});
%    S{i} = sum(BIPv{i}(:));
%    Snorm{i} = BIPv{i}/S{i};
% end