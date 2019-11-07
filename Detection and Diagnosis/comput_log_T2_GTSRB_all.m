function [log_T2_normal,log_T2_test] = comput_log_T2_GTSRB_all(x_train_recon,x_test_recon,x_train,x_test)
gtsrb = 1;mnist = 0;cifar = 0;
mixnum = 3;
X_train = x_train_recon-x_train;
X_adv = x_test_recon-x_test;
if mnist
    NumVari = 784; NumTrain = 60000; Numtest=10000;
elseif cifar
    NumVari = 3072; NumTrain = 50000; Numtest=10000;
else
    NumVari = 6912; NumTrain = 39209; Numtest=12500;
end
train_data_reshape = reshape(X_train,[NumTrain,NumVari]);%784 for mnist; 3072 for cifar
test_data_reshape = reshape(X_adv(1:Numtest,:,:,:),[Numtest,NumVari]);
X_Normal = train_data_reshape(:,:);
x1= X_Normal;%   Traning data, normal operation
x = x1;
[NumSamp,NumVari]= size(X_Normal);
[U,S,V,flag]= svds(double(x)/sqrt(NumSamp-1),NumVari);%   singular value decomposition, x/sqrt(n-1)=USV'
% load SV_mnist
% load SV_gtsrb
sigma= zeros(NumVari,1);
for i= 1:NumVari
    sigma(i,1)= S(i,i);%   singular value
%     sigma(i,1)= S(i);%   singular value
end
lamda= sigma.^2;%   characteristic value
percent_explained= lamda/sum(lamda);%   contribution
sum_per= 0;
% sum_per_lim(1) = 0.80;
% sum_per_lim(2) = 0.85;
% sum_per_lim(3) = 0.90;
% sum_per_lim(4) = 0.95;
% sum_per_lim(5) = 0.995;
% sum_per_lim(6) = 0.9995;
% sum_per_lim(7) = 0.99995;
% sum_per_lim(8) = 0.999995;
% sum_per_lim(9) = 0.9999995;
% sum_per_lim(10) = 0.99999995;
% sum_per_lim(11) = 0.999999995;
sum_per_lim(1) = 0.95;
sum_per_lim(2) = 0.995;
sum_per_lim(3) = 0.9995;
sum_per_lim(4) = 0.99995;
sum_per_lim(5) = 0.999995;
sum_per_lim(6) = 0.9999995;
sum_per_lim(7) = 0.99999995;
sum_per_lim(8) = 0.999999995;
if mnist
    sum_per_lim(10)= 1;
end
sum_per_lim = sum_per_lim;
for j = 1:length(sum_per_lim)
    sum_per = 0;
    for i = 1:NumVari
        sum_per = sum_per+percent_explained(i,1);
        if sum_per >= sum_per_lim(j)
            PCs = i;
            PC_set(j) = i;
            if j == 1
                P_set{j}= V(:,[1:PC_set(j)]);
                PC_sets{j} = [1:PC_set(j)];
            elseif j < length(sum_per_lim)
                P_set{j}= V(:,[PC_set(j-1)+1:PC_set(j)]);%   loading matrix
                PC_sets{j} = [PC_set(j-1)+1:PC_set(j)];
            else 
                P_set{j}= V(:,PC_set(j-1)+1:end);%   loading matrix
                PC_sets{j} = PC_set(j-1)+1:NumVari;
            end
            break;
        end
    end
    Scores{j}= x*P_set{j};%   scores in Principal Component space, or latent variables
end

for k = 2:length(sum_per_lim)
   incr_ratio(k-1) =  (PC_set(k)-PC_set(k-1))/PC_set(k-1);
end
for k = 1:length(sum_per_lim)
    T2_normal(:,k) = zeros(NumSamp,1);
    for i= 1:NumSamp
        for j= 1:length(PC_sets{k})
            T2_normal(i,k)= T2_normal(i,k)+Scores{k}(i,j)*Scores{k}(i,j)/lamda(PC_sets{k}(j));
        end
    end
end
log_T2_normal = log(T2_normal);

x_1 = test_data_reshape;
[NumSampTest,NumVariTest]= size(x_1);

for j = 1:length(sum_per_lim)
    Score_Test{j}= x_1*P_set{j};%   scores of testing data
end
for k = 1:length(sum_per_lim)
    T2_test(:,k) = zeros(NumSampTest,1);
    for i= 1:NumSampTest
        for j= 1:length(PC_sets{k})
            T2_test(i,k)= T2_test(i,k)+Score_Test{k}(i,j)*Score_Test{k}(i,j)/lamda(PC_sets{k}(j))+eps;
        end
    end
end

log_T2_test = log(T2_test);