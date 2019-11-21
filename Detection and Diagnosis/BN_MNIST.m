clear
close all
addpath('../data')
%% Load CVAE generated data package
load('Encoded_MNIST.mat')
%%
encoded = 1;
data = x_train_encoded;
testX = x_test_encoded(1:10000,:);
%%%%%
trainingX = data;
trainingC = double(y_train)+1;

% load pre-calculated T2 statistics for normal data
% load mnist_normal_t2_all
% or run the code to generate the normal training and test statistics.
[log_T2_normal,log_T2_test,S,V] = comput_log_T2_MNIST_normal_train_test(x_train_recon,x_test_recon,x_train,x_test);

data2 = log_T2_normal;
testX2 = log_T2_test(1:10000,:);

% Construct the Bayesian network model
dag = [ 0 1 1 1 1; 0 0 1 0 0; 0 0 0 0 0; 0 0 0 0 1; 0 0 0 0 0];
discrete_nodes = [1 2 4];
nodes = [1 : 5];
classnum = max(trainingC);
mixnum = 2;
varnum = size(trainingX,2);
varnum2 = size(data2,2);
node_sizes=[classnum mixnum varnum mixnum varnum2];
node_sizes = double(node_sizes);
bnet = mk_bnet(dag, node_sizes, 'discrete', discrete_nodes);
bnet.CPD{1} = tabular_CPD(bnet,1);
bnet.CPD{2} = tabular_CPD(bnet,2);
bnet.CPD{3} = gaussian_CPD(bnet, 3, 'cov_type', 'full');
bnet.CPD{4} = tabular_CPD(bnet,4);
bnet.CPD{5} = gaussian_CPD(bnet, 5, 'cov_type', 'full');

training= cell(5,length(trainingX));
training(3,:) = num2cell(trainingX',1);
training(1,:) = num2cell(trainingC,1);
training(5,:) = num2cell(data2',1);
%% Threshold
thd_set = 0.98;
%% Bayesian network training or loading a pretrained network
engine = jtree_inf_engine(bnet);
maxiter= 15;     %% The number of iterations of EM (max)
epsilon=1e-100; %% A very small stopping criterion
%===> Uncomment the following line for training
% [bnet2, ll, engine2] = learn_params_em(engine,training,maxiter,epsilon);
%===> or use a pretrained Bayesian network
load bnet2_mnist_m2_com

%% Test of detection
% == We will calculate the global/local detection indexes step-by-step ==
class0= cell(3,1); %% Create an empty cell array for observations
evidence=class0;   %% Start out with nothing observed
% For simplicity all samples will be inferenced in a batch run
%===> This is performed before we get the class label
for i=1:size(testX,1)
    if mod(i,1000)==0
       fprintf('1 Test Sample (Normal) inference %d\n',i) 
    end
    evidence{3}=testX(i,:)';
    evidence{5}=testX2(i,:)';
    [engine3, llts(i),pot{i},Mahdis{i}] = enter_evidence2(engine2, evidence);
    marg = marginal_nodes(engine3,1);
    p_n(i,:)=marg.T';
    [~,y_bnc(i)] = max(p_n(i,:));
end

s=struct(bnet2.CPD{1});
wts = s.CPT;
sm=struct(bnet2.CPD{2});
wtsm = sm.CPT;
smr=struct(bnet2.CPD{4});
wtsmr = smr.CPT;
sv = struct(bnet2.CPD{3});
wtv_mu = sv.mean;
wtv_cov = sv.cov;
svr = struct(bnet2.CPD{5});
wtv_mur = svr.mean;
wtv_covr = svr.cov;


for t=1:size(testX,1)
    if mod(t,1000)==0
       fprintf('2 Detection for Test Sample (Normal) %d\n',t) 
    end
    potreshp = pot{t}{3}.T;  % Feature potential (pot)
    for i = 1:classnum
        for j = 1:mixnum
            Pv{t}(i,j) = wts(i)*wtsm(i,j)* potreshp(i,j);  
        end
    end
    P_C{t} = mysum_norm3(Pv{t}); 
    P_L{t} = mychi2cdf2(Mahdis{t}{3},varnum); 
    BIP(t) = sum(sum(P_C{t}.*P_L{t}));  
    potreshpr = pot{t}{5}.T;  % Residual pot
    for i = 1:classnum
        for j = 1:mixnum
            Pvr{t}(i,j) = wts(i)*wtsmr(i,j)* potreshpr(i,j);  
        end
    end
    P_Cr{t} = mysum_norm3(Pvr{t}); 
    P_Lr{t} = mychi2cdf2(Mahdis{t}{5},varnum2); 
    BIPr(t) = sum(sum(P_Cr{t}.*P_Lr{t}));  
end
threshold = thd_set*ones(1,size(testX,1));  % Extended as a sequence for plot (if any)
%% Compute the False alarm (global index) on Normal test samples
fprintf('\nFalse alarm (global indexes) on Normal test samples:\n')
FAR_FBAP_G = length(find(BIP>threshold))/size(testX,1) % False alarm rate of FBAP_G index on normal test encodings
FAR_RBAP_G = length(find(BIPr>threshold))/size(testX2,1)  %  False alarm rate of RBAP_G index on normal test encodings
%% validation for conditional case
%===> This is performed after we get the class label
clear P_Ct P_Lt BIPtt Pvt
for t=1:size(testX,1)
    if mod(t,1000)==0
       fprintf('3 Detection for Test Sample (Normal) with a label %d\n',t) 
    end
    potreshp = pot{t}{3}.T;
    cls = y_test_pred(t)+1;
    for j = 1:mixnum
        Pvt{t}(j) = wtsm(cls,j)* potreshp(cls,j);  
    end
    P_Ct{t} = mysum_norm3(Pvt{t});
    if sum(P_Ct{t}) == 0
        P_Ct{t} = 1/mixnum*ones(1,mixnum);
    end
    P_Lt{t} = mychi2cdf3(Mahdis{t}{3},varnum,cls); 
    BIPtt(t) = sum(sum(P_Ct{t}.*P_Lt{t}));
    %%
    potreshpr = pot{t}{5}.T;
    for j = 1:mixnum
        Pvtr{t}(j) = wtsmr(cls,j)* potreshpr(cls,j);  
    end
    P_Ctr{t} = mysum_norm3(Pvtr{t}); 
    if sum(P_Ctr{t}) == 0
        P_Ctr{t} = 1/mixnum*ones(1,mixnum);
    end
    P_Ltr{t} = mychi2cdf3(Mahdis{t}{5},varnum2,cls); 
    BIPttr(t) = sum(sum(P_Ctr{t}.*P_Ltr{t}));
    
end
threshold = thd_set*ones(1,size(testX,1));
%% Compute the False alarm (local index) on Normal test samples
fprintf('\nFalse alarm (local indexes) on Normal test samples:\n')
FAR_FBAP_L = length(find(BIPtt>thd_set))/length(BIPtt)
FAR_RBAP_L = length(find(BIPttr>thd_set))/length(BIPttr)

% detected_normal = BIPtt> thd_set | BIPtt>thd_set | BIP>thd_set | BIPr>thd_set;% sum(detected_normal)/length(detected_normal)
% 
% ALL_FAR = sum(detected_normal)/length(detected_normal)
% 
% detected_L = BIPtt> thd_set | BIP>thd_set;
% L_FAR = sum(detected_L)/length(detected_L)
% 
% detected_R = BIPttr> thd_set | BIPr>thd_set;
% R_FAR = sum(detected_R)/length(detected_R)
%% %% Test of detection on adversarial case
clear P_Ct P_Lt Pvt
fprintf('Now dealing with attacked data samples...\n');

%% Please select the attacked data by uncommenting the line
dataname = 'mnist';
load('Adv_mnist_fgsm.mat');  adv_name = 'fgsm' ;       % fgsm attacked data
% load('Adv_mnist_cw.mat');    adv_name = 'cw' ;           % cw attacked data 
% load('Adv_mnist_bim-a.mat');   adv_name = 'bim-a' ;        % bim attacked data

tic
[log_T2_normal,log_T2_X_adv] = comput_log_T2_MNIST(x_train_recon,X_adv_decoder,x_train,X_adv,S,V);


testX2 = log_T2_X_adv;  % rename by a short one
Y_cat = to_catnum(Y);
Y_preds_cat = to_catnum(Y_preds);
testX = X_adv;          % rename by a short one
if encoded
    testX = X_adv_encoder;
end
class0= cell(3,1); %% Create an empty cell array for observations
evidence=class0;   %% Start out with nothing observed
for i=1:size(testX,1)
    if mod(i,1000)==0
       fprintf('4 Test Sample (Adversary) inference %d\n',i) 
    end
    evidence{3}=testX(i,:)';
    evidence{5}=testX2(i,:)';
    [engine3, llts(i),pot{i},Mahdis{i}] = enter_evidence2(engine2, evidence);
    marg = marginal_nodes(engine3,1);
    p_adv(i,:)=marg.T';
    [~,y_adv_bnc(i)] = max(p_adv(i,:));
end

s=struct(bnet2.CPD{1});
wts = s.CPT;
sm=struct(bnet2.CPD{2});
wtsm = sm.CPT;
smr=struct(bnet2.CPD{4});
wtsmr = smr.CPT;
svt = struct(bnet2.CPD{5});
wtv_mut = svt.mean;
wtv_covt = svt.cov;

%% Global validation
for t=1:size(testX,1)
    if mod(t,1000)==0
       fprintf('5 Detection for Test Sample (Adversary) %d\n',t) 
    end
    potreshp = pot{t}{3}.T;
    for i = 1:classnum
        for j = 1:mixnum
            Pv{t}(i,j) = wts(i)*wtsm(i,j)* potreshp(i,j);  
        end
    end
    P_C{t} = mysum_norm3(Pv{t}); 
    P_L{t} = mychi2cdf2(Mahdis{t}{3},varnum); 
    BIP2(t) = sum(sum(P_C{t}.*P_L{t}));  
    potreshpr = pot{t}{5}.T;
    for i = 1:classnum
        for j = 1:mixnum
            Pvr{t}(i,j) = wts(i)*wtsmr(i,j)* potreshpr(i,j);  
            for k = 1:size(testX2,2)
                Mahdisvt(t,i,j,k) = (testX2(t,k)-wtv_mut(k,i,j))*pinv(wtv_covt(k,k,i,j))*(testX2(t,k)-wtv_mut(k,i,j));
                P_LVt{t}(i,j,k) = chi2cdf(Mahdisvt(t,i,j,k),1);
            end
        end
    end
    P_Cr{t} = mysum_norm3(Pvr{t}); 
    P_Lr{t} = mychi2cdf2(Mahdis{t}{5},varnum2); 
    BIP2r(t) = sum(sum(P_Cr{t}.*P_Lr{t}));  
    for k = 1:size(testX2,2)
        BIP_V2(t,k) = sum(sum(P_Cr{t}.*P_LVt{t}(:,:,k)));
        if BIP_V2(t,k) < thd_set
            normal_subspaceset(t,k) = 1;
        else
            normal_subspaceset(t,k) = 0;
        end
    end
end

fprintf('\nAdversarial detection results of FBAP_G and RBAP_G:\n')
FDR_FBAP_G = length(find(BIP2>thd_set))/length(BIP2)  
FDR_RBAP_G = length(find(BIP2r>thd_set))/length(BIP2r)
%% Local validation
clear P_Ct P_Lt Pvt
for t=1:10000
    if mod(t,1000)==0
       fprintf('6 Detection for Test Sample (Adversary) with a label %d\n',t) 
    end
    potreshp = pot{t}{3}.T;
    cls = Y_preds_cat(t);
    for j = 1:mixnum
        Pvt{t}(j) = wtsm(cls,j)* potreshp(cls,j);  
    end
    P_Ct{t} = mysum_norm3(Pvt{t});
    if sum(P_Ct{t}) == 0
        P_Ct{t} = 1/mixnum*ones(1,mixnum);
    end
    P_Lt{t} = mychi2cdf3(Mahdis{t}{3},varnum,cls); 
    BIPtt2(t) = sum(sum(P_Ct{t}.*P_Lt{t}));
    %%
    potreshpr = pot{t}{5}.T;
    for j = 1:mixnum
        Pvtr{t}(j) = wtsmr(cls,j)* potreshpr(cls,j);  %p(i,m,x_k)
        %             Normfact(k) = Normfact(k) + BIPv(k,t);
    end
    P_Ctr{t} = mysum_norm3(Pvtr{t}); % p(i,m|x_k)
    if sum(P_Ctr{t}) == 0
        P_Ctr{t} = 1/mixnum*ones(1,mixnum);
    end
    P_Ltr{t} = mychi2cdf3(Mahdis{t}{5},varnum2,cls); % pL(x_k)
    BIPttr2(t) = sum(sum(P_Ctr{t}.*P_Ltr{t}));
end

%%
fprintf('\nThe total detection and diagnosis time on 10000 samples\n');
toc

threshold = thd_set*ones(1,size(testX,1));

fprintf('\nAdversarial detection results of FBAP_L and RBAP_L:\n')
FDR_FBAP_L = length(find(BIPtt2>thd_set))/length(BIPtt2)
FDR_RBAP_L  = length(find(BIPttr2>thd_set))/length(BIPttr2)
figure
bar(median(BIP_V2(BIPtt2> thd_set | BIPttr2>thd_set | BIP2>thd_set | BIP2r>thd_set,:)))
xlabel('Subspace (MNIST)')
ylabel('CCI')

detected = BIPtt2> thd_set | BIPttr2>thd_set | BIP2>thd_set | BIP2r>thd_set;
COMBINE_FDR = sum(detected)/length(detected)

detected_L = BIPtt2> thd_set | BIP2>thd_set;
LATENT_FDR = sum(detected_L)/length(detected_L)

detected_R = BIPttr2> thd_set | BIP2r>thd_set;
RESIDUAL_FDR = sum(detected_R)/length(detected_R)

filename = 'det_'+string(dataname)+'_'+string(adv_name)+'_all';
save(['../data/',char(filename)],'detected')