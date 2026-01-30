clc;clear all;
% read train labels
fid = fopen('train-labels.idx1-ubyte', 'rb');
trainLabels = fread(fid, inf, 'uint8', 'l');
trainLabels = trainLabels(9:end);
fclose(fid);
% read train datas
fid = fopen('train-images.idx3-ubyte', 'rb');
trainImages = fread(fid, inf, 'uint8', 'l');
trainImages = trainImages(17:end);
fclose(fid);
trainData = reshape(trainImages, 784, size(trainImages,1) / 784)';


% Loading Data
NS = 60000; %代表读取多少条数据
d  = 784; %因为每个图片数据是20*20的，后来存放是拉成一个响亮的，所以d=400，来存放这个数据
n_test  = sum(trainLabels==0) + sum(trainLabels==1); %标签中 等于 0 和等于8 的标签数的和
X_test  = zeros(n_test, d); % n*d，每一行存放一个图片的数据
dig = zeros(n_test,1);  % 每一行存放一个图片的标签-1,1


%%%% select two classes %%%%% 
j=1;
for i=1:NS
    if((trainLabels(i) == 0) || (trainLabels(i) == 1))
        % 0 ---> -1 and 1 -----> 1
        dig(j) = trainLabels(i)*2-1; 
        % 把小x，即图片的第i个20*20的数据变成1*d维的，并赋值给X的第j行。
        X_test(j,:) = trainData(i,:);
        % 对X的第j行进行规则化(每行和为1)
        X_test(j,:) = X_test(j,:)/norm(X_test(j,:));
        j = j+1;
    end
end

label_raw = dig;
[label,I] = sort(label_raw,'ascend') ; %把label_raw升序排列存放到label中，I是个数组，存放的是排序之后每个位置对应的原始下标。
X = X_test(I,:); %X存放的是按照标签降序的数据 n*d


%%

% 创建全局参数
numProcesses = 5;
[n d] = size(X);

% Create the Network Structure
per = 0.05;
%W = generateW(numProcesses,per);
W = generateW_Cycle(numProcesses,per);
Sigma_MAX_W = max(eig(eye(numProcesses)-W));
eigW=eig(W);
rho = max(abs(eigW(1)),abs(eigW(end-1)));
% in_K = fix(1/sqrt(1-rho))-1;
in_K = 10;


data_ix = cell(numProcesses,1); %16*1的cell，每个cell存放一个agent所用的数据的下标
temp_ix = 1;
temp_inc = ceil(n/numProcesses); %算出每个agent平均有多少个数据的约值
%分发给每个智能体数据
for p_ix = 1:numProcesses
    ix_end = min(temp_ix+temp_inc-1,n);
    data_ix{p_ix} = (temp_ix:ix_end);
    temp_ix = ix_end +1;
end
% 这之后数据就分发到每个agent了
data_full = (1:n);




%------------------------------------------------------------------------------------------------------------------------
%@(w,ix)表示这个obj_fn函数的参数;grad_fn_stoch同理
logi_reg1 = 0.01;
logi_reg2 = 0;

for i = 1:numProcesses
    data_local = data_ix{i};
    flag = X(data_local,:)'*X(data_local,:);
    L_smooth(i) = numProcesses/n * max(eig(flag)) + logi_reg1 * 2;
end

L = max(L_smooth);


obj_fn =  @(w,ix) ( sum(log(1+exp(-label(ix).*(X(ix,:)*w))))/length(ix) + logi_reg1 * norm(w)^2 + logi_reg2*norm(w)); 
grad_fn = @(w,ix)( -X(ix,:)'*(label(ix)./(1+exp(label(ix).*(X(ix,:)*w))))/length(ix)+ 2 * logi_reg1 * w ); %光滑部分的梯度

%obj_fn =  @(w,ix) ( 0.5*(norm(X(ix,:)*w-label(ix),2))^2/length(ix) + logi_reg1 * norm(w)^2 + logi_reg2*norm(w)); 
%grad_fn = @(w,ix)( (X(ix,:))'*(X(ix,:)*w-label(ix))/length(ix)+ 2 * logi_reg1 * w ); %光滑部分的梯度
%


%%
% ------------------------------------------------------------------------------------------------------------------------
%solver
%
tau = 1.5*L;
numIter = 3800;
[ERR opt]= SOLVER(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,data_ix,data_full,logi_reg2);
semilogy(ERR)
hold on

%%
numIter = 600; 
R = 1.5;
noise = 0;
%%

%----------------------------------------------------------------------------------------------------------------
tau = 2/L;
Alg_Name = 'NIDS';
[Acc_NIDS,ERR_ADMM] = solver_X(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,data_ix,data_full,logi_reg2,opt,Alg_Name,in_K,R,noise,X,n_test,label);

semilogy(ERR_ADMM)
hold on
%%
semilogy(Acc_NIDS)
hold on
%%

%----------------------------------------------------------------------------------------------------------------
tau = 2/L;
Alg_Name = 'EXTRA';
[Acc_EXTRA,ERR_EXTRA] = solver_X(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,data_ix,data_full,logi_reg2,opt,Alg_Name,in_K,R,noise,X,n_test,label);

semilogy(ERR_EXTRA)
hold on
%%

%----------------------------------------------------------------------------------------------------------------
tau = 2/L;
Alg_Name = 'GT';
[Acc_GT,ERR_GT] = solver_X(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,data_ix,data_full,logi_reg2,opt,Alg_Name,in_K,R,noise,X,n_test,label);
semilogy(ERR_GT)
hold on
%%

%----------------------------------------------------------------------------------------------------------------
tau = 2/L;
Alg_Name = 'DIGing';
[Acc_DIGing,ERR_DIGing] = solver_X(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,data_ix,data_full,logi_reg2,opt,Alg_Name,in_K,R,noise,X,n_test,label);
semilogy(ERR_DIGing)
%%
T = 500;
t = 1:T;


figure;
semilogy(t,ERR_EXTRA(t),t,ERR_ADMM(t),t,ERR_DIGing(t),t,ERR_GT(t));
hold on

legend('PG-EXTRA','ADMM','DIGing','GT','location','northeast','Interpreter','Latex');

%%
%----------------------------------------------------------------------------------------------------------------
Alg_Name = 'MGED';
[Acc_MGED,ERR_MGED] = solver_X(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,data_ix,data_full,logi_reg2,opt,Alg_Name,in_K,R,noise,X,n_test,label);

semilogy(ERR_MGED)


%%
%----------------------------------------------------------------------------------------------------------------
% ADMM
tau = 0.16;
[Acc_ADMM,ERR_ADMM] = ADMM(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,data_ix,data_full,opt,R,noise,X,n_test,label); 

semilogy(ERR_ADMM(1:501))
% hold on
%%
% %----------------------------------------------------------------------------------------------------------------
R = 5;
chi = 0.5;
p = 1;
[Acc_RandCom_p05,ERR_RandCom_p05] = RandCom(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,chi,p,data_ix,opt,R,noise,X,n_test,label);
semilogy(ERR_RandCom_p05)
hold on

%%

%----------------------------------------------------------------------------------------------------------------
%KSkip
tau = 0.5;
chi = 0.1;
p = 1;
[X_KSkip_p10,Acc_KSkip_p10,ERR_KSkip_p10] = KSkip(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,chi,p,data_ix,in_K,opt,R,noise,X,n_test,label);
% semilogy(ERR_KSkip_p10)
% hold on
semilogy(Acc_KSkip_p10)
hold on

%%
%----------------------------------------------------------------------------------------------------------------
%KSkip
tau = 1;
chi = 0.1;
p = 0.5;
[X_KSkip_p05,Acc_KSkip_p05,ERR_KSkip_p05] = KSkip(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,chi,p,data_ix,in_K,opt,R,noise,X,n_test,label);
semilogy(Acc_KSkip_p05)
hold on
%%

chi = 0.1;
p = 0.2;
[X_KSkip_p02,Acc_KSkip_p02,ERR_KSkip_p02] = KSkip(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,chi,p,data_ix,in_K,opt,R,noise,X,n_test,label);

%%
load('mnist_pb5_er.mat')
T = 500;
t = 1:T;

figure;
semilogy(t,ERR_NIDS(t),t,ERR_Walkman(t),t,ERR_MGED(t),t,ERR_ADMM(t),t,ERR_RandCom_p05_me(t),t,ERR_KSkip_p10(t),t,ERR_KSkip_p05_me(t),t,ERR_KSkip_p02_me(t));
hold on

legend('NIDS','Walkman','MGED','ADMM','Randcom: p=0.5','KSKip: p=1','KSKip: p=0.5','KSKip: p=0.2','NumColumns',2,'location','northeast','Interpreter','Latex');










