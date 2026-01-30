clear all
clc;
close all;

%导入数据集
[X,label] = readIjcnn("ijcnn_train_processd.txt");
[X_test,label_test] = readIjcnn("ijcnn_test_processd.txt");


% 创建全局参数
numProcesses = 4;
[n d] = size(X);
[n_test,d] = size(X_test);

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
logi_reg2 = 0.01;

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
tau = 1/L;
numIter = 4000;
[ERR opt]= SOLVER(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,data_ix,data_full,logi_reg2);
semilogy(ERR)
hold on

%%
numIter = 500;
R = 3;
noise = 0;
%%

%----------------------------------------------------------------------------------------------------------------
Alg_Name = 'NIDS';
tau = 2*1/L;
[Acc_NIDS,ERR_NIDS] = solver_X(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,data_ix,data_full,logi_reg2,opt,Alg_Name,in_K,R,noise,X,n,label);
semilogy(ERR_NIDS)
hold on

%%

%----------------------------------------------------------------------------------------------------------------
Alg_Name = 'EXTRA';
tau = 2*1/L;
[Acc_EXTRA,ERR_EXTRA] = solver_X(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,data_ix,data_full,logi_reg2,opt,Alg_Name,in_K,R,noise,X,n,label);
semilogy(ERR_EXTRA)
hold on
%%
tau = 2*1/L;
%----------------------------------------------------------------------------------------------------------------
Alg_Name = 'DIGing';
[Acc_DIGing,ERR_DIGing] = solver_X(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,data_ix,data_full,logi_reg2,opt,Alg_Name,in_K,R,noise,X,n,label);
semilogy(ERR_DIGing)
hold on

%%
tau = 2*1/L;
%----------------------------------------------------------------------------------------------------------------
Alg_Name = 'GT';
[Acc_GT,ERR_GT] = solver_X(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,data_ix,data_full,logi_reg2,opt,Alg_Name,in_K,R,noise,X,n,label);
semilogy(ERR_GT)
hold on
%%
T = 500;
t = 1:T;


figure;
semilogy(t,ERR_NIDS(t),t,ERR_EXTRA(t),t,ERR_DIGing(t),t,ERR_GT(t));
hold on

legend('PG-NIDS','EXTRA','DIGing','GT','location','northeast','Interpreter','Latex');
%%
%----------------------------------------------------------------------------------------------------------------
tau = 2*1/L;
Alg_Name = 'SONATA';
[Acc_MGED,ERR_MGED] = solver_X(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,data_ix,data_full,logi_reg2,opt,Alg_Name,in_K,R,noise,X_test,n_test,label_test);
semilogy(ERR_MGED)
hold on

%%
%----------------------------------------------------------------------------------------------------------------
% ADMM
tau = 2*1/L;
[Acc_ADMM,ERR_ADMM] = ADMM(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,data_ix,data_full,opt,R,noise,X_test,n_test,label_test); 
semilogy(ERR_ADMM) 
hold on
% semilogy(Acc_ADMM)
% hold on

%%
%----------------------------------------------------------------------------------------------------------------
chi = 0.1;
p = 1;
tau = 0.05;
[Acc_RandCom_p05,ERR_RandCom_p05] = RandCom(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,chi,p,data_ix,opt,R,noise,X_test,n_test,label_test);
% semilogy(ERR_RandCom_p05)
% hold on

semilogy(Acc_RandCom_p05)
hold on

%%
%----------------------------------------------------------------------------------------------------------------
%KSkip
tau = 2*1/L;
chi = 0.2;
p = 1;
[Acc_KSkip_p10,ERR_KSkip_p10] = KSkip(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,chi,p,data_ix,in_K,opt,R,noise,X,n,label);
% semilogy(ERR_KSkip_p10)
% hold on
semilogy(Acc_KSkip_p10)
hold on
%%
%----------------------------------------------------------------------------------------------------------------
tau = 0.3;
chi = 0.1;
p = 0.5;
[Acc_KSkip_p05,ERR_KSkip_p05] = KSkip(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,chi,p,data_ix,in_K,opt,R,noise,X,n,label);
% semilogy(ERR_KSkip_p05(1:1/p:end))
% hold on
semilogy(Acc_KSkip_p05)
hold on
%%

%----------------------------------------------------------------------------------------------------------------
tau = 0.3;
chi = 0.1;
p = 0.2;
[Acc_KSkip_p02,ERR_KSkip_p02] = KSkip(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,chi,p,data_ix,in_K,opt,R,noise,X,n,label);
semilogy(ERR_KSkip_p02(1:1/p:end))
hold on
semilogy(Acc_KSkip_p02)
hold on
%%
T = numIter;
t = 1:T;

figure;
semilogy(t,ERR_NIDS(t),t,ERR_Walkman(t),t,ERR_MGED(t),t,ERR_ADMM(t),t,ERR_RandCom_p05_me(t),t,ERR_KSkip_p10(t),t,ERR_KSkip_p05_me(t),t,ERR_KSkip_p02_me(t));
hold on

legend('NIDS','Walkman','MGED','ADMM','Randcom: p=0.5','KSKip: p=1','KSKip: p=0.5','KSKip: p=0.2','NumColumns',2,'location','northeast','Interpreter','Latex');

%%
T = 50;
t = 1:T;

figure;
semilogy(t,Acc_NIDS(t),t,Acc_Walkman(t),t,Acc_MGED(t),t,Acc_ADMM(t),t,Acc_RandCom_p05(t),t,Acc_KSkip_p10(t),t,Acc_KSkip_p05(t),t,Acc_KSkip_p02(t));
hold on

legend('NIDS','Walkman','MGED','ADMM','Randcom: p=0.5','KSKip: p=1','KSKip: p=0.5','KSKip: p=0.2','NumColumns',2,'location','northeast','Interpreter','Latex');












