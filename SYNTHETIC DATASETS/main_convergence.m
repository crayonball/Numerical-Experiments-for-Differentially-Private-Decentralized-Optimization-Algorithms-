clear all;clc;

% 参数设置：n个节点//d维数据
n = 50; d = 3;

% Create the Network Structure
per = 1;
%W = generateW_Cycle(n,per); % Cycle graph
W = generateW(n,per);

eigW = sort(eig(W));
rho = max(abs(eigW(1)),abs(eigW(end-1)));

% generate date: objective function  obj_fn =  @(w)(1/2*w'*Q*w+q'*w);
cond_f = 10;

for i = 1:n
    VQ = diag(linspace(1/cond_f,1,d));
    UQ = orth(rand(d));
    
    % objective function
    Q(:,:,i) = UQ * VQ * UQ';
end
q(i,:) = randn(1,d);
L = 1;
mu = 1/cond_f;

%--------------------------------------------------------------------------------------------------------------------
% 调用 GM-Solver 求解
% 初始化
X0 = 10*randn(n,d);
Y0 = zeros(n,d);  
solver_numIter = 2000;
alpha = 1/L;
MG_K = 1;
[ERR opt]= MGSolver(Q,q,n,d,W,rho,MG_K,X0,Y0,solver_numIter,alpha);
%semilogy(ERR); hold on

%%
% 全局变量初始化
NUM = 5;

% 连边概率
PER = linspace(0.02,0.06,NUM);
numIter = 500;
epsilon = 1;

%%
p = 1;
alpha = 0.5*1/L;
chi = 0.5;
MG_K = 120;
for k_rho = 1:NUM
    % Create the Network Structure
    per = PER(k_rho);
    W = generateW(n,per);
    eigW = sort(eig(W));
    rho(k_rho) = max(abs(eigW(1)),abs(eigW(end-1)));
    Iter_KSkip_p10(k_rho,:) = KSkip_ERR(Q,q,n,d,W,MG_K,alpha,chi,p,opt,numIter);
    semilogy(Iter_KSkip_p10(k_rho,:))
    hold on
end

%%
p = 0.5;
alpha = 0.5*1/L;
chi = 0.5;
MG_K = 120;
for k_rho = 1:NUM
    % Create the Network Structure
    per = PER(k_rho);
    W = generateW(n,per);
    eigW = sort(eig(W));
    rho(k_rho) = max(abs(eigW(1)),abs(eigW(end-1)));
    for j = 1:20
        Err(j,:) = KSkip_ERR(Q,q,n,d,W,MG_K,alpha,chi,p,opt,numIter);
    end
    Iter_KSkip_p05(k_rho,:) = mean(Err);
    semilogy(Iter_KSkip_p05(k_rho,:))
    hold on
end
%%
p = 0.2;
alpha = 0.5*1/L;
chi = 0.4;
MG_K = 120;
for k_rho = 1:NUM
    % Create the Network Structure
    per = PER(k_rho);
    W = generateW(n,per);
    eigW = sort(eig(W));
    rho(k_rho) = max(abs(eigW(1)),abs(eigW(end-1)));
    for j = 1:20
        Err(j,:) = KSkip_ERR(Q,q,n,d,W,MG_K,alpha,chi,p,opt,numIter);
    end
    Iter_KSkip_p02(k_rho,:) = mean(Err);
    semilogy(Iter_KSkip_p02(k_rho,:))
    hold on
end

%%
for k_rho = 1:NUM
    semilogy(Iter_KSkip_p10(k_rho,:))
    semilogy(Iter_KSkip_p05(k_rho,:))
    semilogy(Iter_KSkip_p02(k_rho,:))
    hold on
end

%%
for k_rho = 1:NUM
    semilogy(ERR_KSkip_p10(k_rho,:))
    semilogy(ERR_KSkip_p05(k_rho,:))
    semilogy(ERR_KSkip_p02(k_rho,:))
    hold on
end