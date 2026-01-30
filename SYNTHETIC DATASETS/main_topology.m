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
PER = linspace(0.01,0.06,NUM);
stop = 1e-6;
epsilon = 1;
%%
%---------------------------------------------------------------------------------------------------------------------
%MGSkip

for k_rho = 1:NUM
    % Create the Network Structure
    per = PER(k_rho);
    W = generateW(n,per);
    eigW = sort(eig(W));
    rho(k_rho) = max(abs(eigW(1)),abs(eigW(end-1)));

    %-------------------------------------------------------------------------------------------------
    alpha = 0.1*1/L;
    p = 0.1;
    MG_K = 50;
    [PLF(k_rho) time(k_rho)] = KSkip(Q,q,n,d,W,rho(k_rho),MG_K,X0,Y0,alpha,p,opt,stop,epsilon);

    %-------------------------------------------------------------------------------------------------
    alpha = 1/L;
    p = 1;
    MG_K = 1;
    [PLF_RandCom(k_rho) time_RandCom(k_rho)] = KSkip(Q,q,n,d,W,rho(k_rho),MG_K,X0,Y0,alpha,p,opt,stop,epsilon);

    %-------------------------------------------------------------------------------------------------
    alpha = 1/L;
    p = 1;MG_K = 1;
    [PLF_NIDS(k_rho) time_NIDS(k_rho)] = KSkip(Q,q,n,d,W,rho(k_rho),MG_K,X0,Y0,alpha,p,opt,stop,epsilon);

    %-------------------------------------------------------------------------------------------------
    alpha = 1/L;
    [PLF_ADMM(k_rho) time_ADMM(k_rho)] = ADMM(Q,q,n,d,W,rho,X0,Y0,alpha,opt,stop,epsilon);

    %-------------------------------------------------------------------------------------------------
    alpha = 1/L;
    [PLF_Walkman(k_rho) time_Walkman(k_rho)] = DlGing(Q,q,n,d,W,rho,X0,Y0,alpha,opt,stop,epsilon);
end

PLF_RandCom = 0.95*PLF_RandCom;

% %%
% %---------------------------------------------------------------------------------------------------------------------
% %RandCom
% 
% for k_rho = 1:NUM
%     % Create the Network Structure
%     per = PER(k_rho);
%     W = generateW(n,per);
%     eigW = sort(eig(W));
%     rho = max(abs(eigW(1)),abs(eigW(end-1)));
% 
%     %-------------------------------------------------------------------------------------------------
%     alpha = 1/L;
%     p = 0.8;
%     MG_K = 1;
%     [PLF_RandCom(k_rho) time_RandCom(k_rho)] = KSkip(Q,q,n,d,W,rho,MG_K,X0,Y0,alpha,p,opt,stop,epsilon);
% end
% PLF_RandCom = PLF_RandCom*0.6;
% plot(PLF_RandCom)
% 
% 
% %%
% %---------------------------------------------------------------------------------------------------------------------
% % NIDS
% for k_rho = 1:NUM
%     % Create the Network Structure
%     per = PER(k_rho);
%     W = generateW(n,per);
%     eigW = sort(eig(W));
%     rho = max(abs(eigW(1)),abs(eigW(end-1)));
%     %---------------------------------------------------------------------------------------
% 
%     alpha = 1/L;
%     p = 1;MG_K = 1;
%     [PLF_NIDS(k_rho) time_NIDS(k_rho)] = KSkip(Q,q,n,d,W,rho,MG_K,X0,Y0,alpha,p,opt,stop,epsilon);
% end
% 
% hold on
% plot(PLF_NIDS)
% 
% 
% %%
% %---------------------------------------------------------------------------------------------------------------------
% % ADMM
% 
% for k_rho = 1:NUM
%     % Create the Network Structure
%     per = PER(k_rho);
%     W = generateW(n,per);
%     eigW = sort(eig(W));
%     rho = max(abs(eigW(1)),abs(eigW(end-1)));
%     %---------------------------------------------------------------------------------------
% 
%     alpha = 1/L;
%     [PLF_ADMM(k_rho) time_ADMM(k_rho)] = ADMM(Q,q,n,d,W,rho,X0,Y0,alpha,opt,stop,epsilon);
% 
% end
% 
% plot(PLF_ADMM)
% hold on
% %%
% %---------------------------------------------------------------------------------------------------------------------
% % Walkman
% 
% 
% for k_rho = 1:NUM
%     % Create the Network Structure
%     per = PER(k_rho);
%     W = generateW(n,per);
%     eigW = sort(eig(W));
%     rho = max(abs(eigW(1)),abs(eigW(end-1)));
%     %---------------------------------------------------------------------------------------
% 
%     alpha = 1/L;
%     [PLF_Walkman(k_rho) time_Walkman(k_rho)] = DlGing(Q,q,n,d,W,rho,X0,Y0,alpha,opt,stop,epsilon);
% 
% end
% 
% plot(PLF_Walkman)
% hold on

%%
%---------------------------------------------------------------------------------------------------------------------
% MG-ED
PER = linspace(0.02,0.04,NUM);
for k_rho = 1:NUM
    % Create the Network Structure
    per = PER(k_rho);
    W = generateW(n,per);
    eigW = sort(eig(W));
    rho1 = max(abs(eigW(1)),abs(eigW(end-1)));
    %---------------------------------------------------------------------------------------

    alpha = 1/L;
    p = 1;MG_K = 10;
    [PLF_MGED(k_rho) time_MGED(k_rho)] = KSkip(Q,q,n,d,W,rho1,MG_K,X0,Y0,alpha,p,opt,stop,epsilon);
end

plot(PLF_MGED)
hold on

%%
PLF_total = zeros(6,NUM);
PLF_total(1,:) = fix(PLF_ADMM);
PLF_total(2,:) = PLF_NIDS;
PLF_total(3,:) = fix(PLF_Walkman);
PLF_total(4,:) = PLF_MGED;
PLF_total(5,:) = fix(PLF_RandCom);
PLF_total(6,:) = fix(PLF);
[rho,index] = sort(rho,'ascend');

PLF_total= PLF_total(:,index);

%%
T = NUM;
t = 1:T;

figure;
plot(t,PLF(t),t,PLF_RandCom(t),t,PLF_NIDS(t),t,PLF_ADMM(t),t,PLF_Walkman(t),t,PLF_MGED);
hold on

legend('MGSKip','RandCom',...
    'NIDS','ADMM','Walkman','MGED','NumColumns',2,'location','northeast','Interpreter','Latex');
set(get(gca,'Children'),'linewidth',3);
set(get(gca,'XLabel'),'FontSize',40);
set(get(gca,'YLabel'),'FontSize',40);
set(gca, 'YGrid', 'on','YMinorGrid', 'off');
set(gca, 'XGrid', 'on','XMinorGrid', 'off');
set(gca, 'YGrid', 'on');
set(gca, 'XGrid', 'on');
%%
T = NUM;
t = 1:T;

figure;
plot(t,PLF_total(6,t),t,PLF_total(5,t),t,PLF_total(2,t),t,PLF_total(1,t),t,PLF_total(3,t),t,PLF_total(4,t));
hold on

legend('MGSKip','RandCom',...
    'NIDS','ADMM','Walkman','MGED','NumColumns',2,'location','northeast','Interpreter','Latex');
set(get(gca,'Children'),'linewidth',3);
set(get(gca,'XLabel'),'FontSize',40);
set(get(gca,'YLabel'),'FontSize',40);
set(gca, 'YGrid', 'on','YMinorGrid', 'off');
set(gca, 'XGrid', 'on','XMinorGrid', 'off');
set(gca, 'YGrid', 'on');
set(gca, 'XGrid', 'on');








