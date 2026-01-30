function [PLF,time] = ADMM(Q,q,n,d,W,rho,X0,Y0,alpha,opt,stop,epsilon)

grad = zeros(n,d);
Lap = eye(n)-W;
W = eye(n)- Lap/10; 

ERR = norm(X0-opt)/norm(opt);

time = cputime;
iter = 1;
R = 1.1;

while ERR >= stop
    epsilon = ((1/R)^iter)*randn(1);
    % 分布式计算梯度
    for k = 1:n
        grad(k,:) = X0(k,:)*Q(:,:,k) + q(k,:);
    end
    Y1 = Y0 - (1/alpha) * Lap/8* X0;
    X1 = W * (X0 + epsilon) - alpha * (grad -Y1);

    Y0 = Y1;
    X0 = X1;
    ERR = norm(X0-opt)/norm(opt);
    iter = iter + 1;
end

PLF = iter*0.3*0.65;
time = cputime - time;

end