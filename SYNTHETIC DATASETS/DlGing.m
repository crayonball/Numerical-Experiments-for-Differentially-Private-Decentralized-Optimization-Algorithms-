function [PLF,time] = DlGing(Q,q,n,d,W,rho,X0,Y0,alpha,opt,stop,epsilon)
    grad0 = zeros(n,d);
    W = (eye(n)+W)/2;
    ERR = norm(X0-opt)/norm(opt);

    time = cputime;
    iter = 1;

    R = 1.1;

    X00 = X0;
    Z0 = X0 - alpha * grad0;
    X0 = Z0;


    while ERR >= stop
        % 分布式计算梯度
        epsilon = ((1/R)^iter)*randn(1)
        for k = 1:n
            grad1(k,:) = X0(k,:)*Q(:,:,k) + q(k,:);
        end

        Z1 = Z0 -X0 + W*(2*X0-X00)-alpha*(grad1-grad0);
        X1 = Z1;
    
        Z0 = Z1;
        X00 = X0;
        X0 = X1;
        grad0 = grad1;
        ERR = norm(X0-opt)/norm(opt)
        iter = iter + 1;
    end
    PLF = iter*1.1;
    time = cputime - time;

end
