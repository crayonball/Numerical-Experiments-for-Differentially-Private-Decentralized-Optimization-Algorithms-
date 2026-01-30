function [PLF] = MGSkip_st(Q,q,n,d,W,rho,MG_K,X0,Y0,alpha,p,opt,stop,epsilon)

%     if MG_K <= 1
%         chi = 0.1;
%         W_b = eye(n)-W;  
%         W_a = eye(n)-chi/2*W_b;
%     else
%         % multi - gossip
%         eta = (1 - sqrt(1-rho^2))/(1 + sqrt(1+rho^2));
%         M00 = eye(n);
%         M0 = eye(n);
%         for k = 1: MG_K
%             M1 = (1+eta) * W * M0 - eta * M00;
%             M00 = M0;
%             M0 = M1;
%         end
%         W_b = eye(n)-M0; 
%         W_a = eye(n)-1/2*W_b;
%     end
    


    W_b = eye(n) - W^MG_K;
    W_a = eye(n)-1/2*W_b;


    ERR = norm(X0-opt)/norm(opt);
    
    time = cputime;
    iter = 1;
    R = 1.1;

    while ERR >= stop
        epsilon = ((1/R)^iter)*randn(1)

        % 分布式计算梯度
        for k = 1:n
            grad(k,:) = X0(k,:)*Q(:,:,k) + q(k,:);
        end
        
        XT = X0 - alpha * grad - alpha * Y0;
        
        if rand(1) <= p
            R = 1;
        else
            R = 0;
        end
        
        X1 = (1-R)* XT + R * W_a * (XT+epsilon);
        Y1 = Y0 + p/alpha * (XT - X1);
        

        Y0 = Y1;
        X0 = X1;

        ERR = norm(X0-opt)/norm(opt);
        iter = iter + 1
    end
    
    PLF = iter * p;
    time = cputime - time;
    



end

