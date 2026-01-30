function ERR = RandCom(Q,q,n,d,W,rho,MG_K,X0,Y0,solver_numIter,alpha,chi,p,opt)

    if MG_K <= 1
        W_b = eye(n)-W;  
        W_a = eye(n)-chi/2*W_b;
    else
        % multi - gossip
        eta = (1 - sqrt(1-rho^2))/(1 + sqrt(1+rho^2));
        M00 = eye(n);
        M0 = eye(n);
        for k = 1: MG_K
            M1 = (1+eta) * W * M0 - eta * M00;
            M00 = M0;
            M0 = M1;
        end
        W_b = eye(n)-M0; 
        W_a = eye(n)-1/2*W_b;
    end
    
    
    ERR = norm(X0-opt)/norm(opt)*ones(1,solver_numIter);
    
    
    for i = 1:solver_numIter
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
        
        X1 = (1-R)* XT + R * W_a * XT;
        Y1 = Y0 + p/alpha * (XT - X1);
        

        Y0 = Y1;
        X0 = X1;

        ERR(i+1) = norm(X0-opt)/norm(opt);
        
    end

end