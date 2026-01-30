function ERR= KSkip_ERR(Q,q,n,d,W,MG_K,alpha,chi,p,opt,numIter)

    W_b = chi*(eye(n) - W^MG_K);

    X0 = zeros(n,d);
    Y0 = zeros(n,d); 
    grad  = zeros(n,d); 
    
    ERR = zeros(1,numIter+1);
    R = 1.05;
    
    
    ERR(1) = norm(X0-opt)/norm(opt);

    for i = 1:numIter
        epsilon = 10*((1/R)^i)*randn(1);
        % epsilon = 0;

        % 分布式计算梯度
        
        for k = 1:n
            grad(k,:) = X0(k,:)*Q(:,:,k) + q(k,:);
        end

        XT = X0 - alpha * grad - alpha * Y0;

        if rand(1) <= p
            Y1 = Y0 + W_b * (XT + epsilon);
            X1 = XT - alpha/p*(Y1-Y0);
        else
            Y1 = Y0;
            X1 = XT;
        end
        

        Y0 = Y1;
        X0 = X1;

        ERR(i+1) = norm(X0-opt)/norm(opt);

    end
   
end