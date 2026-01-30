function [ERR opt]= SOLVER(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,data_ix,data_full,logi_reg2)


    W_b = 0.5*(eye(numProcesses)-W); 
    W_b = sparse(W_b);
    W_a = 0.5*(eye(numProcesses)+W);
    W_a = sparse(W_a);
    
    A = W_a;
    B = A;
    C = W_b;
    
    
    X0 = ones(numProcesses,d);
    Y0 = zeros(numProcesses,d);  
    
    ERR = zeros(1,numIter+1);

    
    for i = 1:numIter
        % 分布式计算梯度
        grad  = zeros(numProcesses,d); 
        X = X0;
        for p_ix = 1 : numProcesses
            y_local = X(p_ix,:);
            data_local = data_ix{p_ix};
            grad(p_ix,:) = grad_fn(y_local',data_local)';
        end
        
        XT = A * X0 - tau * B * grad - Y0;
        Y1 = Y0 + C * XT;
        X1 = prox_l1(XT,tau*logi_reg2);
        
       
        ERR(i) = norm(X1-X0);
        
        Y0 = Y1;
        X0 = X1;
        
        
    end
    
    
    opt = X0;
    
end