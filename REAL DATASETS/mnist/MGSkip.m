function [ERR X0]= MGSkip(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,chi,p,data_ix,data_full,rho,K,opt,logi_reg2,R,noise)
%     eta = (1 - sqrt(1-rho^2))/(1 + sqrt(1+rho^2));
%     M00 = eye(numProcesses);
%     M0 = eye(numProcesses);
%     for k = 1: K
%         M1 = (1+eta) * W * M0 - eta * M00;
%         M00 = M0;
%         M0 = M1;
%     end
%     W_b = eye(numProcesses)-M0; 
%     max_Wb = max(eig(W_b));

    W_b = eye(numProcesses) - W^K;
    max_Wb = max(eig(W_b));

    X0 = zeros(numProcesses,d);
    Y0 = zeros(numProcesses,d);  
    
    ERR = zeros(1,numIter+1);
    
    
    ERR(1) = norm(X0-opt)/norm(opt);
    

    
    for i = 1:numIter
        epsilon = ((1/R)^i)*randn(1);

        % 分布式计算梯度
        
        grad  = zeros(numProcesses,d); 
        X = X0;
        for p_ix = 1 : numProcesses
            y_local = X(p_ix,:);
            data_local = data_ix{p_ix};
            grad(p_ix,:) = grad_fn(y_local',data_local)';
        end
        
        XT = X0 - tau * (grad + noise*randn(numProcesses,d)) - tau * Y0;
        
        if rand(1) <= p
             R = 1;
        else
            R = 0;
        end
        
        Y1 = Y0 + R * p * chi / (max_Wb*tau) * W_b * (XT + epsilon);
        
        
        X1 = prox_l1(XT - tau/p * (Y1-Y0),tau*logi_reg2);
        
        

        Y0 = Y1;
        X0 = X1;
        

        ERR(i+1) = norm(X0-opt)/norm(opt);
        
    end
    

    
end