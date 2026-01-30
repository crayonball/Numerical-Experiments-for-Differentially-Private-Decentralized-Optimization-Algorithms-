function [State_X]= Dain(grad_fn,numProcesses,d,numIter,W_DS,alpha,data_ix)
%     multi - gossip
    
    W_b = eye(n)-W_DS; 


    X0 = zeros(numProcesses,d);
    Y0 = zeros(numProcesses,d);
    grad  = zeros(numProcesses,d);

    % ERR = zeros(1,numIter+1);

    State_X = zeros(numIter+1,d);
    
    beta = 1;

    State_X(1,:) = mean(X0);
    
    for i = 1:numIter

        X = X0;

        for p_ix = 1 : numProcesses
            y_local = X(p_ix,:);
            data_local = data_ix{p_ix};
            grad(p_ix,:) = grad_fn(y_local',data_local)';
        end
         
        XT = X0 - alpha * grad - alpha* Y0;
        Y1 = Y0 + beta * W_b * XT;
        X1 = XT - alpha * (Y1 - Y0);
       
        Y0 = Y1;
        X0 = X1;

        State_X(i+1,:) = mean(X0);
    end
    
end

