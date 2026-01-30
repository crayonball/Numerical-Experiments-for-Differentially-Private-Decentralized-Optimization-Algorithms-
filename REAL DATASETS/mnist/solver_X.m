function [test_acc,ERR] = solver_X(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,data_ix,data_full,logi_reg2,opt,settingName,in_K,R,noise,X_test,n_test,label)

    W_b = 0.5*(eye(numProcesses)-W); 
    W_b = sparse(W_b);
    W_a = 0.5*(eye(numProcesses)+W);
    W_a = sparse(W_a);

    W_s = W^in_K;



    if(strcmp(settingName,'NIDS'))
        A = W_a;
        B = W_a;
        C = W_b;
        sigma = 0.1;
    elseif(strcmp(settingName,'EXTRA'))
        A = W_a;
        B = eye(numProcesses);
        C = W_b;
        sigma = 0.12;
    elseif(strcmp(settingName,'DIGing'))
        A = W^2;
        B = eye(numProcesses);
        C = (eye(numProcesses)-W)^2;
        sigma = 0.13;
    elseif(strcmp(settingName,'GT'))
        A = W^2;
        B = W^2;
        C = (eye(numProcesses)-W)^2;
        sigma = 0.09;
    elseif(strcmp(settingName,'SONATA'))
        A = W_s^2;
        B = W_s^2;
        C = (eye(numProcesses)-W_s)^2;
        sigma = 1;
    elseif(strcmp(settingName,'MGED'))
        A = W_s;
        B = W_s;
        C = eye(numProcesses)-W_s;
        
    end
        

    
    X0 = zeros(numProcesses,d);
    Y0 = zeros(numProcesses,d);  
    
    ERR = zeros(1,numIter+1);
    
    ERR(1) = norm(X0-opt)/norm(opt);
    grad  = zeros(numProcesses,d); 

    test_acc = zeros(1,numIter);

    
    ERR(1) = norm(X0-opt)/norm(opt);
    

    
    for i = 1:numIter

        %epsilon = ((1/R)^i)*randn(1);

        % 分布式计算梯度
        X = X0;
        for p_ix = 1 : numProcesses
            y_local = X(p_ix,:);
            data_local = data_ix{p_ix};
            grad(p_ix,:) = grad_fn(y_local',data_local)';
        end
        
        XT = A * X0 - tau * B * grad - Y0;
        Y1 = Y0 + sigma * C * XT;
        X1 = XT;

       
        Y0 = Y1;
        X0 = X1;

        x_mean = mean(X0);
        Ytest_pred = linear_prediction(X_test,x_mean'); %预测这次迭代结果x产生得标签值
        test_acc(i) = sum(Ytest_pred == label)/n_test; %记录准确率
        
        ERR(i+1) = norm(X0-opt)/norm(opt);
    end
    

    
end