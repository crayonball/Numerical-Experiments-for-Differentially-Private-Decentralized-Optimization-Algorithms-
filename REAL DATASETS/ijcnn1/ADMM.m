function [test_acc,ERR] = ADMM(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,data_ix,data_full,opt,R,noise,X_test,n_test,label_test)

    Lap = eye(numProcesses)-W;
    W = eye(numProcesses)- Lap/10;

    X0 = zeros(numProcesses,d);
    Y0 = zeros(numProcesses,d);  
    
    ERR = zeros(1,numIter+1);
    
    ERR(1) = norm(X0-opt)/norm(opt);
    grad  = zeros(numProcesses,d); 
    test_acc = zeros(1,numIter);

    for i = 1:numIter

        %epsilon = ((1/R)^i)*randn(1);
        % 分布式计算梯度
        X = X0;
        for p_ix = 1 : numProcesses
            y_local = X(p_ix,:);
            data_local = data_ix{p_ix};
            grad(p_ix,:) = grad_fn(y_local',data_local)';
        end
        Y1 = Y0 - (1/tau) * Lap/8* X0;
        %X1 = W * (X0 + epsilon) - tau * (grad + noise*randn(numProcesses,d) -Y1);
        X1 = W * X0 - tau * (grad -Y1);

        Y0 = Y1;
        X0 = X1;

        x_mean = mean(X0);
        Ytest_pred = linear_prediction(X_test,x_mean'); %预测这次迭代结果x产生得标签值
        test_acc(i) = sum(Ytest_pred == label_test)/n_test; %记录准确率
        ERR(i+1) = norm(X0-opt)/norm(opt);
    end

end