function [test_acc,ERR] = KSkip(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,chi,p,data_ix,K,opt,R,noise,X_test,n_test,label_test)

    W_b = chi*(eye(numProcesses) - W^K);

    X0 = rand(numProcesses,d);
    Y0 = rand(numProcesses,d); 
    grad  = zeros(numProcesses,d); 
    
    ERR = zeros(1,numIter/p+1);

    
    test_acc = zeros(1,numIter);

    x_mean = mean(X0);
    Ytest_pred = linear_prediction(X_test,x_mean'); %预测这次迭代结果x产生得标签值
    test_acc(1) = sum(Ytest_pred == label_test)/n_test; %记录准确率
    
    ERR(1) = norm(X0-opt)/norm(opt);

    for i = 1:numIter/p
        epsilon = ((1/R)^i)*randn(1);

        % 分布式计算梯度
        
        X = X0;
        for p_ix = 1 : numProcesses
            y_local = X(p_ix,:);
            data_local = data_ix{p_ix};
            grad(p_ix,:) = grad_fn(y_local',data_local)';
        end

        XT = X0 - tau * (grad + noise*randn(numProcesses,d)) - tau * Y0;

        if rand(1) <= p
            Y1 = Y0 + W_b * (XT + epsilon);
            X1 = XT - tau/p*(Y1-Y0);
        else
            Y1 = Y0;
            X1 = XT;
        end
        

        Y0 = Y1;
        X0 = X1;

        x_mean = mean(X0);
        Ytest_pred = linear_prediction(X_test,x_mean'); %预测这次迭代结果x产生得标签值
        test_acc(i+1) = sum(Ytest_pred == label_test)/n_test; %记录准确率

        ERR(i+1) = norm(X0-opt)/norm(opt);

    end
    
   
end