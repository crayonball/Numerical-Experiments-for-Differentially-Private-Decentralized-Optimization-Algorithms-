function [test_acc,ERR] = RandCom(obj_fn,grad_fn,numProcesses,d,numIter,W,tau,chi,p,data_ix,opt,R,noise,X_test,n_test,label_test)

    
    W_b = chi*(eye(numProcesses) - W);

    X0 = rand(numProcesses,d);
    Y0 = rand(numProcesses,d); 
    grad  = zeros(numProcesses,d); 
    
    ERR = zeros(1,numIter/p+1);
    
    ERR(1) = norm(X0-opt)/norm(opt);

    test_acc = zeros(1,numIter);

    x_mean = mean(X0);
    Ytest_pred = linear_prediction(X_test,x_mean'); %预测这次迭代结果x产生得标签值
    test_acc(1) = sum(Ytest_pred == label_test)/n_test; %记录准确率

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

        ERR(i+1) = norm(X0-opt)/norm(opt);

        x_mean = mean(X0);
        Ytest_pred = linear_prediction(X_test,x_mean'); %预测这次迭代结果x产生得标签值
        test_acc(i+1) = sum(Ytest_pred == label_test)/n_test; %记录准确率

    end
    
   
end