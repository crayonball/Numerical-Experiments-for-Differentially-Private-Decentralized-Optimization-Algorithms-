function [PLF,time] = Walkman(Q,q,n,d,W,rho,X0,Y0,alpha,opt,stop)
%节点个数

%权重矩阵
A = diag(ones(n-1,1),1)+diag(ones(n-1,1),1)';
L = diag(sum(A))-A;

%生成数据
for i =1:n
    V = diag([1 2 3]);
    U = orth(rand(q));
    Q(:,:,i) = U*V*U';
    P(:,i) = rand(q,1);
    
    H((i-1)*q+1:q*i,(i-1)*q+1:q*i) = Q(:,:,i);
    f((i-1)*q+1:i*q,1) = P(:,i);
end
T = 80000;
beta = 0.01;
alpha = 0.1;

lambda0 = zeros(q,n);
x0 = zeros(q,1);
y0 = zeros(q,n);

for i = 1 : T
    lambda00 =lambda0 + beta * (kron(ones(1,n),x0)-y0);
    x1 = x0-alpha*sum(lambda00,2);

    for j = 1:n
        grad(:,j) = Q(:,:,j)*y0(:,j) + P(:,j);
    end
    y1 = y0 - alpha*(grad -lambda00);

    lambda0 = lambda00 + beta*(kron(ones(1,n),x1-x0) - (y1-y0));

    x0 = x1;
    y0 = y1;
end

opt = x0;

lambda0 = zeros(q,n);
x0 = zeros(q,1);
y0 = zeros(q,n);

iter = 1;
time = cputime;
ERR = norm(x0-opt)/norm(opt);

while ERR >= stop
    lambda00 =lambda0 + beta * (kron(ones(1,n),x0)-y0);
    x1 = x0-alpha*sum(lambda00,2);

    for j = 1:n
        grad(:,j) = Q(:,:,j)*y0(:,j) + P(:,j);
    end
    y1 = y0 - alpha*(grad -lambda00);

    lambda0 = lambda00 + beta*(kron(ones(1,n),x1-x0) - (y1-y0));

    x0 = x1;
    y0 = y1;
    ERR = norm(x0-opt)/norm(opt);
    iter = iter + 1;

end

 PLF = iter;
 time = cputime - time;


end