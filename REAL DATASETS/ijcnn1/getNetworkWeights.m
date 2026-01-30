function [W, L, E] = getNetworkWeights(settingName, numProcesses)

Id = eye(numProcesses); % eye是numProcess*numProcess的单位阵。numProcess是节点数
E = Id;
if(strcmp(settingName,'Circular'))
    E = Id + Id(:,[2:end 1]) + Id(:,[end 1:end-1]); %2:end 表示从第2列到最后一列，最后再加上1.比如[2:5 1] = [2,3,4,5,1],即把Id的第一列掉到最后
    % Id(:, [end 1:end-1])表示把最后一行提到第一行去    
    % 最终得到的E是环形图的相关矩阵，对角线为1，然后与这个点相邻的两列为1，即其邻居。
elseif(strcmp(settingName,'Connected'))
    E = ones(numProcesses);
    E = E+0*Id;
elseif(strcmp(settingName,'Barbell'))
    if(mod(numProcesses,2)~=0)
        error('numProcesses must be a multiple of 2');
    end
    E = zeros(numProcesses);
    E(1:numProcesses/2,1:numProcesses/2) = ones(numProcesses/2);
    E(numProcesses/2+1:end,numProcesses/2+1:end) = ones(numProcesses/2);
    E(numProcesses/2+1,numProcesses/2) =1;
    E(numProcesses/2,numProcesses/2+1) =1;
    
elseif(strcmp(settingName,'full2'))
    E = ones(numProcesses);
    E = E+5*Id;
elseif(strcmp(settingName,'Disconnected'))
    E = Id;
else
    error('Unknown architecture');
end
% Weight matrix
W = sinkhornKnopp(E);%正规化矩阵非负定矩阵E，使得每行和每列的和为1。（源代码中是说nonnegative matrix,不知道是不是非负定的意思。拉普拉斯矩阵一定非负定，因为
%它是对角占优矩阵）
L = diag(E*ones(numProcesses,1))-E; %L是laplace矩阵。有符号的



