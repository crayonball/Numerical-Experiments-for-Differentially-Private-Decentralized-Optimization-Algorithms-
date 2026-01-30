function [y] = linear_prediction(X,w)
%     Evaluate the prediction of the linear classification
%       Output:
%           y   classification vector of 1 or -1 

    y = ((X*w) >0);%由于数据经过正则化，故若X*w大于0，y=1,否则y=-1
    y = (y-0.5)*2;
end