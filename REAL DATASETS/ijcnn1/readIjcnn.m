function [features,labels] = readIjcnn(FileName)

% 加载数据
data = load(FileName);
labels = data(:,1);
features = data(:,2:23);
end