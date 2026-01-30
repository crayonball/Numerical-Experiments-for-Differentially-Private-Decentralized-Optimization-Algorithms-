numIter = 2000;
K = 3;
actIter = 400;
t1 = 1:20:actIter+1;

%------------------------------------------------------------------------------------------------------------------
numIter = actIter*K;
t2 = 1:K:numIter+1;
com_ERR_EXTRA = ERR_EXTRA(t2);
semilogy(t1,com_ERR_EXTRA(t1),'-o','MarkerSize',10,'Color',[195,219,224]/255,'MarkerFaceColor',[195,219,224]/255)
hold on

%------------------------------------------------------------------------------------------------------------------
numIter = actIter*K;
t2 = 1:K:numIter+1;
com_ERR_NIDS = ERR_NIDS(t2);
semilogy(t1,com_ERR_NIDS(t1),'-o','MarkerSize',10,'Color',[137,214,216]/255,'MarkerFaceColor',[137,214,216]/255)
hold on

%------------------------------------------------------------------------------------------------------------------
numIter = actIter*K;
t2 = 1:K:numIter+1;
com_ERR_DIGing = ERR_DIGing(t2);
semilogy(t1,com_ERR_DIGing(t1),'-o','MarkerSize',10,'Color',[156,190,214]/255,'MarkerFaceColor',[156,190,214]/255)
hold on

%------------------------------------------------------------------------------------------------------------------
numIter = actIter*K;
t2 = 1:K:numIter+1;
com_ERR_GT = ERR_GT(t2);
semilogy(t1,com_ERR_GT(t1),'-o','MarkerSize',10,'Color',[103,144,163]/255,'MarkerFaceColor',[103,144,163]/255)
hold on

legend('EXTRA','NIDS','DIGing','GT')

%------------------------------------------------------------------------------------------------------------------
numIter = actIter;
t2 = 1:1:numIter+1;
com_ERR_SONATA = ERR_SONATA(t2);
t1 = 1:10:actIter+1;
semilogy(t1,com_ERR_SONATA(t1),'-o','MarkerSize',10,'Color',[222,228,228]/255,'MarkerFaceColor',[222,228,228]/255)
hold on



%------------------------------------------------------------------------------------------------------------------
p = 1;
numIter_p10 = actIter/p;
t2 = 1:1/p:numIter_p10+1;
com_ERR_MGProx_p10 = ERR_MGProx_p10(t2);
semilogy(t1,com_ERR_MGProx_p10(t1),'-s','MarkerSize',10,'Color',[230,221,209]/255,'MarkerFaceColor',[230,221,209]/255)
hold on



%------------------------------------------------------------------------------------------------------------------
p = 0.5;
numIter_p10 = actIter/p;
t2 = 1:1/p:numIter_p10+1;
com_ERR_MGProx_p05 = ERR_MGProx_p05(t2);
semilogy(t1,com_ERR_MGProx_p05(t1),'-s','MarkerSize',10,'Color',[226,164,116]/255,'MarkerFaceColor',[226,164,116]/255)
hold on
%------------------------------------------------------------------------------------------------------------------
p = 0.2;
numIter_p02 = actIter/p;
t2 = 1:1/p:numIter_p02+1;
com_ERR_MGProx_p02 = ERR_MGProx_p02(t2);
semilogy(t1,com_ERR_MGProx_p02(t1),'-s','MarkerSize',10,'Color',[226,103,53]/255,'MarkerFaceColor',[226,103,53]/255)
hold on



