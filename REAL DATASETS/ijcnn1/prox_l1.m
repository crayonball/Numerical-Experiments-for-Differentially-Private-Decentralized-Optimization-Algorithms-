function val = prox_l1(x,t)
% prox_tg(.),这里的t就是那个下标t.sign()信号函数，大于0为1，小于0为-1，等于0为0
    %val = sign(x).*max(abs(x)-t,0);
    val = sign(x).*max(abs(x)-t,0);
end

