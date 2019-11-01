function y=prune(x)
y=x;
t=x(:,1);
t_diff=t(2:length(t))-t(1:length(t)-1);

while(any(t_diff<=0))
    thresh=t_diff<=0;
    thresh=[false;thresh];
    t(thresh)=[];
    y(thresh,:)=[];
    t_diff=t(2:length(t))-t(1:length(t)-1);
end
y(:,1)=y(:,1)-y(1,1);
end

