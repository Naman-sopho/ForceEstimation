
t=joint_pos_measured(:,1)-joint_pos_measured(1,1);
Pm=prune(joint_pos_measured);
Vm=prune(joint_vel_measured);
Tm=prune(joint_tor_measured);
Td=prune(joint_tor_desired);

FS=prune(force_sensor);

J0=jacobian_spatial;
J6=jacobian_body;


if length(t)> length(J0);
    while 1
        t(end)=[];
        if length(t)==length(J0)
            break;
        end
    end;
elseif length(t)<length(J0);
    while 1
        J0(end,:)=[];
        J6(end,:)=[];
        if length(t)==length(J0)
            break;
        end
    end
end
J0=[t J0(1:length(t),:)];
J0=prune(J0);

J6=[t J6(1:length(t),:)];
J6=prune(J6);


x=cartesian_position(:,1);
y=cartesian_position(:,2);
z=cartesian_position(:,3);

x=[t x];
x=prune(x);
y=[t y];
y=prune(y);
z=[t z];
z=prune(z);

