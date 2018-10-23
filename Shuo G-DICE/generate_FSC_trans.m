function q_trans=generate_FSC_trans(FSC_num,observe_num)
q_trans=zeros(observe_num,FSC_num,FSC_num);
for i=1:observe_num
    q_trans(i,:,:)=norm_by_row(rand(FSC_num,FSC_num));
end
end