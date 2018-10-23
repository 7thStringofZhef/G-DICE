function mat=get_q_trans_mat(q_trans_1,FSC_num,index)
mat=zeros(FSC_num,FSC_num);
for i=1:FSC_num
    for j=1:FSC_num
        mat(i,j)=q_trans_1(index,i,j);
    end
end
end