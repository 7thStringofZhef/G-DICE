function new_q_trans=set_trans_mat(q_trans,mat,index,FSC_num)
    new_q_trans=q_trans;
	for i=1:FSC_num
    	for j=1:FSC_num
        	new_q_trans(index,i,j)=mat(i,j);
        end
	end
end