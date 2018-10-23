function [q1_trans,q2_trans,q1_emit,q2_emit]=generate_new_policy_from_theta(theta_trans_1,theta_trans_2,theta_emit_1,theta_emit_2,FSC_num,observe_num,action_num)
    q1_trans=theta_trans_1;
    q2_trans=theta_trans_2;
    q1_emit=theta_emit_1;
    q2_emit=theta_emit_2;
    for i=1:FSC_num
        for j=1:action_num
            q1_emit(i,j)=exprnd(theta_emit_1(i,j));
            q2_emit(i,j)=exprnd(theta_emit_2(i,j));
        end
    end
    q1_emit=norm_by_row(q1_emit);
    q2_emit=norm_by_row(q2_emit);
    
    for k=1:observe_num
        temp1_mat=zeros(FSC_num,FSC_num);
        temp2_mat=zeros(FSC_num,FSC_num);
        for i=1:FSC_num
            for j=1:FSC_num
                temp1_mat(i,j)=exprnd(theta_trans_1(k,i,j));
                temp2_mat(i,j)=exprnd(theta_trans_2(k,i,j));
            end
        end
        temp1_mat=norm_by_row(temp1_mat);
        temp2_mat=norm_by_row(temp2_mat);
        q1_trans=set_trans_mat(q1_trans,temp1_mat,k,FSC_num);
        q2_trans=set_trans_mat(q1_trans,temp2_mat,k,FSC_num);
    end
end