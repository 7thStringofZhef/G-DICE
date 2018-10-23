function new_list=add_policy_to_list(list,q1_trans,q1_emit,q2_trans,q2_emit,reward)
    [m,n]=size(list);
    new_list=list;
    
    new_list{m+1,1}=q1_trans;
    new_list{m+1,2}=q1_emit;
    new_list{m+1,3}=q2_trans;
    new_list{m+1,4}=q2_emit;
    new_list{m+1,5}=reward;    
end