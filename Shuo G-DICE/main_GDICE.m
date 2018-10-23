clear
clc
close all


observe_num=20;
FSC_num=11;
action_num=5;

% model parameters
theta_trans_1=generate_FSC_trans(FSC_num,observe_num);
theta_trans_2=generate_FSC_trans(FSC_num,observe_num);
theta_emit_1=norm_by_row(zeros(FSC_num,action_num)+1);
theta_emit_2=norm_by_row(zeros(FSC_num,action_num)+1);



N_k=200;    % iteration of optimization
N_s=12;     % sample num
N_b=6;      % kept num

worst_value=-100000;   % worst value for agent 1
best_value =-100000;   % best value for agent 1

best_q1_trans=[];
best_q1_emit=[];
best_q2_trans=[];
best_q2_emit=[];

%initialize joint policies
policy_list=[];
policy_best_list=[];



%optimization
best_velue_plot=zeros(1,N_k);
for opti_iter=1:N_k
    disp(['iter ',num2str(opti_iter),' of ',num2str(N_k)]);
    policy_list=[];
    policy_best_list=[];
    for j=1:N_s   %generate numbers of policies
        % sample new policy from theta
        [temp_q1_trans,temp_q2_trans,temp_q1_emit,temp_q2_emit]=generate_new_policy_from_theta(theta_trans_1,theta_trans_2,theta_emit_1,theta_emit_2,FSC_num,observe_num,action_num);
        
        %evaluate policy
        gain=evaluate_policy(temp_q1_trans,temp_q2_trans,temp_q1_emit,temp_q2_emit,FSC_num);
        if(gain>worst_value)
            policy_list=add_policy_to_list(policy_list,temp_q1_trans,temp_q1_emit,temp_q2_trans,temp_q2_emit,gain);
        end

        if(gain>best_value)
            best_value=gain;
            best_q1_trans=temp_q1_trans;
            best_q1_emit=temp_q1_emit;
            best_q2_trans=temp_q2_trans;
            best_q2_emit=temp_q2_emit;
        end
        disp(['gain is ',num2str(gain)]);
    end
    best_velue_plot(1,opti_iter)=best_value;
    disp(['best value is ',num2str(best_value)]);
    % choose N_b best policies in policy_list
    policy_best_list=choose_best_policies(policy_list,N_b);
    
    % update worst_value
    [m,n]=size(policy_best_list);
    if(m~=0)
        worst_value=policy_best_list{m,5};
        % update parameters
        [theta_trans_1,theta_trans_2,theta_emit_1,theta_emit_2]=update_theta(policy_best_list,theta_trans_1,theta_trans_2,theta_emit_1,theta_emit_2,0.3,6);
    end     
end
   
figure
k=1:opti_iter;
plot(k,best_velue_plot);


%test_policy(best_q1_trans,best_q1_emit,best_q2_trans,best_q2_emit,FSC_num);




