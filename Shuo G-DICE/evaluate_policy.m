function gain=evaluate_policy(q_trans_1,q_trans_2,q_emit_1,q_emit_2,FSC_num)
    agent1_acc_reward=0;
    agent2_acc_reward=0;
    
    agent1_q=1;
    agent2_q=1;
    start_1=[3;4];
    start_2=[3;7];
    des_1=[2;9];
    des_2=[2;2];
    agent1_pos=start_1;
    agent2_pos=start_2;
    occupancy=[ 1 1 1 1 1 1 1 1 1 1;
            1 0 0 0 0 0 0 0 0 1;
            1 1 1 1 1 1 1 1 1 1;
            1 1 1 1 1 1 1 1 1 1];
        
    max_MC_iter=30000;
    for k=1:max_MC_iter     % Monte Carlo Simulation
    % observe q

        obs=get_observation(agent1_pos,occupancy,agent2_pos);
        index=get_obs_index(obs);
        mat=get_q_trans_mat(q_trans_1,FSC_num,index);
        agent1_q = discretesample(mat(agent1_q,:), 1);

        obs=get_observation(agent2_pos,occupancy,agent1_pos);
        index=get_obs_index(obs);
        mat=get_q_trans_mat(q_trans_2,FSC_num,index);
        agent2_q = discretesample(mat(agent2_q,:), 1);

        % sample action
        action1=discretesample(q_emit_1(agent1_q,:), 1);
        action2=discretesample(q_emit_2(agent2_q,:), 1);

        % move under action  
        [agent1_pos,occupancy,reward1]=agent_move(agent1_pos,action1,occupancy,start_1,des_1);
        [agent2_pos,occupancy,reward2]=agent_move(agent2_pos,action2,occupancy,start_2,des_2);

        agent1_acc_reward=agent1_acc_reward+reward1;
        agent2_acc_reward=agent2_acc_reward+reward2;
    end
    gain=agent1_acc_reward+agent2_acc_reward;
end