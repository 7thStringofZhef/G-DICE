function test_policy(q1_trans,q1_emit,q2_trans,q2_emit,FSC_num)
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
for k=1:10000     % Monte Carlo Simulation
    disp(['iter ',num2str(k),' of',num2str(10000)]);
    
    % observe q
    obs=get_observation(agent1_pos,occupancy,agent2_pos);
    index=get_obs_index(obs);
    mat=get_q_trans_mat(q1_trans,FSC_num,index);
    agent1_q = discretesample(mat(agent1_q,:), 1);
    
    obs=get_observation(agent2_pos,occupancy,agent1_pos);
    index=get_obs_index(obs);
    mat=get_q_trans_mat(q2_trans,FSC_num,index);
    agent2_q = discretesample(mat(agent2_q,:), 1);

    % sample action
    action1=discretesample(q1_emit(agent1_q,:), 1);
    action2=discretesample(q2_emit(agent2_q,:), 1);
    
    % move under action  
    [agent1_pos,occupancy,reward1]=agent_move(agent1_pos,action1,occupancy,start_1,des_1);
    [agent2_pos,occupancy,reward2]=agent_move(agent2_pos,action2,occupancy,start_2,des_2);
    
    agent1_acc_reward=agent1_acc_reward+reward1;
    agent2_acc_reward=agent2_acc_reward+reward2;
    agent1_acc_reward_plot(1,k)=agent1_acc_reward;
    agent2_acc_reward_plot(1,k)=agent2_acc_reward;
    
    
    % plot new position 
    clf
    plot_map(occupancy);
    plot_agent(agent1_pos,agent2_pos);
    pause(0.2);
end
k=1:max_MC_iter;
figure
plot(k,agent1_acc_reward_plot,'r');
hold on
plot(k,agent2_acc_reward_plot,'b');
legend('agent1','agent2');

end