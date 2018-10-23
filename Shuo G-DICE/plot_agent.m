function plot_agent(agent1_p,agent2_p)

rectangle('position',[ 10*(agent1_p(2,1)-1) -10*agent1_p(1,1) 10 10], 'FaceColor','r' );

rectangle('position',[ 10*(agent2_p(2,1)-1) -10*agent2_p(1,1)  10 10], 'FaceColor','b' );
end