function [new_agent_p,new_occupancy,reward]=agent_move(agent_p,action,occupancy,start,des)
    % action= 1.up 2.down 3.left 4.right 5.wait
    reward=-1;
    new_agent_p=agent_p;
    new_occupancy=occupancy;
    if(action==1)
        if(occupancy(agent_p(1,1)-1,agent_p(2,1))~=1) % if can move
            new_agent_p(1,1)=agent_p(1,1)-1;
            new_occupancy(agent_p(1,1),agent_p(2,1))=0;
            new_occupancy(agent_p(1,1)-1,agent_p(2,1))=1;
        end
        if(new_agent_p==des)  % reach the destination
            new_occupancy(new_agent_p(1,1),new_agent_p(2,1))=0;
            new_agent_p=start;
            new_occupancy(new_agent_p(1,1),new_agent_p(2,1))=1;
            reward=100;  % get reward
        end
    elseif(action==2)
        if(occupancy(agent_p(1,1)+1,agent_p(2,1))~=1)
            new_agent_p(1,1)=agent_p(1,1)+1;
            new_occupancy(agent_p(1,1),agent_p(2,1))=0;
            new_occupancy(agent_p(1,1)+1,agent_p(2,1))=1;
        end
        if(new_agent_p==des)  % reach the destination
            new_occupancy(new_agent_p(1,1),new_agent_p(2,1))=0;
            new_agent_p=start;
            new_occupancy(new_agent_p(1,1),new_agent_p(2,1))=1;
            reward=100;  % get reward
        end
    elseif(action==3)
        if(occupancy(agent_p(1,1),agent_p(2,1)-1)~=1)
            new_agent_p(2,1)=agent_p(2,1)-1;
            new_occupancy(agent_p(1,1),agent_p(2,1))=0;
            new_occupancy(agent_p(1,1),agent_p(2,1)-1)=1;
        end
        if(new_agent_p==des)  % reach the destination
            new_occupancy(new_agent_p(1,1),new_agent_p(2,1))=0;
            new_agent_p=start;
            new_occupancy(new_agent_p(1,1),new_agent_p(2,1))=1;
            reward=100;  % get reward
        end
    elseif(action==4)
        if(occupancy(agent_p(1,1),agent_p(2,1)+1)~=1)
            new_agent_p(2,1)=agent_p(2,1)+1;
            new_occupancy(agent_p(1,1),agent_p(2,1))=0;
            new_occupancy(agent_p(1,1),agent_p(2,1)+1)=1;
        end
        if(new_agent_p==des)  % reach the destination
            new_occupancy(new_agent_p(1,1),new_agent_p(2,1))=0;
            new_agent_p=start;
            new_occupancy(new_agent_p(1,1),new_agent_p(2,1))=1;
            reward=100;  % get reward
        end
    elseif(action==5)
        reward=0;
    end
end