function vec=get_observation(agent_pos,occupancy,other_pos)
vec=zeros(1,8);
if(occupancy(agent_pos(1,1)-1,agent_pos(2,1)-1)==1)
    vec(1,1)=1;
end
if(occupancy(agent_pos(1,1)-1,agent_pos(2,1))==1)
    vec(1,2)=1;
end
if(occupancy(agent_pos(1,1)-1,agent_pos(2,1)+1)==1)
    vec(1,3)=1;
end
if(occupancy(agent_pos(1,1),agent_pos(2,1)-1)==1)
    vec(1,4)=1;
end
if(occupancy(agent_pos(1,1),agent_pos(2,1)+1)==1)
    vec(1,5)=1;
end
if(occupancy(agent_pos(1,1)+1,agent_pos(2,1)-1)==1)
    vec(1,6)=1;
end
if(occupancy(agent_pos(1,1)+1,agent_pos(2,1))==1)
    vec(1,7)=1;
end
if(occupancy(agent_pos(1,1)+1,agent_pos(2,1)+1)==1)
    vec(1,8)=1;
end

%detect the position of other
if(isequal(agent_pos,other_pos+[-1;0]))
    vec(1,7)=2;
end
if(isequal(agent_pos,other_pos+[1;0]))
    vec(1,2)=2;
end
if(isequal(agent_pos,other_pos+[0;-1]))
    vec(1,5)=2;
end
if(isequal(agent_pos,other_pos+[0;1]))
    vec(1,4)=2;
end


end