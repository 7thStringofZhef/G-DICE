function plot_map(occupancy)
hold off
[m,n]=size(occupancy);
for i=1:m
    for j=1:n
        if(occupancy(i,j)==1)
            rectangle('position',[10*(j-1) -10*i 10 10], 'FaceColor','k' );
        end
    end
end
axis([0 120 -120 0]);
end