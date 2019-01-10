close all
clc

subplot(2,1,1)
cb3 = newcolorbar('north');

a = makeColorMap([0 0 0],colors(1,:));
colormap(gca,a)
freezeColors

% mycb1 = colorbar;

% cbfreeze(mycb1)

pause

subplot(2,1,2)
cb3 = newcolorbar('south');

a = makeColorMap([0 0 0],colors(2,:));
colormap(gca,a)
freezeColors

% mycb2 = colorbar('location','northoutside')
% cbfreeze(mycb2)
