function subplot_positions = plot_discrete_obs_density(fHandle, single_agent_obs_density, numNodes, n_bins_xy, dom)
    figure(fHandle)
    clf;
    
    subplot_positions = zeros(numNodes,4);
    
    for idxNode = 1:numNodes
        subplot(1,numNodes,idxNode)
        subplot_positions(idxNode,:) = get(gca,'position');
        % this is specific to the plume domain (since it's 2D and the flipud+reshape takes care of the
        % XeIdx to XY coord mapping)
%         imagesc(flipud(reshape(single_agent_obs_density(idxNode ,:),[n_bins_xy,n_bins_xy])))
        imagesc(reshape(single_agent_obs_density(idxNode ,:),[n_bins_xy,n_bins_xy]))
        set(gca,'YDir','normal')
        axis square
        colormap summer
        ax1 = gca;
        
        
%         hold on
        set(gca,'xticklabel',[])
        set(gca,'yticklabel',[])

        % plot the plume domain
        axes('Position',ax1.Position);
        should_pop_colors = true;
        dom.plotBeliefNodes([0.8 0.8 0.8], should_pop_colors,1);
        view(0,90)
        set(gca,'color','none')
        set(gca,'xtick',0:5)
        set(gca,'ytick',0:5)
        title(['$q=' num2str(idxNode) '$'],'interpreter','latex')
        xlabel('$o^e_1$','interpreter','latex')
        ylabel('$o^e_2$','interpreter','latex')
        
        xlabh = get(gca,'XLabel');
        set(xlabh,'Position',get(xlabh,'Position') - [0 -0.2 0])
        ylabh = get(gca,'YLabel');
        set(ylabh,'Position',get(ylabh,'Position') - [-0.2 0 0])
    end
end