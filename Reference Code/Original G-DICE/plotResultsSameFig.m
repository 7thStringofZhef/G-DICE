close all
clear all
clc

all = false;
fig = figure;


    load('randPolicyResults.mat')
    hold on
    p1 = plot(curBestValue, 'r','linewidth',2);
    grid on
    xlabel('Iteration')
    ylabel('Policy Value')
    
    clear all
    load('randSearchConsistencyResults.mat')
    hold on
    p2 = plot(curBestValue, 'b','linewidth',2);
    axis([0 1000 0 5])

    span = 100;
%     close all
    load('randPolicyResults.mat')
    meanAllValues = mean(allValues);
    allValuesMovingAvg = smooth(allValues,span);
    allValues(allValues ==0) = -1;
    p3 = plot(allValues,'color',[0.8 0.2 0.2],'LineStyle','none','Marker','o');
    hold on
    p4 = plot(allValuesMovingAvg,'m','linewidth',2);
    grid on
    xlabel('Iteration')
    ylabel('Policy Value')
    
%     clear all
    span = 100;
    load('randSearchConsistencyResults.mat')
    allValuesMovingAvg = smooth(allValues,span);
    allValues(allValues ==0) = -1;
    p5 = plot(allValues,'color',[0.4 0.4 0.9],'LineStyle','none','Marker','x');
    hold on
    p6 = plot(allValuesMovingAvg,'k','linewidth',2);
    axis([0 1000 0 5])
    figure(fig)
    gridLegend([p1 p3 p4 p2  p5 p6],2,{'MC Search Best Policy', 'MC Search Samples', 'MC Search Moving Average (n = 100)', 'MMCS Best Policy', 'MMCS Samples', 'MMCS Moving Average (n = 100)'},'location','southoutside')
% gridLegend([p1 p2 p3 p4 p5 p6],2,{'1','2','3','4','5','6'},'location','southoutside')
    set(fig, 'Position', [100, 100, 700, 600]);
    
    length(allValues(allValues>meanAllValues))/length(allValues)

    set(gcf,'PaperPositionMode','auto')
    print('-depsc2','comparisonAllValues.eps')
