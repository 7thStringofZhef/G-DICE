close all
clear all
clc

all = true;

if (all)
    
    load('randPolicyResults.mat')
    hold on
    plot(curBestValue, 'r')
    grid on
    legend('Current Value','Best Value')
    xlabel('Iteration')
    ylabel('Policy Value')
    
    clear all
    load('randSearchConsistencyResults.mat')
    hold on
    plot(curBestValue, 'b')
    axis([0 1000 0 5])
    legend('location','SouthOutside','Monte Carlo Search','MMCS')
    set(gcf,'PaperPositionMode','auto')
    print('-depsc2','comparisonBestValues.eps')
else
    span = 100;
    close all
    fig = figure;
    load('randPolicyResults.mat')
    meanAllValues = mean(allValues);
    allValuesMovingAvg = smooth(allValues,span);
    allValues(allValues ==0) = -1;
    plot(allValues,'color',[0.8 0.2 0.2],'LineStyle','none','Marker','o')
    hold on
    plot(allValuesMovingAvg,'r','linewidth',2)
    grid on
    legend('Current Value','Best Value')
    xlabel('Iteration')
    ylabel('Policy Value')
    
    clear all
    load('randSearchConsistencyResults.mat')
    allValuesMovingAvg = smooth(allValues,span);
    allValues(allValues ==0) = -1;
    plot(allValues,'color',[0.4 0.4 0.9],'LineStyle','none','Marker','x')
    hold on
    plot(allValuesMovingAvg,'b','linewidth',2)
    axis([0 1000 0 5])
    legend('location','SouthOutside','Monte Carlo Search Samples', 'Monte Carlo Search Moving Average (n = 100)', 'MMCS Samples', 'MMCS Moving Average (n = 100)')
    set(fig, 'Position', [100, 100, 600, 600]);
    
    length(allValues(allValues>meanAllValues))/length(allValues)

    set(gcf,'PaperPositionMode','auto')
    print('-depsc2','comparisonAllValues.eps')
end