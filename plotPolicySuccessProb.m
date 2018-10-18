function plotPolicySuccessProb()
    close all
    clc
    
    numPackages = 1:10;
%     cTs = zeros(length(numPackages),2);
    sPs = zeros(length(numPackages),2);
    
    load('randPolicyResults.mat')
    for numPackage = numPackages
        clc
        numPackage
        [~, completionTime, successProb] = evalPolicy(curBestPolicy,numPackage);
        
%         cTs(numPackage,1) = completionTime;
        sPs(numPackage,1) = successProb;
    end
    
    load('randSearchConsistencyResults.mat')
    for numPackage = numPackages
        clc
        numPackage
        [~, completionTime, successProb] = evalPolicy(curBestPolicy,numPackage);
        
%         cTs(numPackage,2) = completionTime;
        sPs(numPackage,2) = successProb;
    end
        
%     figure;
%     bar(cTs);
%     hold on;
%     xlabel('# Packages');
%     ylabel('Completion Time Units');
    
    save('policySuccessProb.mat')

    figure
    mBar = bar(sPs);
    hold on;
    xlabel('# Packages');
    ylabel('Success Probability');
    set(mBar(1),'FaceColor','r')
    set(mBar(2),'FaceColor','b')
    legend('Monte Carlo Search', 'MMCS','location','southoutside')
    set(gcf,'PaperPositionMode','auto')
    print('-depsc2','policySuccessProb.eps')
    
end