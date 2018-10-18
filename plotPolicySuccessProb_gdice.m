clear all
close all
clc

load('policySuccessProb.mat')

numPackages = 1:10;
%     cTs = zeros(length(numPackages),2);

load('results\crossEntropySearch_numNodes=13_Nk=300_Ns=30_alpha=0p1_bestValue=14p4427.mat')
for numPackage = numPackages
    numPackage
    [~, completionTime, successProb] = evalPolicy(mGraphPolicyController,numPackage);    
    sPs(numPackage,3) = successProb
end


figure
mBar = bar(sPs);
hold on;
xlabel('# Packages');
ylabel('Success Probability');
set(mBar(1),'FaceColor','r')
set(mBar(2),'FaceColor','b')
legend('Monte Carlo Search', 'MMCS', 'G-DICE','location','eastoutside')
set(gcf,'PaperPositionMode','auto')
grid on
print('-depsc2','policySuccessProb.eps')

