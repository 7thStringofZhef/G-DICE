clc
clear all
close all

continuePrevRun = false;

if (continuePrevRun)
    load('crossEntropySearch15k_3_68');
    N_k = 50; %number of iterations
    N_s = 50; %number of samples per iteration
    N_b = 5; %number of "best" samples kept from each iteration
    
    bestValue = 0;
    allValues = zeros(N_k*N_s,1);
else
    numNodes = 10; %number of graph controller nodes
    alpha = 0.2; %learning rate
    numTMAs = 13; %cardinality of TMA space
    numObs = 13; %cardinality of observation space
    N_k = 300; %number of iterations
    N_s = 30; %number of samples per iteration
    N_b = 3; %number of "best" samples kept from each iteration
    
    mGraphPolicyController = GraphPolicyController(numNodes, alpha, numTMAs, numObs, N_s);
    
    bestValue = 0;
    allValues = zeros(N_k*N_s,1);
end

fHandle = figure;
figure(fHandle);
tic
for idxIteration = 1:N_k
    %reset N_b best sample values
    curIterationValues = -100*ones(N_s,1);
    %draw N_s samples based on pdfs
    mGraphPolicyController.sample(N_s);
    
    %evaluate each sample
        
    parfor idxSample = 1:N_s
        fprintf('Iteration %d of %d. Best value so far: %f\n', (idxIteration-1)*N_s + idxSample, N_k*N_s, bestValue);
        mGraphPolicyController.setGraph(idxSample);
        
        [newValue, ~, ~] = evalPolicy(mGraphPolicyController);
        curIterationValues(idxSample) = newValue;
    end
    
    
    for idxSample = 1:N_s
        allValues((idxIteration-1)*N_s + idxSample) = curIterationValues(idxSample);
        
        if (curIterationValues(idxSample)> bestValue)
            bestValue = curIterationValues(idxSample);
            mGraphPolicyController.setGraph(idxSample)
            [bestTMAs, bestTransitions] = mGraphPolicyController.getPolicyTable();
        end
    end
    
    %performs update and filter of pdfs
    mGraphPolicyController.updateProbs(curIterationValues, N_b);
    
    %plot values so far
%     figure(fHandle)
    plot(allValues(1:idxIteration*N_s),'bx');
end
toc


% %test best policy robustness
% mGraphPolicyController.setPolicyUsingTables(bestTMAs, bestTransitions)
% for i = 1:10
%     evalPolicy(mGraphPolicyController);
% end

delete(fHandle)
filename = ['results\crossEntropySearch_numNodes=' num2str(numNodes) '_Nk=' num2str(N_k) '_Ns=' num2str(N_s) '_alpha=' num2str(alpha) '_bestValue=' num2str(bestValue)];
filename(filename=='.') = 'p';
save(filename);
%%

plotValueWithErrBars(fHandle, allValues, N_s, 'r');

return

%%
plot(allValues,'rx')
hold on
plot(curBestValue)
grid on
axis([0 1000 0 4])
legend('Current Value','Best Value')
xlabel('Iteration')
ylabel('Policy Value')
% set(gcf,'PaperPositionMode','auto')
% print('-depsc','-zbuffer','-r200','policyIterationRandom.eps')
