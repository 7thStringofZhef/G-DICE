clc
clear all
close all


numNodes = 13; %number of graph controller nodes
alpha = 0.2; %learning rate
numTMAs = 13; %cardinality of TMA space
numObs = 13; %cardinality of observation space
N_k = 50; %number of iterations
N_s = 50; %number of samples per iteration
N_b = 5; %number of "best" samples kept from each iteration

mGraphPolicyController = GraphPolicyController(numNodes, alpha, numTMAs, numObs, N_s);

bestValue = 0;
allValues = zeros(N_k*N_s,1);

fHandle = figure;

for idxIteration = 1:N_k
    %reset N_b best sample values
    curIterationValues = -100*ones(N_s,1);
    %draw N_s samples based on pdfs
    mGraphPolicyController.sample(N_s);
    tic
    %evaluate each sample
    for idxSample = 1:N_s
        fprintf('Iteration %d of %d. Best value so far: %f\n', (idxIteration-1)*N_s + idxSample, N_k*N_s, bestValue)
        mGraphPolicyController.setGraph(idxSample)
        
        [newValue, ~, ~] = evalPolicy(mGraphPolicyController);
        curIterationValues(idxSample) = newValue;
        allValues((idxIteration-1)*N_s + idxSample) = newValue;
        
        if (newValue > bestValue)
           bestValue = newValue;
           [bestTMAs, bestTransitions] = mGraphPolicyController.getPolicyTable();
        end
    end
    
    %performs update and filter of pdfs
    mGraphPolicyController.updateProbs(curIterationValues, N_b);
    
    %plot values so far
    figure(fHandle);
    plot(allValues(1:idxIteration*N_s),'bx');
end

plot(allValues,'bx');
save('crossEntropySearch')
%%
figure
for idxIteration = 1:N_k
    samplesStart = (idxIteration-1)*N_s+1;
    samplesEnd = (idxIteration)*N_s;
    plot(idxIteration, allValues(samplesStart:samplesEnd),'bx');
    hold on
end
xlabel('Iteration')
ylabel('Value')
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
