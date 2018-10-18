close all
clear all
clc

isOutputOn = false;
numPackages = 9999;
isSimOn = true;

load('results\crossEntropySearch_numNodes=7_Nk=300_Ns=30_alpha=0p2_bestValue=11p8535');
mGraphPolicyController.setPolicyUsingTables(bestTMAs,bestTransitions)

dom = Domain();
dom.setPolicyForAllAgents(mGraphPolicyController);
dom.evalCurPolicy(isOutputOn, numPackages, isSimOn)