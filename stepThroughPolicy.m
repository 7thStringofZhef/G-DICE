clear all
close all
clc

load('results\crossEntropySearch_numNodes=3_Nk=300_Ns=30_alpha=0p2_bestValue=1p548')
isOutputOn = true;
[newValue, ~, ~] = evalPolicy(mGraphPolicyController, [], isOutputOn)