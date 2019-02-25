//#include <Python.h>
#include "library.h"

#include <iostream>
//#include "boost/random.hpp"
#include "boost/array.hpp"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/discrete_distribution.hpp>

boost::mt19937 gen;

#include <array>
#include <cstddef>

void hello() {
    std::cout << "Hello, World!" << std::endl;
}


//Convert T to vector of distribution
template <size_t nStates>
template <size_t nActions>
std::array< std::array< boost::random::discrete_distribution<>, nActions>, nStates> tToDist(std::array< std::array< std::array<float, nStates>, nActions>, nStates>& T){
    std::array< std::array< boost::random::discrete_distribution<>, nActions>, nStates> initArray;
    for (int i=0; i < nStates; i++) {
        for (int j=0; j < nActions; j++){
            initArray[i][j] = boost::random::discrete_distribution<>(T[i][j]);
        }
    }
    return initArray;
}

//Convert O to vector of distribution
template <std::size_t nStates>
template <std::size_t nActions>
template <std::size_t nObs>
std::array< std::array< std::array< boost::random::discrete_distribution<>, nStates>, nActions>, nStates> oToDist(std::array< std::array< std::array< std::array<float, nObs>, nStates>, nActions>, nStates>& O){
    initArray = std::array< std::array< std::array< boost::random::discrete_distribution<>, nStates>, nActions>, nStates>;
    for (i=0; i < nStates; i++) {
        for (j=0; j < nActions; j++){
            for(k=0; k < nStates; k++) {
                initArray[i][j][k] = boost::random::discrete_distribution<>(T[i][j][k]);
            }
        }
    }
    return initArray;
}

//First step is to define my functions

//For each timestep

//Choose actions (currentNode, aTrans)
//aTrans is numNodes*numSamples
//Count indices from outer to inner
template <std::size_t nSims>
template <std::size_t nNodes>
template <std::size_t nSamples>
void chooseActionsAll(std::array<int, nSims*nSamples>& nodeVec, std::vector< std::vector<int, nNodes>, nSamples>& aTrans, const int numSamples, const int numSim, std::array<int, nSims*nSamples>& actionVec){
    for (i=0; i < numSamples*numSim; i++) {
        j = i / numSim;  //Floor division
        actionVec[i] = aTrans[j][nodeVec[i]];
    }
}

//Choose actions (currentNode, aTrans)
//aTrans is numNodes
template <std::size_t nSims>
template <std::size_t nNodes>
void chooseActions(std::array<int, nSims>& nodeVec, std::vector<int, nNodes>& aTrans, const int numSim, std::array<int, nSims>& actionVec){
    for (i=0; i < numSim; i++) {
        actionVec[i] = aTrans[nodeVec[i]];
    }
}

//Sample from TDist
//Where T is numStates*numActions set of discrete dist
template <std::size_t nSims>
template <std::size_t nStates>
template <std::size_t nActions>
void getNewState(std::array<int, nSims>& actionVec, std::array<int, nSims>& stateVec, std::array< std::array< boost::random::discrete_distribution<>, nActions>, nStates>& Tdist, std::array<int, nSims>& newStateVec){
    for (i=0; i < nSims; i++){
        newStateVec[i] = Tdist[stateVec[i], actionVec[i]](gen);
    }
}

void getNewStateAll(){

}


//Sample from O
//Where O is numStates*numActions*numStates set of discrete dist
template <std::size_t nSims>
template <std::size_t nStates>
template <std::size_t nActions>
void getObs(std::array<int, nSims>& actionVec, std::array<int, nSims>& stateVec, std::array<int, nSims>& newStateVec, std::array< std::array< std::array< boost::random::discrete_distribution<>, nStates>, nActions>, nStates>& Odist, std::array<int, nSims>& obsVec){
    for (i=0; i < nSims; i++){
        obsVec[i] = Odist[stateVec[i], actionVec[i], newStateVec[i]](gen);
    }
}

void getObsAll(){

}

//Index into R
//Where R is numStates*numActions*numStates*numObs
template <std::size_t nSims>
template <std::size_t nStates>
template <std::size_t nActions>
template <std::size_t nObs>
void getReward(std::array<int, nSims>& actionVec, std::array<int, nSims>& stateVec, std::array<int, nSims>& newStateVec, std::array<int, nSims>& obsVec, std::array< std::array< std::array< std::array<double, nObs>, nStates>, nActions>, nStates>& R, std::array<int, nSims>& valueVec){
    for (i=0; i < nSims; i++){
        valueVec[i] = R[stateVec[i], actionVec[i], newStateVec[i], obsVec[i]];
    }
}

//Index into R
//Where R is numStates*numActions*numStates*numObs
template <std::size_t nSims>
template <std::size_t nStates>
template <std::size_t nActions>
template <std::size_t nObs>
void getRewardAll(std::array<int, nSims>& actionVec, std::array<int, nSims>& stateVec, std::array<int, nSims>& newStateVec, std::array<int, nSims>& obsVec, std::array< std::array< std::array< std::array<double, nObs>, nStates>, nActions>, nStates>& R, std::array<int, nSims>& valueVec){
    for (i=0; i < nSims; i++){
        valueVec[i] = R[stateVec[i], actionVec[i], newStateVec[i], obsVec[i]];
    }
}
