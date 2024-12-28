% m is the number of measurements
% Ntracks is the number of confirmed tracks
% Pd is the probability of detection
% Pg is the gate probability
% betaFA is the FA rate


% TrackArray is the array of track structures in my own code. You should
% replace the quantities related to this with yours

% MeasArray is the array of measurements it is a 2xm matrix
% I will frequently use measurement index 0 for no measurement association

% GateDecisions is the validation matrix. It has Ntracks rows and m
% columns. It is binary. Note that this is the transpose of the validation 
% matrix in slide 7 of Lecture on GNN.

function JPDAprobs = JPDA_ProbCalc(m,Ntracks,Pd,Pg,betaFA,TrackArray,MeasurementSet,ValidationMatrix)
GateDecisions = ValidationMatrix';
MeasArray = zeros(2,length(MeasurementSet));
for i = 1:length(MeasurementSet)
    MeasArray(:,i) = MeasurementSet{i};
end

TracksPossibleAssociations=cell(1,Ntracks);%Cell array holding the possible associations for the tracks
TracksNofPossibleAssociations=zeros(1,Ntracks);%Array holding the number of possible associations to the tracks
for i=1:Ntracks
    MeasurementIndices=find(GateDecisions(i,:)>0);%indices of measurements in the gate of the track
    TracksPossibleAssociations{i}=[0 MeasurementIndices];%all possible associations to the track. 0 represents no measurement association
    TracksNofPossibleAssociations(i)=length(MeasurementIndices)+1;%number of all possible associations
end
Nhypotheses=prod(TracksNofPossibleAssociations);%Number of all possible hypotheses (both valid and invalid hypotheses)
JPDAhypothesisMatrix=zeros(Nhypotheses,Ntracks);%Hypotheses matrix. Each row represents a hypothesis. Each column represents a track. 
%Elements keep the index of the measurement associated to the track for the
%hypothesis with zero representing no measurement association for the corresponding track (column). 
HypothesisProbabilities=ones(Nhypotheses,1);% Hypotheses probabilities

IndexMatrix=zeros(Nhypotheses,Ntracks);%This is an index matrix that will be useful in generating all possible hypotheses
Numbers=0:(Nhypotheses-1);
ModeNumber=Nhypotheses;
for i=1:Ntracks
    ModeNumber=ModeNumber/TracksNofPossibleAssociations(i);
    IndexMatrix(:,i)=floor(Numbers/ModeNumber)+1;
    Numbers=rem(Numbers,ModeNumber);
end
%Check the IndexMatrix to see what it looks like. 

for i=1:Ntracks %For all tracks
    TrackPossibleAssociations=TracksPossibleAssociations{i};%Get the possible associations to the track
    TrackNofPossibleAssociations=TracksNofPossibleAssociations(i);%Get the number of possible associations    

    ProbabilityFactors=zeros(TrackNofPossibleAssociations,1);%Probability factors for calculating the probabilities

    yhat=TrackArray(i).C*TrackArray(i).PredictedState;%measurement prediction for the track
    S=TrackArray(i).PredictedInnovationCovariance;%innovation covariance for the track
    sqrtS=cholcov(S);%cholesky decomposition of the innovation covariance for calculating weighted square cheaply
    for j=1:TrackNofPossibleAssociations %For all possible associations
        MeasurementIndex=TrackPossibleAssociations(j);%Get the index of the measurement associated (it can be nonzero of zero)
        if MeasurementIndex>0 %Track is associated to a measurement
            y=MeasArray(:,MeasurementIndex);%Get the value of the measurement
            ytilda=y-yhat;%Calculate the innovation
            ProbabilityFactors(j)=Pd*exp(-0.5*sum((ytilda'/sqrtS).^2,2))/sqrt(det(2*pi*S));%likelihood factor
        else %Track is not associated to a measurement
            ProbabilityFactors(j)=1-Pd*Pg;%No measurement association factor
        end        
    end
    JPDAhypothesisMatrix(:,i)=TrackPossibleAssociations(IndexMatrix(:,i));%Fill in the associations corresponding to the tracks
    HypothesisProbabilities=HypothesisProbabilities.*ProbabilityFactors(IndexMatrix(:,i));%Update the hypothesis probabilities
end
HypothesisProbabilities=HypothesisProbabilities.*(betaFA.^(-sum(JPDAhypothesisMatrix>0,2)));%Multiply with the false alarm terms
%(m-sum(JPDAhypothesisMatrix>0,2)) is number of false alarms for the
%hypotheses. Since m is constant for all hypotheses, it was deleted above
for i=1:m 
    HypothesisProbabilities(sum(JPDAhypothesisMatrix==i,2)>1)=0;%set the probability of infeasible hypotheses to zero
    % The number sum(JPDAhypothesisMatrix==i,2) is the number the ith measurement appears in each
    % hypothesis. If that number is greater than 1, the hypothesis is
    % invalid i.e., a measurement is associated to more than one track
end
HypothesisProbabilities=HypothesisProbabilities/sum(HypothesisProbabilities); %Normalize the probabilities

JPDAprobs=zeros(Ntracks,m+1);%JPDA probability matrix
%Rows represent tracks
%Columns represent associations. Last column represents no measurement
%association.
for i=1:Ntracks %For all tracks
    TrackPossibleAssociations=TracksPossibleAssociations{i};%Get the possible associations to the track
    TrackNofPossibleAssociations=TracksNofPossibleAssociations(i);%Get the number of possible associations    
    for j=1:TrackNofPossibleAssociations 
        MeasIndex=TrackPossibleAssociations(j);
        if MeasIndex>0 %Track is associated to a measurement
            JPDAprobs(i,MeasIndex)=sum(HypothesisProbabilities(IndexMatrix(:,i)==j));%Sum the corresponding elements in the list of  hypothesis probabilities to obtain the corresponding probability
        else %Track is not associated to a measurement
            JPDAprobs(i,m+1)=sum(HypothesisProbabilities(IndexMatrix(:,i)==j));%Sum the corresponding elements in the list of  hypothesis probabilities to obtain the corresponding probability
        end
    end
end
end
%JPDA Probabilities calculated. QED