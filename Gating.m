function GateDecisions = Gating(InitiatorArray,MeasurementSet,T,vmax,gate_threshold,C,R)
GateDecisions = zeros(length(InitiatorArray),length(MeasurementSet));
for k = 1:length(InitiatorArray)
    for l = 1:length(MeasurementSet)
        % fresh initiator, use rectangular gating
        if InitiatorArray(k).Age == 1
            if norm(InitiatorArray(k).StateEstimate(1:2,:)...
                     - MeasurementSet{l}) <= T*vmax
                GateDecisions(k,l) = 1;
            end
        % if initiator is not fresh, use ellipsoidal gating
        else
            S = C * InitiatorArray(k).CovarianceEstimate * C'+R;
            U = cholcov(S);
            if norm(U\(MeasurementSet{l} - ...
                    InitiatorArray(k).StateEstimate(1:2,:))) <= sqrt(gate_threshold)
                GateDecisions(k,l) = 1;
            end
        end
    end
end