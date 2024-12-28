function InitiatorArray = Update(InitiatorArray,MeasurementSet,AssociationDecisions,A,G,Q,T,measurement_variance,C,R,vmax,kappa)
    for k = 1:length(InitiatorArray)
        InitiatorArray(k).Age = InitiatorArray(k).Age + 1;
        % no measurement
        if AssociationDecisions(k) == 0
            InitiatorArray(k).TotalMissed = ...
                InitiatorArray(k).TotalMissed+1;
            InitiatorArray(k) = ...
                InitiatorArray(k).predictionUpdate(A,G,Q);
        % there is an associated measurement
        else
            InitiatorArray(k).TotalMeasurements = ...
                InitiatorArray(k).TotalMeasurements+1;
            % if age is 2 apply two point initiaton
            if InitiatorArray(k).Age == 2
                InitiatorArray(k).StateEstimate = ...
                    [MeasurementSet{AssociationDecisions(k)};...
                    (MeasurementSet{AssociationDecisions(k)} - ...
                    InitiatorArray(k).StateEstimate(1:2,:))/T];
                InitiatorArray(k).CovarianceEstimate = ...
                    [measurement_variance*eye(2),...
                    measurement_variance/T*eye(2);...
                    measurement_variance/T*eye(2),...
                    2*measurement_variance/T^2*eye(2)];
            else
                InitiatorArray(k) = ... 
                    InitiatorArray(k).measurementUpdate(...
                    MeasurementSet{AssociationDecisions(k)},C,R);
            end
            InitiatorArray(k) = ... 
                InitiatorArray(k).predictionUpdate(A,G,Q);
        end
    end
    % create new initiators for unassociated measurements
    for i = 1:length(MeasurementSet)
        if ~ismember(i,AssociationDecisions)
            fresh_initiator = Initiator(MeasurementSet{i}...
                ,measurement_variance,vmax,kappa);
            InitiatorArray = [InitiatorArray,fresh_initiator];
        end
    end
end