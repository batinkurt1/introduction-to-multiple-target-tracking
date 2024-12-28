function AssociationDecisions = Associate(InitiatorArray,MeasurementSet,GateDecisions)
    AssociationDecisions = zeros(1,length(InitiatorArray));
    
    % first find the minimum distance associations
    for k = 1:length(InitiatorArray)
        initiator_associations = GateDecisions(k,:);
        indices = find(initiator_associations == 1);
        minimum_distance = inf;
        minimum_distance_index = 0;
        for i = indices
            current_distance = norm(InitiatorArray(k).StateEstimate(1:2,:)...
                 - MeasurementSet{i});
            if current_distance < minimum_distance
                minimum_distance = current_distance;
                minimum_distance_index = i;
            end
        end  
    AssociationDecisions(k) = minimum_distance_index;
    end
    
    % then if the same measurement is assigned to multiple initiators,
    % assign the measurement to the oldest initiator
    for k = 1:length(InitiatorArray)
        for l = k+1:length(InitiatorArray)
            if AssociationDecisions(k) == 0
                break
            end
            if AssociationDecisions(k) == AssociationDecisions(l) ...
                    && AssociationDecisions(k) ~= 0
                if InitiatorArray(k).Age >= InitiatorArray(l).Age
                    AssociationDecisions(l) = 0;
                elseif InitiatorArray(k).Age < InitiatorArray(l).Age
                    AssociationDecisions(k) = 0;
                end
            end
        end
    end
end