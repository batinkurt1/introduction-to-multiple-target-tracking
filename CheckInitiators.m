function [UpdatedInitiatorArray,ConfirmedTracks] = CheckInitiators(InitiatorArray,N1,M2,N2)
    UpdatedInitiatorArray = [];
    ConfirmedTracks = [];
    for k = 1:length(InitiatorArray)
        if InitiatorArray(k).Age<= N1
            if InitiatorArray(k).TotalMissed > 0 
                InitiatorArray(k).State = 0;
            else
                InitiatorArray(k).State = 1;
            end
        else
            if InitiatorArray(k).TotalMissed > (N2-M2)
                InitiatorArray(k).State = 0;
            elseif InitiatorArray(k).TotalMeasurements >= (N1+M2)
                InitiatorArray(k).State = 2;
            else
                InitiatorArray(k).State = 1;
            end
        end
        if InitiatorArray(k).State == 1
            UpdatedInitiatorArray = [UpdatedInitiatorArray, InitiatorArray(k)];
        elseif InitiatorArray(k).State == 2
            ConfirmedTracks = [ConfirmedTracks, InitiatorArray(k)];
        else
        end
    end
end