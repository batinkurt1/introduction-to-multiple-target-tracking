function SingleTargetTracker(detections,NNorPDA,measurement_sigma,vmax,kappa,xrange,yrange,T,A,G,Q,C,R,gate_threshold,N1,M2,N2,track_deletion_treshold,t,trueTargets,P_d,beta_FA)
figure;
hold on;
TrackCount = 0;
ConfirmedTracks = [];
AllTracks = [];
for i = 1:length(t)
    current_detections_x = detections{2,i};
    current_detections_y = detections{3,i};
    % for NN plotting
    AssociatedMeasurementSet = {};
    if i == 1
        for k = 1:length(current_detections_x)
            fresh_initiator = Initiator([current_detections_x(k);current_detections_y(k)]...
                ,measurement_sigma^2,vmax,kappa);
            InitiatorArray(k) = fresh_initiator;
        end
    else
        MeasurementSet = cell(1,length(current_detections_x));
        for k = 1:length(current_detections_x)
            MeasurementSet{k} = [current_detections_x(k);current_detections_y(k)];
        end
        OriginalMeasurementSet = MeasurementSet;
        indexes_to_delete = [];
        for k =1:length(ConfirmedTracks)
            [ConfirmedTracks(k),MeasurementSet] = ...
                ConfirmedTracks(k).Gating(MeasurementSet,gate_threshold);
            if NNorPDA == 1
            [ConfirmedTracks(k),associated_measurement,unused_measurements] = ConfirmedTracks(k).NN();
            MeasurementSet = [MeasurementSet,unused_measurements];
            if associated_measurement ~= inf
                AssociatedMeasurementSet{end+1} = associated_measurement;
                % measurement exists
                ConfirmedTracks(k) = ConfirmedTracks(k).measurementUpdate(associated_measurement,t(i));
                ConfirmedTracks(k).NumberOfConsecutiveMissedDetections = 0;
                ConfirmedTracks(k) = ConfirmedTracks(k).predictionUpdate();
            else
                % no measurement
                ConfirmedTracks(k) = ConfirmedTracks(k).predictionUpdate();
                ConfirmedTracks(k).NumberOfConsecutiveMissedDetections...
                    = ConfirmedTracks(k).NumberOfConsecutiveMissedDetections + 1;
            end
            elseif NNorPDA == 2
                
                ConfirmedTracks(k) = ConfirmedTracks(k).PDA(P_d,chi2cdf(gate_threshold,2),beta_FA,t(i));
                ConfirmedTracks(k) = ConfirmedTracks(k).predictionUpdate();
                % check
%                 MeasurementSet = OriginalMeasurementSet;
            end

            if ConfirmedTracks(k).NumberOfConsecutiveMissedDetections >= track_deletion_treshold
                indexes_to_delete = [indexes_to_delete,k];
            end
        end
        ConfirmedTracksAfterDeletion = [];
        for k = 1:length(ConfirmedTracks)
            if ~ismember(k,indexes_to_delete)
                ConfirmedTracksAfterDeletion = [ConfirmedTracksAfterDeletion,ConfirmedTracks(k)];
            else
                AllTracks = [AllTracks,ConfirmedTracks(k)];
            end
        end
        ConfirmedTracks = ConfirmedTracksAfterDeletion; 
        
        % for initiators
        GateDecisions = Gating(InitiatorArray,MeasurementSet,T,vmax,gate_threshold,C,R);
        AssociationDecisions = Associate(InitiatorArray,MeasurementSet,GateDecisions);
        InitiatorArray = Update(InitiatorArray,MeasurementSet,...
            AssociationDecisions,A,G,Q,T,measurement_sigma^2,C,R,vmax,kappa);
        [InitiatorArray,NewTracks] = CheckInitiators(InitiatorArray,N1,M2,N2);
        for k = 1:length(NewTracks)
            TrackCount = TrackCount + 1;
            ConfirmedTracks = [ConfirmedTracks, ...
                ConfirmedTrack(NewTracks(k).CovarianceEstimate,...
                NewTracks(k).StateEstimate,A,G,Q,C,R,TrackCount,P_d,chi2cdf(gate_threshold,2))];
        end
    end

    scatter(detections{2,i}, detections{3,i},'black');
    for l = 1:length(AssociatedMeasurementSet)
        current_point = AssociatedMeasurementSet{l};
        scatter(current_point(1),current_point(2),'blue')
    end

    title(['Time: ', num2str((i-1)*T)]);
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    xlim([0, xrange]);
    ylim([0, yrange]); 

    % Update the plot
    drawnow;


    pause(0.1);


end
hold off;


AllTracks = [AllTracks, ConfirmedTracks];


figure(4);
hold on;
figure(5);
hold on;
figure(6);
hold on;
for k = 1:length(AllTracks)
    for l = 1:length(AllTracks(k).StateEstimateHistorySet)
        if l == 1
            estimated_x_position = zeros(1,length(AllTracks(k).StateEstimateHistorySet));
            estimated_y_position = zeros(1,length(AllTracks(k).StateEstimateHistorySet));
            estimated_x_velocity = zeros(1,length(AllTracks(k).StateEstimateHistorySet));
            estimated_y_velocity = zeros(1,length(AllTracks(k).StateEstimateHistorySet));
        end
    current_state = AllTracks(k).StateEstimateHistorySet{l};
    estimated_x_position(l) = current_state(1);
    estimated_y_position(l) = current_state(2);
    estimated_x_velocity(l) = current_state(3);
    estimated_y_velocity(l) = current_state(4);
    end
%     plot(x_position_true,y_position_true,LineWidth=1.5,Color="#77AC30");
    figure(4);
    plot(AllTracks(k).CorrespondingTimeSteps,estimated_x_position,LineWidth=1.5,DisplayName=['Estimated Target Trajectory ',num2str(k)])
    title("True Targets vs. Estimated Target Trajectories - x position");
    ylabel("x position");
    ylim([0,xrange]);
    xlabel("Time");
    xlim([0,t(end)]);
%     legend("True Target Trajectory","Estimated Target Trajectory");
    grid on;
    figure(5);
    plot(AllTracks(k).CorrespondingTimeSteps,estimated_y_position,LineWidth=1.5,DisplayName=['Estimated Target Trajectory ',num2str(k)])
    title("True Targets vs. Estimated Target Trajectories - y position");
    ylabel("y position");
    ylim([0,yrange]);
    xlabel("Time");
    xlim([0,t(end)]);
%     legend("True Target Trajectory","Estimated Target Trajectory");
    grid on;
    figure(6);
    plot(estimated_x_position,estimated_y_position,LineWidth=1.5,DisplayName=['Estimated Target Trajectory ',num2str(k)])
    title("True Targets vs. Estimated Target Trajectories");
    ylabel("y position");
    ylim([0,yrange]);
    xlabel("x position");
    xlim([0,xrange]);
    grid on;
end

[~,b] = size(trueTargets);
for i = 1:b
    figure(4);
    plot(trueTargets{3,i},trueTargets{1,i},':',LineWidth=1.5,DisplayName=['True Target Trajectory ',num2str(i)]);
    figure(5);
    plot(trueTargets{3,i},trueTargets{2,i},':',LineWidth=1.5,DisplayName=['True Target Trajectory ',num2str(i)]);
    figure(6);
    plot(trueTargets{1,i},trueTargets{2,i},':',LineWidth=1.5,DisplayName=['True Target Trajectory ',num2str(i)]);
end
figure(4);
legend();
figure(5);
legend();
figure(6);
legend();
end