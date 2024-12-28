function MultiTargetTracker(detections,GNNorJPDA,measurement_sigma,vmax,kappa,xrange,yrange,T,A,G,Q,C,R,gate_threshold,N1,M2,N2,track_deletion_treshold,t,trueTargets,P_d,beta_FA)

beta_EX = beta_FA;


figure;
hold on;
TrackCount = 0;
ConfirmedTracks = [];
AllTracks = [];
for i = 1:length(t)
    current_detections_x = detections{2,i};
    current_detections_y = detections{3,i};
    % for GNN plotting
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
        ValidationMatrix = zeros(length(MeasurementSet),length(ConfirmedTracks));
        for k = 1:length(ConfirmedTracks)
            [ConfirmedTracks(k),ValidationMatrix] = ...
                ConfirmedTracks(k).Gating_MTT(ValidationMatrix,MeasurementSet,gate_threshold,k);
        end
        
        % GNN
        if GNNorJPDA == 1
            % form the AssignmentMatrix
            % AssignmentMatrix = zeros(length(MeasurementSet),length(MeasurementSet)+length(ConfirmedTracks));
            TargetAssignments = ValidationMatrix;
            [row, col] = find(TargetAssignments == 1);
            for k = 1:length(row)

                PredictedMeasurement = ConfirmedTracks(col(k)).C * ...
                    ConfirmedTracks(col(k)).PredictedState;

                TargetAssignments(row(k), col(k)) = ...
                    log10(ConfirmedTracks(col(k)).P_d * ...
                    mvnpdf(MeasurementSet{row(k)},...
                    PredictedMeasurement, ...
                    ConfirmedTracks(col(k)).PredictedInnovationCovariance)...
                    /(1 - ConfirmedTracks(col(k)).P_d ...
                    * ConfirmedTracks(col(k)).P_g));
            end

            ExternalAssignments = eye(length(MeasurementSet))*log10(beta_EX);

            AssignmentMatrix = [TargetAssignments,ExternalAssignments];

            AssignmentMatrix(AssignmentMatrix == 0) = -10^(10);

            [MeasurementAssignments,~] = Auction(AssignmentMatrix);
            

            ExternalMeasurementSet = {};

            for k = 1:length(MeasurementAssignments)
                if MeasurementAssignments(k) > length(ConfirmedTracks)
                    % Measurement is external source, send to initiator
                    % logic
                    ExternalMeasurementSet{end+1} = MeasurementSet{k};
                elseif MeasurementAssignments(k) <= length(ConfirmedTracks)
                    % Measurement belongs to a track
                    ConfirmedTracks(MeasurementAssignments(k)).GNNAssociatedMeasurement = MeasurementSet{k};
                    AssociatedMeasurementSet{end+1} = MeasurementSet{k};
                end
            end
            for k = 1:length(ConfirmedTracks)
                if ConfirmedTracks(k).GNNAssociatedMeasurement ~= inf
                    % measurement exists
                    ConfirmedTracks(k) = ConfirmedTracks(k).measurementUpdate(ConfirmedTracks(k).GNNAssociatedMeasurement,t(i));
                    ConfirmedTracks(k).GNNAssociatedMeasurement = inf;
                    ConfirmedTracks(k).NumberOfConsecutiveMissedDetections = 0;
                    ConfirmedTracks(k) = ConfirmedTracks(k).predictionUpdate();
                else
                    % no measurement
                    ConfirmedTracks(k) = ConfirmedTracks(k).predictionUpdate();
                    ConfirmedTracks(k).NumberOfConsecutiveMissedDetections...
                    = ConfirmedTracks(k).NumberOfConsecutiveMissedDetections + 1;
                end

            end


        
        elseif GNNorJPDA == 2
            JPDAprobs = JPDA_ProbCalc(length(MeasurementSet),...
                length(ConfirmedTracks),P_d,chi2cdf(gate_threshold,2),...
                beta_FA,ConfirmedTracks,MeasurementSet,ValidationMatrix);
            ExternalMeasurementSet = {};
            for k = 1:length(MeasurementSet)
                if ~sum(ValidationMatrix(k,:))
                    ExternalMeasurementSet{end+1} = MeasurementSet{k};
                end
            end
            for k = 1:length(ConfirmedTracks)
                ConfirmedTracks(k) = ConfirmedTracks(k).JPDA(JPDAprobs(k,:),MeasurementSet,t(i));
                ConfirmedTracks(k) = ConfirmedTracks(k).predictionUpdate();
            end
        end
        
        for k = 1:length(ConfirmedTracks)
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
        GateDecisions = Gating(InitiatorArray,ExternalMeasurementSet,T,vmax,gate_threshold,C,R);
        AssociationDecisions = Associate(InitiatorArray,ExternalMeasurementSet,GateDecisions);
        InitiatorArray = Update(InitiatorArray,ExternalMeasurementSet,...
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

figure(8);
hold on;
figure(9);
hold on;
figure(10);
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
    figure(8);
    plot(AllTracks(k).CorrespondingTimeSteps,estimated_x_position,LineWidth=1.5,DisplayName=['Estimated Target Trajectory ',num2str(k)])
    title("True Targets vs. Estimated Target Trajectories - x position");
    ylabel("x position");
    ylim([0,xrange]);
    xlabel("Time");
    xlim([0,t(end)]);
%     legend("True Target Trajectory","Estimated Target Trajectory");
    grid on;
    figure(9);
    plot(AllTracks(k).CorrespondingTimeSteps,estimated_y_position,LineWidth=1.5,DisplayName=['Estimated Target Trajectory ',num2str(k)])
    title("True Targets vs. Estimated Target Trajectories - y position");
    ylabel("y position");
    ylim([0,yrange]);
    xlabel("Time");
    xlim([0,t(end)]);
%     legend("True Target Trajectory","Estimated Target Trajectory");
    grid on;
    figure(10);
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
    figure(8);
    plot(trueTargets{3,i},trueTargets{1,i},':',LineWidth=1.5,DisplayName=['True Target Trajectory ',num2str(i)]);
    figure(9);
    plot(trueTargets{3,i},trueTargets{2,i},':',LineWidth=1.5,DisplayName=['True Target Trajectory ',num2str(i)]);
    figure(10);
    plot(trueTargets{1,i},trueTargets{2,i},':',LineWidth=1.5,DisplayName=['True Target Trajectory ',num2str(i)]);
end
figure(8);
legend();
figure(9);
legend();
figure(10);
legend();
end
