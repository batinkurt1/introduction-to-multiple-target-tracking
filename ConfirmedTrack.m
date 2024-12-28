classdef ConfirmedTrack
    properties
        StateEstimate
        CovarianceEstimate
        PredictedState
        PredictedCovariance
        PredictedInnovationCovariance
        A
        G
        Q
        C
        R
        AssociatedMeasurementSet
        NumberOfConsecutiveMissedDetections
        StateEstimateHistorySet
        CorrespondingTimeSteps
        CovarianceEstimateHistorySet
        TrackNumber
        P_d
        P_g
        GNNAssociatedMeasurement;
    end

    methods
        function obj = ConfirmedTrack(PredictedCovariance,PredictedState,A,G,Q,C,R,TrackNumber,P_d,P_g)
            obj.PredictedState = PredictedState;
            obj.PredictedCovariance = PredictedCovariance;
            obj.StateEstimate = PredictedState;
            obj.CovarianceEstimate = PredictedCovariance;
            obj.A = A;
            obj.G = G;
            obj.Q = Q;
            obj.C = C;
            obj.R = R;
            obj.PredictedInnovationCovariance = ...
                C * obj.PredictedCovariance * C' + R;
            obj.AssociatedMeasurementSet = {};
            obj.NumberOfConsecutiveMissedDetections = 0;
            obj.CorrespondingTimeSteps = [];
            obj.TrackNumber = TrackNumber;
            obj.P_d = P_d;
            obj.P_g = P_g;
            obj.GNNAssociatedMeasurement = inf;
        end
        function obj = predictionUpdate(obj)
            obj.PredictedState = obj.A * obj.StateEstimate;  
            obj.PredictedCovariance = ...
                obj.A * obj.CovarianceEstimate * obj.A' +...
                obj.G * obj.Q * obj.G';
            obj.PredictedInnovationCovariance = ...
                obj.C * obj.PredictedCovariance * obj.C' + obj.R;
        end

        function obj = measurementUpdate(obj, measurement,current_time)
            K = obj.PredictedCovariance * obj.C' /...
                obj.PredictedInnovationCovariance;
            obj.StateEstimate = obj.PredictedState + ...
                K * (measurement - obj.C * obj.PredictedState);
            obj.CovarianceEstimate = (eye(size(obj.PredictedCovariance))...
                - K * obj.C) * obj.PredictedCovariance;
            obj.CovarianceEstimate = 1/2 * ...
                (obj.CovarianceEstimate+obj.CovarianceEstimate');
            obj.StateEstimateHistorySet{end+1} = obj.StateEstimate;
            obj.CorrespondingTimeSteps(end+1) = current_time;
            obj.CovarianceEstimateHistorySet{end+1} = obj.CovarianceEstimate;
        end
        function [obj,UnassociatedMeasurementSet] = Gating(obj,MeasurementSet,gate_threshold)
            U = cholcov(obj.PredictedInnovationCovariance);
            UnassociatedMeasurementSet = {};
            for l = 1:length(MeasurementSet)
            if norm(U\(MeasurementSet{l}-obj.PredictedState(1:2,:))) <= sqrt(gate_threshold)
                obj.AssociatedMeasurementSet{end+1} = MeasurementSet{l}; 
            else
                UnassociatedMeasurementSet{end+1} = MeasurementSet{l};
            end
            end
        end
        function [obj,associated_measurement,unused_measurements] = NN(obj)
            minimum_distance = inf;
            minimum_distance_index = 0;
            unused_measurements = obj.AssociatedMeasurementSet;
            for l = 1:length(obj.AssociatedMeasurementSet)
                current_distance = norm(obj.PredictedState(1:2,:)...
                 - obj.AssociatedMeasurementSet{l});
                if current_distance < minimum_distance
                    minimum_distance = current_distance;
                    minimum_distance_index = l;
                end
            end
                if minimum_distance_index ~=0
                    associated_measurement = ...
                        obj.AssociatedMeasurementSet{minimum_distance_index};
                    unused_measurements(minimum_distance_index) = [];
                    obj.AssociatedMeasurementSet = {};
                else
                    % if no measurement send inf
                    associated_measurement = inf;
                    obj.AssociatedMeasurementSet = {};
                end
        end
        function obj = PDA(obj,Pd,Pg,beta_FA,current_time)
            % calculate weights
            % first one is theta_0
            weights = zeros(1,length(obj.AssociatedMeasurementSet) + 1);
            PredictedMeasurement = obj.C * obj.PredictedState;
            weights(1) = (1-Pd*Pg)*beta_FA;
            for l = 2:length(weights)
                weights(l) = Pd*...
                    mvnpdf(obj.AssociatedMeasurementSet{l-1},...
                    PredictedMeasurement,...
                    obj.PredictedInnovationCovariance);

            end
            % normalize weights
            weights = weights./sum(weights);
            EquivalentMeasurement = weights(1)*PredictedMeasurement;
            for l = 1:length(obj.AssociatedMeasurementSet)
                EquivalentMeasurement = EquivalentMeasurement + ...
                    weights(l+1)*obj.AssociatedMeasurementSet{l};
            end
            % measurement update

            % state estimate calculation
            K = obj.PredictedCovariance * obj.C' /...
                obj.PredictedInnovationCovariance;
            obj.StateEstimate = obj.PredictedState + ...
                K * (EquivalentMeasurement - obj.C * obj.PredictedState);

            % covariance calculation 

            obj.CovarianceEstimate =  ...
               weights(1)*(obj.PredictedCovariance + ...
               (obj.PredictedState - obj.StateEstimate) * ... 
               (obj.PredictedState - obj.StateEstimate)');
            % sigma k given k ^ i 
            sigma_kk = obj.PredictedCovariance - ...
                K*obj.PredictedInnovationCovariance*K';
            for l = 1:length(obj.AssociatedMeasurementSet)
                % ith predicted state
                state_kk = obj.PredictedState + ...
                    K * (obj.AssociatedMeasurementSet{l} - ...
                    obj.C*obj.PredictedState);
                obj.CovarianceEstimate = obj.CovarianceEstimate + ...
                weights(l+1) * (sigma_kk + (state_kk - ... 
                obj.StateEstimate) * (state_kk - obj.StateEstimate)');
            end
            obj.CovarianceEstimate = 1/2 * ...
            (obj.CovarianceEstimate+obj.CovarianceEstimate');
            obj.StateEstimateHistorySet{end+1} = obj.StateEstimate;
            obj.CorrespondingTimeSteps(end+1) = current_time;
            obj.CovarianceEstimateHistorySet{end+1} = obj.CovarianceEstimate;
            obj.AssociatedMeasurementSet = {};
            if  length(weights) == 1
                obj.NumberOfConsecutiveMissedDetections = ... 
                    obj.NumberOfConsecutiveMissedDetections + 1;
            else
                obj.NumberOfConsecutiveMissedDetections = 0;
            end
        end
        function [obj,ValidationMatrix] = Gating_MTT(obj,ValidationMatrix,MeasurementSet,gate_threshold,k)
            U = cholcov(obj.PredictedInnovationCovariance);
            for l = 1:length(MeasurementSet)
                if norm(U\(MeasurementSet{l}-obj.PredictedState(1:2,:))) <= sqrt(gate_threshold)
                    ValidationMatrix(l,k) = 1; 
                end
            end
        end

        function [obj,ExternalMeasurementSet] = JPDA(obj,weights,MeasurementSet,current_time)
            % moving theta zero into the first index
            weights = [weights(end), weights(1:end-1)];
            PredictedMeasurement = obj.C * obj.PredictedState;
            EquivalentMeasurement = weights(1)*PredictedMeasurement;
            for l = 1:length(MeasurementSet)
                EquivalentMeasurement = EquivalentMeasurement + ...
                    weights(l+1)*MeasurementSet{l};
            end
            % measurement update

            % state estimate calculation
            K = obj.PredictedCovariance * obj.C' /...
                obj.PredictedInnovationCovariance;
            obj.StateEstimate = obj.PredictedState + ...
                K * (EquivalentMeasurement - obj.C * obj.PredictedState);

            % covariance calculation 

            obj.CovarianceEstimate =  ...
               weights(1)*(obj.PredictedCovariance + ...
               (obj.PredictedState - obj.StateEstimate) * ... 
               (obj.PredictedState - obj.StateEstimate)');
            % sigma k given k ^ i 
            sigma_kk = obj.PredictedCovariance - ...
                K*obj.PredictedInnovationCovariance*K';
            for l = 1:length(MeasurementSet)
                % ith predicted state
                state_kk = obj.PredictedState + ...
                    K * (MeasurementSet{l} - ...
                    obj.C*obj.PredictedState);
                obj.CovarianceEstimate = obj.CovarianceEstimate + ...
                weights(l+1) * (sigma_kk + (state_kk - ... 
                obj.StateEstimate) * (state_kk - obj.StateEstimate)');
            end
            obj.CovarianceEstimate = 1/2 * ...
            (obj.CovarianceEstimate+obj.CovarianceEstimate');
            obj.StateEstimateHistorySet{end+1} = obj.StateEstimate;
            obj.CorrespondingTimeSteps(end+1) = current_time;
            obj.CovarianceEstimateHistorySet{end+1} = obj.CovarianceEstimate;
            if  weights(1) == max(weights)
            % if  weights(1) == 1
                obj.NumberOfConsecutiveMissedDetections = ... 
                    obj.NumberOfConsecutiveMissedDetections + 1;
            else
                obj.NumberOfConsecutiveMissedDetections = 0;
            end
        end

    end

end