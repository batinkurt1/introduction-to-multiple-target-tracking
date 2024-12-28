classdef Initiator
     properties
        State               % The current state of the initiator
        Age                 % The age of the initiator (number of time steps since creation)
        TotalMeasurements   % Total number of measurements associated with this initiator
        TotalMissed         % Total number of missed measurements
        StateEstimate       % Estimated state of the initiator
        CovarianceEstimate  % Estimated covariance of the state
    end
    
    methods
        function obj = Initiator(initialMeasurement,measurement_variance,vmax,kappa)
            % Constructor method to initialize a new Initiator object
            % single point initiation
            obj.StateEstimate = [initialMeasurement;0;0];  % Set the initial state estimate to the first measurement
            % vmax = 50, kappa = 3
            obj.CovarianceEstimate = diag([measurement_variance,measurement_variance,(vmax/kappa)^2,(vmax/kappa)^2]);
            obj.Age = 1;  % Initialize age to 0
            obj.TotalMeasurements = 1;  % Start with one measurement (the initial measurement)
            obj.TotalMissed = 0;  % Initialize total missed measurements to 0
            obj.State = 1; 
        end
        
        function obj = predictionUpdate(obj, A, G, Q)
            obj.StateEstimate = A * obj.StateEstimate;  
            obj.CovarianceEstimate = A * obj.CovarianceEstimate * A' + G * Q * G';
        end

        function obj = measurementUpdate(obj, measurement, C, R)
            K = obj.CovarianceEstimate * C' / (C * obj.CovarianceEstimate * C' + R);
            obj.StateEstimate = obj.StateEstimate + K * (measurement - C * obj.StateEstimate);
            obj.CovarianceEstimate = (eye(size(obj.CovarianceEstimate)) - K * C) * obj.CovarianceEstimate;
            obj.CovarianceEstimate = 1/2 * (obj.CovarianceEstimate+obj.CovarianceEstimate');
        end
    end
end