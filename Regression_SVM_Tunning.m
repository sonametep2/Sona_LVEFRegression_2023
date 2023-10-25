% Load your data here or replace it with your dataset
load('your_data.mat');  % Replace 'your_data.mat' with your data file

% Define your features and EF values
X = your_features;  % Replace with your feature matrix
Y = your_EF_values;  % Replace with your EF values

% Hyperparameter grid
CValues = [0.1, 1, 10, 14];
KernelTypes = {'linear', 'polynomial', 'rbf'};

% Initialize variables to store RMSE and MAE values
numObservations = size(X, 1);
rmseValues = zeros(numObservations, length(CValues), length(KernelTypes));
maeValues = zeros(numObservations, length(CValues), length(KernelTypes));

% Initialize variables to store average RMSE and MAE
avgRMSE = zeros(length(CValues), length(KernelTypes));
avgMAE = zeros(length(CValues), length(KernelTypes));

% Outer loop: Leave-One-Out Cross-Validation
for testIndex = 1:numObservations
    trainIndices = [1:testIndex - 1, testIndex + 1:numObservations];
    testIndices = testIndex;

    % Initialize variables to store RMSE and MAE for inner loop
    innerRMSE = zeros(5, length(CValues), length(KernelTypes));
    innerMAE = zeros(5, length(CValues), length(KernelTypes));

    % Inner loop: 5-fold Cross-Validation for Hyperparameter Tuning
    for fold = 1:5
        % Split the training data into training and validation sets
        innerTrainIndices = trainIndices(mod(1:numObservations - 1, 5) == fold);
        innerValidationIndices = trainIndices(mod(1:numObservations - 1, 5) ~= fold);

        innerXTrain = X(innerTrainIndices, :);
        innerYTrain = Y(innerTrainIndices);

        innerXValidation = X(innerValidationIndices, :);
        innerYValidation = Y(innerValidationIndices);

        for j = 1:length(CValues)
            for k = 1:length(KernelTypes)
                % Train the SVM regression model
                mdl = fitrsvm(innerXTrain, innerYTrain, 'KernelFunction', KernelTypes{k}, 'BoxConstraint', CValues(j));

                % Predict EF values for the validation set
                YPred = predict(mdl, innerXValidation);

                % Calculate RMSE and MAE for the current parameter combination
                rmse = sqrt(mean((YPred - innerYValidation).^2));
                mae = mean(abs(YPred - innerYValidation));

                innerRMSE(fold, j, k) = rmse;
                innerMAE(fold, j, k) = mae;
            end
        end
    end

    % Find the best hyperparameter combination (minimal RMSE)
    [bestFold, bestCIndex, bestKernelIndex] = ind2sub(size(innerRMSE), find(innerRMSE == min(innerRMSE(:)));

    % Train the SVM regression model with the best hyperparameters on the full training set
    mdl = fitrsvm(X(trainIndices, :), Y(trainIndices), 'KernelFunction', KernelTypes{bestKernelIndex}, 'BoxConstraint', CValues(bestCIndex));

    % Predict EF values for the test set
    YPred = predict(mdl, X(testIndices, :));

    % Calculate RMSE and MAE for the current test fold
    rmseValues(testIndex, bestCIndex, bestKernelIndex) = sqrt(mean((YPred - Y(testIndices)).^2));
    maeValues(testIndex, bestCIndex, bestKernelIndex) = mean(abs(YPred - Y(testIndices)));
end

% Calculate average RMSE and MAE across all observations
for j = 1:length(CValues)
    for k = 1:length(KernelTypes)
        avgRMSE(j, k) = mean(squeeze(rmseValues(:, j, k)));
        avgMAE(j, k) = mean(squeeze(maeValues(:, j, k)));
    end
end
% Find the best hyperparameter combination based on minimal RMSE
[minRMSE, minRMSEIndex] = min(avgRMSE(:));
[bestCIndex, bestKernelIndex] = ind2sub(size(avgRMSE), minRMSEIndex);
bestC = CValues(bestCIndex);
bestKernel = KernelTypes{bestKernelIndex};

fprintf('Best Hyperparameter Combination:\n');
fprintf('Kernel Type: %s\n', bestKernel);
fprintf('Box Constraint (C): %f\n', bestC);


% Display the average RMSE and MAE values
disp('Average RMSE for Each Hyperparameter Combination:');
disp(avgRMSE);
disp('Average MAE for Each Hyperparameter Combination:');
disp(avgMAE);
