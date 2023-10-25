% Load your data here or replace it with your dataset
load('your_data.mat');  % Replace 'your_data.mat' with your data file

% Define your features and EF values
X = your_features;  % Replace with your feature matrix
Y = your_EF_values;  % Replace with your EF values

% Hyperparameter grid for GPR
KernelFunctions = {'rational quadratic', 'matern52'};
AlphaRange = [0.01, 0.1, 1];

% Initialize variables to store RMSE and MAE values
numObservations = size(X, 1);
rmseValues = zeros(numObservations, length(KernelFunctions), length(AlphaRange));
maeValues = zeros(numObservations, length(KernelFunctions), length(AlphaRange));

% Initialize variables to store average RMSE and MAE
avgRMSE = zeros(length(KernelFunctions), length(AlphaRange));
avgMAE = zeros(length(KernelFunctions), length(AlphaRange));

% Outer loop: Leave-One-Out Cross-Validation
for testIndex = 1:numObservations
    trainIndices = [1:testIndex - 1, testIndex + 1:numObservations];
    testIndices = testIndex;

    % Initialize variables to store RMSE and MAE for inner loop
    innerRMSE = zeros(5, length(KernelFunctions), length(AlphaRange));
    innerMAE = zeros(5, length(KernelFunctions), length(AlphaRange));

    % Inner loop: 5-fold Cross-Validation for Hyperparameter Tuning for GPR
    for fold = 1:5
        % Split the training data into training and validation sets
        innerTrainIndices = trainIndices(mod(1:numObservations - 1, 5) == fold);
        innerValidationIndices = trainIndices(mod(1:numObservations - 1, 5) ~= fold);

        innerXTrain = X(innerTrainIndices, :);
        innerYTrain = Y(innerTrainIndices);

        innerXValidation = X(innerValidationIndices, :);
        innerYValidation = Y(innerValidationIndices);

        for kfIdx = 1:length(KernelFunctions)
            for alphaIdx = 1:length(AlphaRange)
                % Set hyperparameters for GPR
                kernelFunc = KernelFunctions{kfIdx};
                alpha = AlphaRange(alphaIdx);

                % Create a GPR model with the specified hyperparameters
                gprMdl = fitrgp(innerXTrain, innerYTrain, 'KernelFunction', kernelFunc, 'Standardize', 1, 'BasisFunction', 'constant', 'SigmaLowerBound', alpha);

                % Predict EF values for the validation set
                YPred = predict(gprMdl, innerXValidation);

                % Calculate RMSE and MAE for the current parameter combination
                rmse = sqrt(mean((YPred - innerYValidation).^2));
                mae = mean(abs(YPred - innerYValidation));

                innerRMSE(fold, kfIdx, alphaIdx) = rmse;
                innerMAE(fold, kfIdx, alphaIdx) = mae;
            end
        end
    end

    % Find the best hyperparameter combination (minimal RMSE) for GPR
    [bestFold, bestKfIndex, bestAlphaIndex] = ind2sub(size(innerRMSE), find(innerRMSE == min(innerRMSE(:)));

    % Train the GPR model with the best hyperparameters on the full training set
    bestKernelFunc = KernelFunctions{bestKfIndex};
    bestAlpha = AlphaRange(bestAlphaIndex);

    gprMdl = fitrgp(X(trainIndices, :), Y(trainIndices), 'KernelFunction', bestKernelFunc, 'Standardize', 1, 'BasisFunction', 'constant', 'SigmaLowerBound', bestAlpha);

    % Predict EF values for the test set
    YPred = predict(gprMdl, X(testIndices, :));

    % Calculate RMSE and MAE for the current test fold
    rmseValues(testIndex, bestKfIndex, bestAlphaIndex) = sqrt(mean((YPred - Y(testIndices)).^2));
    maeValues(testIndex, bestKfIndex, bestAlphaIndex) = mean(abs(YPred - Y(testIndices)));
end

% Calculate average RMSE and MAE across all observations
for kfIdx = 1:length(KernelFunctions)
    for alphaIdx = 1:length(AlphaRange)
        avgRMSE(kfIdx, alphaIdx) = mean(squeeze(rmseValues(:, kfIdx, alphaIdx)));
        avgMAE(kfIdx, alphaIdx) = mean(squeeze(maeValues(:, kfIdx, alphaIdx)));
    end
end

% Find the best hyperparameter combination based on minimal RMSE for GPR
[minRMSE, minRMSEIndex] = min(avgRMSE(:));
[bestKfIndex, bestAlphaIndex] = ind2sub(size(avgRMSE), minRMSEIndex);
bestKernelFunc = KernelFunctions{bestKfIndex};
bestAlpha = AlphaRange(bestAlphaIndex);

fprintf('Best Hyperparameter Combination for GPR:\n');
fprintf('Kernel Function: %s\n', bestKernelFunc);
fprintf('Alpha: %f\n', bestAlpha);

% Display the average RMSE and MAE values for GPR
disp('Average RMSE for Each Hyperparameter Combination (GPR):');
disp(avgRMSE);
disp('Average MAE for Each Hyperparameter Combination (GPR):');
disp(avgMAE);
