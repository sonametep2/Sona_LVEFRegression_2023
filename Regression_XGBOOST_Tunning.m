% Load your data here or replace it with your dataset
load('your_data.mat');  % Replace 'your_data.mat' with your data file

% Define your features and EF values
X = your_features;  % Replace with your feature matrix
Y = your_EF_values;  % Replace with your EF values

% Hyperparameter grid for XGBoost
LearningRates = [0.1, 0.2];
MaxDepths = [3, 5];
Gammas = [0, 0.01];

% Initialize variables to store RMSE and MAE values
numObservations = size(X, 1);
rmseValues = zeros(numObservations, length(LearningRates), length(MaxDepths), length(Gammas));
maeValues = zeros(numObservations, length(LearningRates), length(MaxDepths), length(Gammas));

% Initialize variables to store average RMSE and MAE
avgRMSE = zeros(length(LearningRates), length(MaxDepths), length(Gammas));
avgMAE = zeros(length(LearningRates), length(MaxDepths), length(Gammas));

% Outer loop: Leave-One-Out Cross-Validation
for testIndex = 1:numObservations
    trainIndices = [1:testIndex - 1, testIndex + 1:numObservations];
    testIndices = testIndex;

    % Initialize variables to store RMSE and MAE for inner loop
    innerRMSE = zeros(5, length(LearningRates), length(MaxDepths), length(Gammas));
    innerMAE = zeros(5, length(LearningRates), length(MaxDepths), length(Gammas));

    % Inner loop: 5-fold Cross-Validation for Hyperparameter Tuning for XGBoost
    for fold = 1:5
        % Split the training data into training and validation sets
        innerTrainIndices = trainIndices(mod(1:numObservations - 1, 5) == fold);
        innerValidationIndices = trainIndices(mod(1:numObservations - 1, 5) ~= fold);

        innerXTrain = X(innerTrainIndices, :);
        innerYTrain = Y(innerTrainIndices);

        innerXValidation = X(innerValidationIndices, :);
        innerYValidation = Y(innerValidationIndices);

        for lrIdx = 1:length(LearningRates)
            for depthIdx = 1:length(MaxDepths)
                for gammaIdx = 1:length(Gammas)
                    % Set hyperparameters for XGBoost
                    params = struct( ...
                        'learning_rate', LearningRates(lrIdx), ...
                        'max_depth', MaxDepths(depthIdx), ...
                        'gamma', Gammas(gammaIdx) ...
                    );

                    % Create an XGBoost model with the specified hyperparameters
                    mdl = xgboost(innerXTrain, innerYTrain, params);

                    % Predict EF values for the validation set
                    YPred = predict(mdl, innerXValidation);

                    % Calculate RMSE and MAE for the current parameter combination
                    rmse = sqrt(mean((YPred - innerYValidation).^2));
                    mae = mean(abs(YPred - innerYValidation));

                    innerRMSE(fold, lrIdx, depthIdx, gammaIdx) = rmse;
                    innerMAE(fold, lrIdx, depthIdx, gammaIdx) = mae;
                end
            end
        end
    end

    % Find the best hyperparameter combination (minimal RMSE) for XGBoost
    [bestFold, bestLRIndex, bestDepthIndex, bestGammaIndex] = ind2sub(size(innerRMSE), find(innerRMSE == min(innerRMSE(:)));

    % Train the XGBoost model with the best hyperparameters on the full training set
    bestLearningRate = LearningRates(bestLRIndex);
    bestMaxDepth = MaxDepths(bestDepthIndex);
    bestGamma = Gammas(bestGammaIndex);

    params = struct( ...
        'learning_rate', bestLearningRate, ...
        'max_depth', bestMaxDepth, ...
        'gamma', bestGamma ...
    );

    mdl = xgboost(X(trainIndices, :), Y(trainIndices), params);

    % Predict EF values for the test set
    YPred = predict(mdl, X(testIndices, :));

    % Calculate RMSE and MAE for the current test fold
    rmseValues(testIndex, bestLRIndex, bestDepthIndex, bestGammaIndex) = sqrt(mean((YPred - Y(testIndices)).^2));
    maeValues(testIndex, bestLRIndex, bestDepthIndex, bestGammaIndex) = mean(abs(YPred - Y(testIndices)));
end

% Calculate average RMSE and MAE across all observations
for lrIdx = 1:length(LearningRates)
    for depthIdx = 1:length(MaxDepths)
        for gammaIdx = 1:length(Gammas)
            avgRMSE(lrIdx, depthIdx, gammaIdx) = mean(squeeze(rmseValues(:, lrIdx, depthIdx, gammaIdx)));
            avgMAE(lrIdx, depthIdx, gammaIdx) = mean(squeeze(maeValues(:, lrIdx, depthIdx, gammaIdx)));
        end
    end
end

% Find the best hyperparameter combination based on minimal RMSE for XGBoost
[minRMSE, minRMSEIndex] = min(avgRMSE(:));
[bestLRIndex, bestDepthIndex, bestGammaIndex] = ind2sub(size(avgRMSE), minRMSEIndex);
bestLearningRate = LearningRates(bestLRIndex);
bestMaxDepth = MaxDepths(bestDepthIndex);
bestGamma = Gammas(bestGammaIndex);

fprintf('Best Hyperparameter Combination for XGBoost:\n');
fprintf('Learning Rate: %f\n', bestLearningRate);
fprintf('Max Depth: %d\n', bestMaxDepth);
fprintf('Gamma: %f\n', bestGamma);

% Display the average RMSE and MAE values for XGBoost
disp('Average RMSE for Each Hyperparameter Combination (XGBoost):');
disp(avgRMSE);
disp('Average MAE for Each Hyperparameter Combination (XGBoost):');
disp(avgMAE);

