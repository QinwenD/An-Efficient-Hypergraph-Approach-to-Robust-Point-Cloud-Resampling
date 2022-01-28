function location_resampled = LHF(location_original,resampling_ratio,Na,Nb)
% location_original is the coordinates of the original point cloud, which
% is a N by 3 matrix. resampling_ratio is the resampling ratio, which is a
% number from 0 to 1. location_resampled is the coordinates of the
% resampled point cloud, which is a Nk by 3 matrix. Na and Nb are different
% local lengths
%
N = size(location_original,1); % Number of points in the original point cloud.
N_resampled = ceil(N*resampling_ratio); % Number of points in the resampled point cloud.
ptCloud_ori = pointCloud(location_original); % Original PointCloud


%% Calculate the indicators of each point in the original point cloud
indicators_Na = zeros(N,1); % indicators corresponding to local length Na
indicators_Nb = zeros(N,1); % indicators corresponding to local length Nb
for i1 = 1:N
    pointi_position = location_original(i1,:); % position of the i1-th point in the original point cloud
    [pointi_neighbors_index_Na,~] = findNearestNeighbors(ptCloud_ori,pointi_position,Na); % find the index of the neighbors of i1-th point in the original point cloud
    [pointi_neighbors_index_Nb,~] = findNearestNeighbors(ptCloud_ori,pointi_position,Nb);
    
    if ~isempty(pointi_neighbors_index_Na)
        pointi_signal_Na = location_original(pointi_neighbors_index_Na,:); % Use the relative 3D coordinates as the signal
        pointi_signal_Na = pointi_signal_Na - ones(size(pointi_signal_Na,1),1)*pointi_position;
    end
    if ~isempty(pointi_neighbors_index_Nb)
        pointi_signal_Nb = location_original(pointi_neighbors_index_Nb,:); % Use the relative 3D coordinates as the signal
        pointi_signal_Nb = pointi_signal_Nb - ones(size(pointi_signal_Nb,1),1)*pointi_position;
    end
    
    pointi_signal_Na_normalize = pointi_signal_Na - sum(pointi_signal_Na,2)/3*ones(1,3); % Normalize the signals
    pointi_signal_Nb_normalize = pointi_signal_Nb - sum(pointi_signal_Nb,2)/3*ones(1,3);
    
    pointi_RMat_Na = pointi_signal_Na_normalize*pointi_signal_Na_normalize';
    pointi_RMat_Nb = pointi_signal_Nb_normalize*pointi_signal_Nb_normalize';
    
    [V_Na,Lambda_Na] = eig(pointi_RMat_Na); % Estimate the hypergraph spectrum basis
    [V_Nb,Lambda_Nb] = eig(pointi_RMat_Nb);
    
    Lambda_Na = abs(diag(Lambda_Na));
    Lambda_Nb = abs(diag(Lambda_Nb));

    Lambda_Na = Lambda_Na/max(Lambda_Na); % Normalize
    Lambda_Nb = Lambda_Nb/max(Lambda_Nb);
    
    Lambda_Na_diff = Lambda_Na(2:end) - Lambda_Na(1:end-1);
    Lambda_Nb_diff = Lambda_Nb(2:end) - Lambda_Nb(1:end-1);
    
    [~,max_lambda_Na_diff_index] = max(Lambda_Na_diff);
    [~,max_lambda_Nb_diff_index] = max(Lambda_Nb_diff);
    
    pointi_signal_Na_transform = V_Na\pointi_signal_Na;
    pointi_signal_Nb_transform = V_Nb\pointi_signal_Nb;
    
    indicators_Na(i1) = sum(abs(pointi_signal_Na_transform(1:max_lambda_Na_diff_index,:)),'all')/sum(abs(pointi_signal_Na_transform),'all');
    indicators_Nb(i1) = sum(abs(pointi_signal_Nb_transform(1:max_lambda_Nb_diff_index,:)),'all')/sum(abs(pointi_signal_Nb_transform),'all');
end
epsilon1 = indicators_Nb(N_resampled)/(indicators_Na(N_resampled)+indicators_Nb(N_resampled));
[~,indicators_sort_index] = sort(epsilon1*indicators_Na+(1-epsilon1)*indicators_Nb,'descend');
location_resampled = location_original(indicators_sort_index(1:N_resampled),:);
end