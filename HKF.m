function location_resampled = HKF(location_original,resampling_ratio)
% location_original is the coordinates of the original point cloud, which
% is a N by 3 matrix. resampling_ratio is the resampling ratio, which is a
% number from 0 to 1. location_resampled is the coordinates of the
% resampled point cloud, which is a Nk by 3 matrix.

N = size(location_original,1); % Number of points in the original point cloud.
N_resampled = ceil(N*resampling_ratio); % Number of points in the resampled point cloud.
Nk = 27;
ptCloud_ori = pointCloud(location_original); % Original PointCloud

%% Estimate the intrinsic resolution
N1 = min(N,1e3); % Number of points used to estimate the intrinsic resolution
index_N1_rand = randperm(N,N1); % index of the randomly select N1 points in the original point cloud
Dis_sum = 0; % sum of the intrinsic resolution of selected N1 points
for i1 = 1:N1
    pointPosition = location_original(index_N1_rand(i1),:); % position of the i1-th selected point
    [~,dist1] = findNearestNeighbors(ptCloud_ori,pointPosition,2); % find the distance of the nearest neighbor of i1-th selected point in the original point cloud
    Dis_sum = Dis_sum + dist1(2); % since the closest point is the i1-th selected point itself, dist1(1)=0. What we need is dist1(2)
end
intrinsicResolution = Dis_sum/N1; % Estimation of the intrinsic resolution
%% Construct the hypergraph kernel
d = 1.3*intrinsicResolution*ones(1,3); % distance d is to define the kernel
% Since the estimation of the intrinsic resolution varys, a slightly larger
% value of distance d would help to get a more robust result. Also the
% value in different axis can be different if the kernel is not a cube.
kernel_Location = zeros(Nk,3); % Coordinates of the center of the voxel in each kernel, Nk = 27
for i = 1:3
    for j = 1:3
        for k = 1:3
            kernel_Location(9*(i-1)+3*(j-1)+k,:) = [(i-2)*d(1),(j-2)*d(2),(k-2)*d(3)];
        end
    end
end
%% Estimate the hypergraph spectrum basis V
kernel_Location_mean = sum(kernel_Location,2)/3;
kernel_Location_normalized = kernel_Location - kernel_Location_mean*ones(1,3); % Normalized signal of the coordinates of kernel
kernel_adjMat = kernel_Location_normalized*kernel_Location_normalized.'; % Adjency matrix of kernel
[V,lambda_kernel] = eig(kernel_adjMat); % hypergraph spectrum basis V and its corresponding eigenvalues
lambda_kernel = diag(lambda_kernel);
lambda_kernel = lambda_kernel/max(lambda_kernel); % Normalize the eigenvalues
lambda_kernel_diff = lambda_kernel(2:end) - lambda_kernel(1:end-1); % difference of the eigenvalues
[~,max_lambda_kernel_diff_index] = max(lambda_kernel_diff);
%% Calculate the indicators of each point in the original point cloud
indicators = zeros(N,1); % indicators
for i1 = 1:N
    pointi_position = location_original(i1,:); % position of the i1-th point in the original point cloud
    pointi_signal = zeros(Nk,1); % signal of point i1
    pointi_ROI = [pointi_position(1)-3*d(1)/2, pointi_position(1)+3*d(1)/2,...
 pointi_position(2)-3*d(2)/2, pointi_position(2)+3*d(2)/2, pointi_position(3)-3*d(3)/2, pointi_position(3)+3*d(3)/2]; % Region of interest of point i1
    pointi_neighbors = findPointsInROI(ptCloud_ori, pointi_ROI); % index of neighbors of point i1
    pointi_neighbors_position = location_original(pointi_neighbors,:); % position of neighbors of point i1
    for j = 1:Nk
        pointi_neighbors_voxel = find(abs(pointi_neighbors_position(:,1)-kernel_Location(j,1)-pointi_position(1))<d(1)/2 ...
        & abs(pointi_neighbors_position(:,2)-kernel_Location(j,2)-pointi_position(2))<d(2)/2 ...
        & abs(pointi_neighbors_position(:,3)-kernel_Location(j,3)-pointi_position(3))<d(3)/2);
        pointi_signal(j) = length(pointi_neighbors_voxel); % Number of points in each voxel
    end
    pointi_signal_transform = V'*pointi_signal;
    indicators(i1) = norm(pointi_signal_transform(1:max_lambda_kernel_diff_index),1)/norm(pointi_signal_transform,1);
end
[~,indicators_sort_index] = sort(indicators,'ascend');
location_resampled = location_original(indicators_sort_index(1:N_resampled),:);
end