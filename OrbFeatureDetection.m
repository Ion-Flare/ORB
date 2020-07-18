function [matches,warpedImage] = OrbFeatureDetection(InputImage1,InputImage2,brief1, brief2,corner1 , corner2)
% Matlab operation to find matches 
[index1,dist1]= knnsearch(brief2,brief1,'K',2,'Distance','hamming'); % knn distance for descriptor 1
[index2,~]= knnsearch(brief1,brief2,'K',2,'Distance','hamming'); % knn distance for descriptor 2

matches = zeros(size(brief1,1),4);
for i = 1:size(brief1,1)
    if (dist1(i,1) <= 64/256 && dist1(i,1)/dist1(i,2) <=0.98 && i == index2(index1(i),1))
        % hamming distance < 0.25 + ratio between smallest and second
        % smallest < 0.98  and cross minimum value check 
        matches(i,:) = [corner1(i,:),corner2(index1(i,1),:)];
    end
end
% matches found 

% remove outliers with homomorphic filtering 
filter = find(matches(:,1));
matches = matches(filter,:);

% matched points (note that there are a number of outliers since not all points will match.)
feature1 = matches(:,1:2);
feature2 = matches(:,3:4);

% RANdom SAmple Consensus (RANSAC) c.f. it is a necessary step to remove outliers!
[H,inlr] = computeHomography(feature1,feature2,3,1000);


Hp = H';  % since Matlab transposed x and y;
tform = projective2d(Hp);
%warps image so that the two input images are the same size
out = imwarp(InputImage1,tform,'OutputView', imref2d(size(InputImage2)));
out = uint8((double(out) + double(InputImage2))./2); 

warpedImage=out; % "this is the threaded image for demonstration purposes"

end



%Homography kit from github 
function [H,inlr_loc] = computeHomography(feature1,feature2, thres, maxiter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Estimate Homography matrix using RANdom SAmple Consensus &
%   Levenberg-Marquardt method
%   inputs - feature1: matched points in frame 1 
%            feature2: matched points in frame 2
%            thres   : inlier threshold (default 3)
%            maxiter : max itermation (default 1000)
%
%   output - H: Homography matrix
%
%                    Juheon Lee (21/06/2018) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Random seed
rng(1113732)

pval = 0.99; % confidence parameter for dynamic update
isnormal = 1; % direct linear transform is commited with normalisation 
cinlr = 0; % the number of inliers at current iteration
inlr_loc = []; % index of inliers

for i = 1:maxiter % dynamic iteration
    
    % select 4 random points
    pts = randperm(size(feature1,1),4); % random permutation of N integers
    pts1 = feature1(pts,:);
    pts2 = feature2(pts,:);
    
    % check geometric constraint of the selected points
    T = GeometricTest(pts1,pts2);
    if (T == 0); continue; end % random points does not satisfy geometric constraints
    
    % direct linear transform for estimating homography matrix
    Htmp = homography(pts1,pts2,0);

    % calculate reprojection error
    [~,loc] = compute_reproj_error(feature1,feature2,Htmp,thres);
    
    % dynamic update 
    if length(loc) > cinlr
        cinlr = length(loc); % update the number of inliers
        H = Htmp; % update homography matrix
        inlr_loc = loc;
        maxiter = ransacIter(pval,cinlr,maxiter); % update iteration number (maximum 1000)
    end
end
H = H;
% optional Levenberg-Marquardt optimisation 

end

function iter = ransacIter(p,ninlr, maxiter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dynamic Update for RANSAC
%
%   Inputs - p : confidence level (default 0.99);
%            ninlr : number of inliers
%            maxiter :
%
%
%   output - iter : maximum iteration
%
%
%                   Juheon Lee (21/06/2018)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (p >1 || p <0)
    error('p should be larger than 0 and smaller than 1');
end

ep = 1-p;
tmp = 1 - (1 - ep)^ninlr;

num =  log(ep);
denum = log(tmp);

if (num/denum > maxiter)
    iter = maxiter;
else
    iter = round(num/denum);
end
end

function T = GeometricTest(feature1,feature2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Check geometric consistency of randomly selected points (it helps to
%   accelerate ransac algorithm
%   
%   Inputs - Feature1 : 4 randomly chosen source points
%            Feature2 : 4 randomly chosen destination points
%
%   Outputs- T : logical indicator that 4 randomly selected points are
%   geometrically consistent. 
%
%   Ref. Marquez-Neila et al. "Speeding-up homography estimation in mobile devices"
%                           Juheon Lee  (21/06/2018)
%                       Feel free to redistribute!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if size(feature1,1) ~= 4 || size(feature2,1) ~= 4
    error('The size of each point set must be 4')
end

M = [1 2 3; 1 2 4; 1 3 4; 2 3 4]; % all possible combinations
v = zeros(4,1); % indication vector

for i = 1:4
    A = [feature1(M(i,1),1),feature1(M(i,1),2),1; ...
         feature1(M(i,2),1),feature1(M(i,2),2),1; ...
         feature1(M(i,3),1),feature1(M(i,3),2),1;];
    B = [feature2(M(i,1),1),feature2(M(i,1),2),1; ...
         feature2(M(i,2),1),feature2(M(i,2),2),1; ...
         feature2(M(i,3),1),feature2(M(i,3),2),1;];
    v(i) = det(A)*det(B)<0;
end

if (sum(v) == 0 || sum(v) == 4)
    T = true;
else
    T = false;
end
end

function [val,loc] = compute_reproj_error(pts1,pts2,H,thres)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Compute Homography reprojection error
% 
%   Inputs - pts1 : points from frame 1
%            pts2 : corresponding points in frame 2
%            H : Homography matrix
%            thres: threshold for inliers
%
%   Output- val : mean reprojection error in L2 distance (euclidean metric)
%           loc : index for inliears
%
%                       Juheon Lee (21/06/2018)
%               Feel Free to redistribute! 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin <4
   thres = 3;
end

leng = size(pts1,1);
val = zeros(leng,1);

for i = 1:leng
    % compute reprojection 
    xp = (pts1(i,1)* H(1,1)  + pts1(i,2) * H(1,2) + H(1,3))/(pts1(i,1) * H(3,1) + pts1(i,2) * H(3,2) + H(3,3));
    yp = (pts1(i,1)* H(2,1)  + pts1(i,2) * H(2,2) + H(2,3))/(pts1(i,1) * H(3,1) + pts1(i,2) * H(3,2) + H(3,3));
    % compute euclidean reprojection error
    val(i) = sqrt((xp - pts2(i,1))^2 + (yp - pts2(i,2))^2); 
end

loc = find(val < thres);
end

function H = homography(pts1, pts2, isnormal)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Estimate homography matrix H using direct linear transform (4-point
%   method)
%
%   Inputs - pts1: points from Frame 1
%            pts2: points from Frame 2
%            isnormal: apply normalisation (default 1)  
%
%   Output - H: Homography matrix
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 3
    isnormal = 1;
end
A= [];
if isnormal == 0 
    X1 = pts1(:,1); Y1 = pts1(:,2);
    X2 = pts2(:,1); Y2 = pts2(:,2);
    % create a matrix
    for i = 1:size(pts1,1)
        tmp = [X1(i), Y1(i), 1, 0, 0, 0, -X2(i)*X1(i), -X2(i)*Y1(i), -X2(i); ...
               0, 0, 0, X1(i), Y1(i), 1, -Y2(i)*X1(i), -Y2(i)*Y1(i), -Y2(i);];
        A = [A;tmp];
    end
        %{
        X1(2), Y1(2), 1, 0, 0, 0, -X2(2)*X1(2), -X2(2)*Y1(2), -X2(2); ...
        0, 0, 0, X1(2), Y1(2), 1, -Y2(2)*X1(2), -Y2(2)*Y1(2), -Y2(2); ...
        X1(3), Y1(3), 1, 0, 0, 0, -X2(3)*X1(3), -X2(3)*Y1(3), -X2(3); ...
        0, 0, 0, X1(3), Y1(3), 1, -Y2(3)*X1(3), -Y2(3)*Y1(3), -Y2(3); ...
        X1(4), Y1(4), 1, 0, 0, 0, -X2(4)*X1(4), -X2(4)*Y1(4), -X2(4); ...
        0, 0, 0, X1(4), Y1(4), 1, -Y2(4)*X1(4), -Y2(4)*Y1(4), -Y2(4)];
        %}
    [s,v,d] = svd(A);
    Hp = d(:,9); % smallest eigenvector is the solution
    H = reshape(Hp,3,3)'/Hp(9); % normalised w.r.t. the last component i.e. H(3,3) = 1;
else 
   % get mean
   cx = 0; cy = 0;
   cX = 0; cY = 0;
   for i = 1:4
       cx = cx + pts1(i,1);
       cy = cy + pts1(i,2);
       cX = cX + pts2(i,1);
       cY = cY + pts2(i,2);
   end
   
   cx = cx/4; cy = cy/4;
   cX = cX/4; cY = cy/4;
   
   % scale
   sx = 0; sy = 0;
   sX = 0; sY = 0;
   
   for j = 1:4
       sx = sx + abs(pts1(j,1) - cx);
       sy = sy + abs(pts1(j,2) - cy);
       sX = sX + abs(pts2(j,1) - cX);
       sY = sY + abs(pts2(j,2) - cY);
   end
   
   sx = 4/sx; sy = 4/sy;
   sX = 4/sX; sY = 4/sY;
   
   % H norm
   InvHnorm = [1./sX, 0, cX; 0, 1./sY, cY;0, 0, 1];
   Hnorm = [sx, 0, -cx*sx; 0, sy, -cy*sy; 0, 0, 1];
   
   % contruct matrix 
   X1 = (pts1(:,1) - cx)*sx; Y1 = (pts1(:,2) - cy)*sy;
   X2 = (pts2(:,1) - cX)*sX; Y2 = (pts2(:,2) - cY)*sY;
   
   A = [X1(1), Y1(1), 1, 0, 0, 0, -X2(1)*X1(1), -X2(1)*Y1(1), -X2(1); ...
        0, 0, 0, X1(1), Y1(1), 1, -Y2(1)*X1(1), -Y2(1)*Y1(1), -Y2(1); ...
        X1(2), Y1(2), 1, 0, 0, 0, -X2(2)*X1(2), -X2(2)*Y1(2), -X2(2); ...
        0, 0, 0, X1(2), Y1(2), 1, -Y2(2)*X1(2), -Y2(2)*Y1(2), -Y2(2); ...
        X1(3), Y1(3), 1, 0, 0, 0, -X2(3)*X1(3), -X2(3)*Y1(3), -X2(3); ...
        0, 0, 0, X1(3), Y1(3), 1, -Y2(3)*X1(3), -Y2(3)*Y1(3), -Y2(3); ...
        X1(4), Y1(4), 1, 0, 0, 0, -X2(4)*X1(4), -X2(4)*Y1(4), -X2(4); ...
        0, 0, 0, X1(4), Y1(4), 1, -Y2(4)*X1(4), -Y2(4)*Y1(4), -Y2(4)];
    [s,v,d] = svd(A);
    Hp = d(:,9);
    tmp = reshape(Hp,3,3)';
    Htmp = InvHnorm*tmp*Hnorm;
    H = Htmp/Htmp(3,3);
end
end
