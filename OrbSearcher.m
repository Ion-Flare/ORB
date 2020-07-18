%Ian Hudis 
%12/15/19
function [Orbpoints, cornerID] = OrbSearcher(InputImage)
%This Looks searches for orb points using fast orientation and rotated
%briefs.

%converts the image to greyscale
[~, ~, numberOfColorChannels] = size(InputImage); %checks to see if image is greyscale
if numberOfColorChannels > 1
GreyInputImage = rgb2gray(InputImage);
else
    GreyInputImage=InputImage;
end  


% extract FAST corners and its score
[corners, score] = FindFast(GreyInputImage,20, 1); %takes scores and corners
[corners,~] = FAST(GreyInputImage,corners,score); %uses scores to perform fast

% compute Harris corner score
HarrisInput = harris(GreyInputImage);
harris1 = HarrisInput(sub2ind(size(HarrisInput),corners(:,2),corners(:,1)));

% refine FAST corners with harris scores
[~,Index] = sort(harris1);
cornerID = corners(Index(1:size(Index)),:);

% get orientations (angles) for the selected points 
angles = orientation(GreyInputImage,[cornerID(:,2),cornerID(:,1)]);

% compute rotational BRIEF  
%run('sampling_param.m')
sample=briefpoints;
Orbpoints = rotatedBrief(GreyInputImage,cornerID,sample,angles);
end


% function for doing the Orientated Fast
function [NonMaxFastValues,fastScale] = FAST(InputImage,corners,fscore)

% image size
[y,~]=size(InputImage);

% get vectorial indices for corners
pixel = zeros(size(corners,1),1);
for n = 1:size(corners,1)
    pixel(n) = (corners(n,1)-1)*y + corners(n,2); 
end

% get indicies for surrounding pixels
KernelSize = 5;
kernel = KernelSize*2+1; 
KernelVector = zeros(kernel^2,1);
%redistribute kernel
for i = 1:kernel
    for j = 1:kernel
        KernelVector((i-1)*kernel + j,1) = j-6 + (i-6)*y;
    end
end

% create fast score map
ScoreMap = zeros(size(InputImage));
ScoreMap(pixel) = fscore;

% initiate non maximum suppression 
CronerMax = zeros(size(corners,1),1);

for i = 1:size(corners,1)
    surround = pixel(i) + KernelVector;
    CronerMax(i) = (sum(fscore(i) >= ScoreMap(surround)) == kernel^2);    
end

NonMaxFastValues = corners(logical(CronerMax),:);
fastScale = fscore(logical(CronerMax));
end

function features = rotatedBrief(InputImage,corners,patterns,angle)

% initialise features
features = zeros(size(corners,1),256);

for a = 1:size(corners,1)
    Rotation1 = round( [ cos(angle(a)), -sin(angle(a));sin(angle(a)),cos(angle(a))] * patterns(:,1:2)')'; 
    Rotation2 = round( [ cos(angle(a)), -sin(angle(a));sin(angle(a)),cos(angle(a))] * patterns(:,3:4)')';
    for b = 1:256
        p1 = InputImage(corners(a,2) +  Rotation1 (a,2),corners(a,1) +  Rotation1 (b,1));
        p2 = InputImage(corners(a,2) + Rotation2(b,2),corners(a,1) + Rotation2(b,1));
        features(a,b) = double(p1 < p2);
    end
end
end


% compute harris corner score
function H = harris(InputImage)
InputImage = double(InputImage);
% parameters 
sig=2; r = 6;
% gradient kernel
dx = [-1 0 1; -1 0 1; -1 0 1]; % The Mask 
dy = dx';

% main
Ix = conv2(InputImage, dx, 'same');   
Iy = conv2(InputImage, dy, 'same');
g = fspecial('gaussian',max(1,fix(6*sig)), sig); % Gaussian Filter

Ix2 = conv2(Ix.^2, g, 'same');  
Iy2 = conv2(Iy.^2, g, 'same');
Ixy = conv2(Ix.*Iy, g,'same');

k = 0.04;
Hp = (Ix2.*Iy2 - Ixy.^2) - k*(Ix2 + Iy2).^2;

H = (10000/max(abs(Hp(:))))*Hp;

% set edges zeros 
H(1:r,:) = 0;
H(:,1:r) = 0;
H(end-r+1:end,:) = 0;
H(:,end-r+1:end) = 0;
end


% compute angles corresponding to the selected points
function angle = orientation(InputImage,coordinates)
% transpose to compute angle
radious = 3; % FAST corner detector's default radious
InputImage = double(InputImage);  % converts column major to row major for efficency.
m = strel('octagon',radious); mask = m.Neighborhood; % FAST search mask
Ip = padarray(InputImage,[radious radious],0,'both'); % padding

r = size(mask,2);
c = size(mask,1);

angle = zeros(size(coordinates,1),1);

for i = 1:size(coordinates,1)
    
    vert = 0;
    horz = 0;
    
    Rinitial = coordinates(i,2); %initializes corners
    Cinitial = coordinates(i,1);
    
    for j= 1:r
        for k = 1:c
            if mask(k,j) 
               pixel = Ip(Cinitial + k-1, Rinitial + j-1);
               vert = vert + pixel * (-radious + k - 1);
               horz = horz + pixel * (-radious + j - 1);
            end
        end
    end
    angle(i) = atan2(vert,horz);
end
end



% predefined sampling patterns (from opencv) 
% For output consistancy -> like a seed function
function sample = briefpoints()

sample = [ 8,-3, 9,5;
    4,2, 7,-12;
    -11,9, -8,2;
    7,-12, 12,-13;
    2,-13, 2,12;
    1,-7, 1,6;
    -2,-10, -2,-4;
    -13,-13, -11,-8;
    -13,-3, -12,-9;
    10,4, 11,9;
    -13,-8, -8,-9;
    -11,7, -9,12;
    7,7, 12,6;
    -4,-5, -3,0;
    -13,2, -12,-3;
    -9,0, -7,5;
    12,-6, 12,-1;
    -3,6, -2,12;
    -6,-13, -4,-8 ;
    11,-13, 12,-8 ;
    4,7, 5,1 ;
    5,-3, 10,-3 ;
    3,-7, 6,12 ;
    -8,-7, -6,-2 ;
    -2,11, -1,-10 ;
    -13,12, -8,10 ;
    -7,3, -5,-3 ;
    -4,2, -3,7 ;
    -10,-12, -6,11 ;
    5,-12, 6,-7 ;
    5,-6, 7,-1 ;
    1,0, 4,-5 ;
    9,11, 11,-13 ;
    4,7, 4,12 ;
    2,-1, 4,4 ;
    -4,-12, -2,7 ;
    -8,-5, -7,-10 ;
    4,11, 9,12 ;
    0,-8, 1,-13 ;
    -13,-2, -8,2 ;
    -3,-2, -2,3 ;
    -6,9, -4,-9 ;
    8,12, 10,7 ;
    0,9, 1,3 ;
    7,-5, 11,-10 ;
    -13,-6, -11,0 ;
    10,7, 12,1 ;
    -6,-3, -6,12 ;
    10,-9, 12,-4 ;
    -13,8, -8,-12 ;
    -13,0, -8,-4 ;
    3,3, 7,8 ;
    5,7, 10,-7 ;
    -1,7, 1,-12 ;
    3,-10, 5,6 ;
    2,-4, 3,-10 ;
    -13,0, -13,5 ;
    -13,-7, -12,12 ;
    -13,3, -11,8 ;
    -7,12, -4,7 ;
    6,-10, 12,8 ;
    -9,-1, -7,-6 ;
    -2,-5, 0,12 ;
    -12,5, -7,5 ;
    3,-10, 8,-13 ;
    -7,-7, -4,5 ;
    -3,-2, -1,-7 ;
    2,9, 5,-11 ;
    -11,-13, -5,-13 ;
    -1,6, 0,-1 ;
    5,-3, 5,2 ;
    -4,-13, -4,12 ;
    -9,-6, -9,6 ;
    -12,-10, -8,-4 ;
    10,2, 12,-3 ;
    7,12, 12,12 ;
    -7,-13, -6,5 ;
    -4,9, -3,4 ;
    7,-1, 12,2 ;
    -7,6, -5,1 ;
    -13,11, -12,5 ;
    -3,7, -2,-6 ;
    7,-8, 12,-7 ;
    -13,-7, -11,-12 ;
    1,-3, 12,12 ;
    2,-6, 3,0 ;
    -4,3, -2,-13 ;
    -1,-13, 1,9 ;
    7,1, 8,-6 ;
    1,-1, 3,12 ;
    9,1, 12,6 ;
    -1,-9, -1,3 ;
    -13,-13, -10,5 ;
    7,7, 10,12 ;
    12,-5, 12,9 ;
    6,3, 7,11 ;
    5,-13, 6,10 ;
    2,-12, 2,3 ;
    3,8, 4,-6 ;
    2,6, 12,-13 ;
    9,-12, 10,3 ;
    -8,4, -7,9 ;
    -11,12, -4,-6 ;
    1,12, 2,-8 ;
    6,-9, 7,-4 ;
    2,3, 3,-2 ;
    6,3, 11,0 ;
    3,-3, 8,-8 ;
    7,8, 9,3 ;
    -11,-5, -6,-4 ;
    -10,11, -5,10 ;
    -5,-8, -3,12 ;
    -10,5, -9,0 ;
    8,-1, 12,-6 ;
    4,-6, 6,-11 ;
    -10,12, -8,7 ;
    4,-2, 6,7 ;
    -2,0, -2,12 ;
    -5,-8, -5,2 ;
    7,-6, 10,12 ;
    -9,-13, -8,-8 ;
    -5,-13, -5,-2 ;
    8,-8, 9,-13 ;
    -9,-11, -9,0 ;
    1,-8, 1,-2 ;
    7,-4, 9,1 ;
    -2,1, -1,-4 ;
    11,-6, 12,-11 ;
    -12,-9, -6,4 ;
    3,7, 7,12 ;
    5,5, 10,8 ;
    0,-4, 2,8 ;
    -9,12, -5,-13 ;
    0,7, 2,12 ;
    -1,2, 1,7 ;
    5,11, 7,-9 ;
    3,5, 6,-8 ;
    -13,-4, -8,9 ;
    -5,9, -3,-3 ;
    -4,-7, -3,-12 ;
    6,5, 8,0 ;
    -7,6, -6,12 ;
    -13,6, -5,-2 ;
    1,-10, 3,10 ;
    4,1, 8,-4 ;
    -2,-2, 2,-13 ;
    2,-12, 12,12 ;
    -2,-13, 0,-6 ;
    4,1, 9,3 ;
    -6,-10, -3,-5 ;
    -3,-13, -1,1 ;
    7,5, 12,-11 ;
    4,-2, 5,-7 ;
    -13,9, -9,-5 ;
    7,1, 8,6 ;
    7,-8, 7,6 ;
    -7,-4, -7,1 ;
    -8,11, -7,-8 ;
    -13,6, -12,-8 ;
    2,4, 3,9 ;
    10,-5, 12,3 ;
    -6,-5, -6,7 ;
    8,-3, 9,-8 ;
    2,-12, 2,8 ;
    -11,-2, -10,3 ;
    -12,-13, -7,-9 ;
    -11,0, -10,-5 ;
    5,-3, 11,8 ;
    -2,-13, -1,12 ;
    -1,-8, 0,9 ;
    -13,-11, -12,-5 ;
    -10,-2, -10,11 ;
    -3,9, -2,-13 ;
    2,-3, 3,2 ;
    -9,-13, -4,0 ;
    -4,6, -3,-10 ;
    -4,12, -2,-7 ;
    -6,-11, -4,9 ;
    6,-3, 6,11 ;
    -13,11, -5,5 ;
    11,11, 12,6 ;
    7,-5, 12,-2 ;
    -1,12, 0,7 ;
    -4,-8, -3,-2 ;
    -7,1, -6,7 ;
    -13,-12, -8,-13 ;
    -7,-2, -6,-8 ;
    -8,5, -6,-9 ;
    -5,-1, -4,5 ;
    -13,7, -8,10 ;
    1,5, 5,-13 ;
    1,0, 10,-13 ;
    9,12, 10,-1 ;
    5,-8, 10,-9 ;
    -1,11, 1,-13 ;
    -9,-3, -6,2 ;
    -1,-10, 1,12 ;
    -13,1, -8,-10 ;
    8,-11, 10,-6 ;
    2,-13, 3,-6 ;
    7,-13, 12,-9 ;
    -10,-10, -5,-7 ;
    -10,-8, -8,-13 ;
    4,-6, 8,5 ;
    3,12, 8,-13 ;
    -4,2, -3,-3 ;
    5,-13, 10,-12 ;
    4,-13, 5,-1 ;
    -9,9, -4,3 ;
    0,3, 3,-9 ;
    -12,1, -6,1 ;
    3,2, 4,-8 ;
    -10,-10, -10,9 ;
    8,-13, 12,12 ;
    -8,-12, -6,-5 ;
    2,2, 3,7 ;
    10,6, 11,-8 ;
    6,8, 8,-12 ;
    -7,10, -6,5 ;
    -3,-9, -3,9 ;
    -1,-13, -1,5 ;
    -3,-7, -3,4 ;
    -8,-2, -8,3 ;
    4,2, 12,12 ;
    2,-5, 3,11 ;
    6,-9, 11,-13 ;
    3,-1, 7,12 ;
    11,-1, 12,4 ;
    -3,0, -3,6 ;
    4,-11, 4,12 ;
    2,-4, 2,1 ;
    -10,-6, -8,1 ;
    -13,7, -11,1 ;
    -13,12, -11,-13 ;
    6,0, 11,-13 ;
    0,-1, 1,4 ;
    -13,3, -9,-2 ;
    -9,8, -6,-3 ;
    -13,-6, -8,-2 ;
    5,-9, 8,10 ;
    2,7, 3,-9 ;
    -1,-6, -1,-1 ;
    9,5, 11,-2 ;
    11,-3, 12,-8 ;
    3,0, 3,5 ;
    -1,4, 0,10 ;
    3,-6, 4,5 ;
    -13,0, -10,5 ;
    5,8, 12,11 ;
    8,9, 9,-6 ;
    7,-4, 8,-12 ;
    -10,4, -10,9 ;
    7,3, 12,4 ;
    9,-7, 10,-2 ;
    7,0, 12,-2 ;
    -1,-6, 0,-11 ;
];
end



