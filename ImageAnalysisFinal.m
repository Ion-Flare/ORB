%% image toolbox version for comparison 
clear all;

image1 = imread('testImage.JPG');
image1 = rgb2gray(image1);

orbPoints =  detectORBFeatures(image1);

figure(1); image(image1); %View original image for visual comparison.
% hold on
% plot(selectUniform(orbPoints, 40, size(image1)));
% hold off
colormap(gray(256)); 
title Original-Image;
axis ij;        % left upper corner is the origin
set(gcf, 'Position',  [100, 100, 500, 500]) %controls image size

%% written code test 1 (testing the image with itself
clear all;

% Read images
image1 = imread('testImage.JPG');
image2 = imread('testImage.JPG');


[brief1, corner1] = OrbSearcher(image1); % find the ORB points of image 1
[brief2, corner2] = OrbSearcher(image2); % find the ORB points of image 2 

%find matches between the image 1 and 2
[matches,newimage]=OrbFeatureDetection(image1,image2,brief1,brief2,corner1,corner2); 
showresult(image1,image2,brief1,brief2,corner1,corner2,matches,newimage); %show images


%% written code test 2
clear all;

% Read images
image1 = imread('testImage.JPG');
image2 = imread('testImage_altered.JPG');

%make image 1 the same size as image 2
    [Y1,X1] = size(image1);
    [Y2,X2] = size(image2);
    image2 = imresize(image2,[Y1, X1]);
 figure(7)
 imshow(image2);    %display image 2

    
[brief1, corner1] = OrbSearcher(image1); % find the ORB points of image 1
[brief2, corner2] = OrbSearcher(image2); % find the ORB points of image 2 
 
%find matching orb points between the image 1 and 2
[matches,newimage]=OrbFeatureDetection(image1,image2,brief1,brief2,corner1,corner2); 
showresult(image1,image2,brief1,brief2,corner1,corner2,matches,newimage); %show images



%% written code Cameraman demo
clear all;

% Read images
image2 = imread('Cameraman.JPG');
image1 = imread('DarkerCameraMan.JPG');

image1 = imresize(image1,[256,320]);
image2= imresize(image2,[256,320]);

 figure(7)
 imshow(image2);    %display image 2
figure(8)
 imshow(image1);    %display image 1
    
[brief1, corner1] = OrbSearcher(image1); % find the ORB points of image 1
[brief2, corner2] = OrbSearcher(image2); % find the ORB points of image 2 
 
%find matching orb points between the image 1 and 2
[matches,newimage]=OrbFeatureDetection(image1,image2,brief1,brief2,corner1,corner2); 
showresult(image1,image2,brief1,brief2,corner1,corner2,matches,newimage); %show images


%% Fun Example

clear all;
% Read images
image1 = imread('Green.JPG');
image2 = imread('Green2.JPG');

image1 = imresize(image1,[254,250]);
image2= imresize(image2,[254,400]);


 figure(7)
 imshow(image2);    %display image 2
figure(8)
 imshow(image1);    %display image 1
    
[brief1, corner1] = OrbSearcher(image1); % find the ORB points of image 1
[brief2, corner2] = OrbSearcher(image2); % find the ORB points of image 2 
 
%find matching orb points between the image 1 and 2
[matches,newimage]=OrbFeatureDetection(image1,image2,brief1,brief2,corner1,corner2); 
showresult(image1,image2,brief1,brief2,corner1,corner2,matches,newimage); %show images



