function output = showresult(image1,image2,feature1,feature2,corner1,corner2,matches,newimage)
%this was a program for printing test results quicker

figure(1);
imshow(image1) %show image1
set(gcf, 'Position',  [100, 100, 256, 256]) %controls image size
hold on;plot(corner1(:,1),corner1(:,2),'r*')  %show image 1 orb points
title('What we Are Looking For');
figure(2);
imshow(image2) %show image2
set(gcf, 'Position',  [100, 100, 256, 256]) %controls image size
hold on;plot(corner2(:,1),corner2(:,2),'r*') %show image 2 orb points
title('What we Are Looking Into');
figure(3)

newImg = cat(2,image2,image1);
imshow(newImg)
hold on
plot(feature2(:,1),feature2(:,2), 'g.')
plot(feature1(:,1)+size(image2,2),feature1(:,2), 'r.')
for i = 1:size(matches,1)
    plot([matches(i,3) matches(i,1)+size(image2,2)],[matches(i,4) matches(i,2)],'b')
end
title('The Matched feature between the two images')

output = image1;

figure(4); % warped image
imshow(newimage);

end