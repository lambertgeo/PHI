% This program applies to a BMP image of a river network
% extracted from SRTM data. It finds circular features within 
% a range of radius and a minimum probability of occurrence (sigmas)
% and then plots the circles on the image.
clear all;clc;
pkg load image;

% INPUTS
UTM ='35S';
NWCORNER = [169185, 8902464];
SECORNER = [499987, 8749288];
dist = SECORNER - NWCORNER ;
FILE = 'katako kombe rivers.bmp'
radmin =60;
maxstep = 60;
sigmas = 6.0;
stepmul = 3;

%READ THE IMAGE
I = imread(FILE);
height = rows(I);
width = columns(I);
Xpixel = dist(1)/width;
Ypixel = dist(2)/height;
I = im2bw(I);
I = imcomplement(I);
results = [0,0,0,0,0,0];

%START THE SEARCH STEPWISE
for step = 1:maxstep
radius = radmin + step*stepmul;
 H = houghtf(I, "circle", radius);
 rim = int32( radius);
 f = fspecial("average",rim);
 M = imfilter(H,f);
 H = (H - M)/radius;
 CROP =[radius, radius,(width-2*radius),(height-2*radius)];
 RES = imcrop(H, CROP);
 cush = round(radius/10);
 
 sill = sigmas*std2(RES) + mean2(RES);
 [R,C,V] = immaximas(RES,cush,sill);
 m = [R,C,V];
 rad = radius*ones(rows(m),1);
 tip=   [rad, (m(:,1)+rad), (m(:,2)+rad), (NWCORNER(1)+(m(:,1)+rad)*Xpixel),...
 (NWCORNER(2)+(m(:,2)+rad)*Ypixel), ((m(:,3)-mean2(RES)) / std2(RES))];
 results = [tip; results];
endfor

 % Showing image and 3d plot of the Hough transform
 figure(2); imagesc(fltrdsum); colormap('jet'); axis image;
 figure(3);surf(fltrdsum, 'EdgeColor', 'none'); axis ij;
 

 
  % SHOWING THE BEST CIRCLES
 I = imread(FILE);
 %I = imcomplement(I);
 figure;
 imshow(I);
 title('katako kombe Raw image with circles detected, radii marked'])
 hold on;
 tresh1= 7
tresh2= 7.5
tresh3= 8

tag = ones(rows(results),1);
rescop = [tag, results];
for j = 1 : 5
  a = [rescop(:,3),rescop(:,4)];
  [neighbors distances] = kNearestNeighbors(a,a,2);
  for i =1 : rows(rescop)
    if (distances(i,2) < rescop(i,2)/3)
      if  (rescop(neighbors(i,1), 7)) < (rescop(neighbors(i,2), 7))
        rescop(i,1)= 0;
      endif
    endif
  endfor
  
  for i = 1:rows(rescop) 
    if rescop(i,1)!= 0 
     copie = [rescop(i,:); copie];
    endif
  endfor
 rescop = copie;
 copie =[];
endfor

radius = rescop(:,2);

 R = rescop(:,3);
 C = rescop(:,4);
 V = rescop(:,7);
 for k = 1 : size((R)-1),
  plot(C(k), R(k), 'r+');
  if((V(k) >tresh2)&(V(k)<tresh3))
        DrawCircle( C(k),R(k),radius(k),32,'g-');
      elseif((V(k) >tresh1)&(V(k)<tresh2))
        DrawCircle( C(k),R(k),radius(k),32,'b-');
      elseif (V(k) >tresh3)
        DrawCircle( C(k),R(k),radius(k),32,'r-'); 
       else
      endif
  endfor
