% This program uses the Radon transform to outline
% the distribution of linears'orientations from the
% river courses. The three largest alignments are
% drawn.
%
clc
clear -a
pkg load image;
theta = (0:179);
IM = imread('komana rivers.bmp');
P = imcrop(IM);
pause(10);
P = imcomplement(P);
width = columns(P)
height = rows(P)
figure, imshow (P, []), title ("Cropped image")
[R,xp]= radon (P, theta);
ratio = height/width;
k = zeros(180,1);
di = round(atan(ratio)*180/pi)
for c = 1:di
k(c,1) = 1/cos(c*pi/180);
endfor
for c = (di+1):90
k(c,1) =(ratio)/sin(c*pi/180);
endfor
for c = 91:180
k(c,1) = k(181-c);
endfor
k = k.';
for n = 1:rows(R)
R(n,:)= R(n,:)./k;
endfor
f = fspecial("average",5,5);
R = imfilter(R,f);

figure(3); surf( R, 'EdgeColor', 'none');axis ij;
title('3-D View of the Accumulation Array');
colorbar
[Row,Column,Value]= immaximas([R,xp],20,20000);
m = [Row,Column,Value];
[S, I] = sort (m,"descend");

P = imcomplement(P);
figure(1); imshow(P);
title(['komana rivers cropped image with main orientation marked'])
hold on;
diag = rows(R);

for i = 1:3;
  if (i == 1)
            paint = "red";
      elseif (i == 2)
            paint = "green";
      else
            paint = "blue";
   endif
hold on;
dist = m(I(i,3),1)
angle = m(I(i,3),2)
value = m(I(i,3),3)
abspt1 = width/2;
ordpt1 = height/2 +((diag/2 - dist)/sin((angle)*0.017453));
abspt2 = (width + height)/2;
ordpt2 = ordpt1 + ((height/2)* cot((angle)*0.017453));
X = [abspt1;abspt2];
Y = [ordpt1;ordpt2];
plot(X, Y, 'r+',i);
line (X, Y, "linewidth", 32, "color", paint);
endfor   
[R,xp]= radon (P, theta);
n = ones(height,width);
[N,xp] = radon (n, theta);
NOR = R./N;
NOR(~isfinite(NOR))=0;
[U,V] = [U,V]= max(sum(NOR) (1:90));
[U,180-V]
[M,N]= max(sum(NOR) (91:180));
[M,90-N]

figure(5);plot(theta, sum(NOR))
figure(6), imagesc(theta, xp, NOR); colormap(hot);
xlabel('\theta (degrees)'); ylabel('X\prime');
title('R_{\theta} (X\prime)');
colorbar
  
