function [s, fmax] = deb_sensitivity(X, dx, alpha, beta)
%DEB_SENSITIVITY
%	Computes the true maximum of f for any value of X and, consequently, the 
%	true sensitivity index. The true maximum was obtained "analytically" for 
%	each point, using at most a line search optimization in one of the 
%	variables.
%	This is used when comparing the different methods of estimating the 
%	sensitivity. Since the adapted version of the function is always greater 
%	than one, the relative version is used.
%
%	Syntax:
%		[s, fmax] = deb_sensitivity(X, dx, alpha, beta)
%
%	Input arguments:
%		X: a (n x mu) matrix with mu points and n dimensions each
%		dx: uncertainty in each variable. It can be a scalar (same dx for each
%		    coordinate), or a vector with individidual uncertainties.
%		alpha and beta: parameters of the deb_robust function.
%
%	Output arguments:
%		s: a (2 x mu) matrix with the sensitivity of mu points in each objective
%		fmax: (2 x mu) matrix with the maximum objective value in the
%		      neighborhood of each of the mu points
%
%	See also deb_robust, deb_range

if nargin < 4, alpha = 1; end
if nargin < 5, beta = 1; end

f = @(x) deb_robust(x, alpha, beta); %Deb's objective function
[n, mu] = size(X);
xrange = deb_range(n);
if isscalar(dx), dx = dx(ones(n,1),:); end
Fx = f(X);

s = zeros(2,mu); %initialize the sensibility

%% f1 = x1
% The maximum of f1 is always x1 + dx/2
f1max = Fx(1,:) + dx(1)/2;
s(1,:) = (f1max - Fx(1,:))./Fx(1,:);

%% f2
% Maximum of h: it is always in x1 - dx/2
h = @(x1) 1 - x1.^2;
hmax = h(X(1,:) - dx(1)/2);

% Maximum of S
S = @(x1) alpha./(0.2 + x1) + beta*x1.^2;
xmin = fminbnd(S, 0, 1); %get the minimum of this function
mask_lt = X(1,:) <= xmin; %where x1 is smaller than xmin
mask_gt = ~mask_lt; %X(1,:) > xmin; %where x1 is greater than xmin
Smax = zeros(1,mu);
Smax(mask_lt) = S(X(1,mask_lt) - dx(1)/2);
Smax(mask_gt) = S(X(1,mask_gt) + dx(1)/2);

% Maximum of g
gi = @(x) 10 + x.^2 - 10*cos(4*pi*x);
% This one is more intricate. Fortunately, it is symmetric over 0, so I first
% transform all positive numbers into negative
X2 = X(2:end,:); %get the second variable to n
X2(X2 > 0) = -X2(X2 > 0);
gmax = zeros(size(X2));

% First peak
mask1 = X2 <= -0.5;
xmax1 = -0.75095;
x2p1 = X2(mask1); %x2 of the first peak
gmax1 = gmax(mask1);

m1 = x2p1 <= xmax1 - dx(2)/2;
m2 = x2p1 >= xmax1 - dx(2)/2 & x2p1 <= xmax1 + dx(2)/2;
m3 = x2p1 >= xmax1 + dx(2)/2;

gmax1(m1) = gi(x2p1(m1) + dx(2)/2);
gmax1(m2) = 20.563;
gmax1(m3) = gi(x2p1(m3) - dx(2)/2);
gmax(mask1) = gmax1;

% Second peak
mask2 = X2 > -0.5;
xmax2 = -0.25032;
x2p2 = X2(mask2); %x2 of the first peak
gmax2 = gmax(mask2);

m1 = x2p2 <= xmax2 - dx(2)/2;
m2 = x2p2 >= xmax2 - dx(2)/2 & x2p2 <= xmax2 + dx(2)/2;
m3 = x2p2 >= xmax2 + dx(2)/2;

gmax2(m1) = gi(x2p2(m1) + dx(2)/2);
gmax2(m2) = 20.063;
gmax2(m3) = gi(x2p2(m3) - dx(2)/2);
gmax(mask2) = gmax2;

Gmax = sum(gmax,1); %in the end, sum the components of gmax
f2max = hmax + Gmax.*Smax + 1;

%% Correct the case when x1 > 0.66628
X1 = X(1,mask_gt);
gaux = Gmax(mask_gt);
f2aux = zeros(size(X1));
for ii = 1:size(X1,2)
	Faux = @(x) -(h(x) + gaux(ii).*S(x) + 1);
	xi = X1(ii);
	[a, b] = fminbnd(Faux, xi - dx(1)/2, xi + dx(1)/2);
	f2aux(ii) = -b;
end
f2max(mask_gt) = f2aux;

%% Take the maximum of f2 and compute the sensitivity
s(2,:) = (f2max - Fx(2,:))./Fx(2,:);
fmax = [f1max; f2max];

% Finally, penalizes the points closer to the boundaries
mask = any(X - dx/2 < xrange(:,1), 1) | any(X + dx/2 > xrange(:,2), 1);
s(:,mask) = 10^6;
