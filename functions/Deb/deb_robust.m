function y = deb_robust(x, alpha, beta)
%DEB_ROBUST Implements function f1 from [1], and the test function used in [2]
%	The implementation here is the adapted one in [2], where all functions are
%	greater than one, so the relative sensitivity can be adopted.
%	
%	Syntax:
%		y = deb_robust(x)
%		y = deb_robust(x, alpha)
%		y = deb_robust(x, alpha, beta)
%
%	Input arguments:
%		x: a (n x mu) matrix with mu point of n dimensions each
%		alpha: a parameter that controls the function's sensitivity (def.: 1)
%		beta: another parameter as alpha (def.: 1)
%
%	Output argument:
%		y: a (2 x mu) matrix with the objective values of each of the mu points
%
%	References:
%	[1]  Kalyanmoy Deb and Himanshu Gupta. 2006. Introducing robustness in 
%	multi-objective optimization. Evolutionary Computation 14, 4 (Dec. 2006), 
%	463–494. DOI:http://dx.doi.org/10.1162/evco.2006.14.4.463
%	[2] Goulart, Fillipe, et al. "Robust multiobjective optimization using 
%	regression models and linear subproblems." Proceedings of the Genetic and 
%	Evolutionary Computation Conference. ACM, 2017.
%
%	See also deb_range, deb_sensitivity	

% Handle input
if nargin < 2
	alpha = 1;
end
if nargin < 3
	beta = 1;
end
	
% Auxiliary functions
x1 = x(1,:);
h = 1 - x1.^2;
g = sum(10 + x(2:end,:).^2 - 10*cos(4*pi*x(2:end,:)), 1);
S = alpha./(0.2 + x1) + beta*x1.^2;

% Objective functions
y(1,:) = x1;
y(2,:) = h + g.*S;
y = y + 1; %to prevent crossing
