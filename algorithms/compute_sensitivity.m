function P = compute_sensitivity(f, P, xrange, options)
%COMPUTE_SENSITIVITY Compute the sensitivity of each point in P
%	The sensitivity of a solution x is given by
%
%		s_abs = |f^{eff}(x) - f(x)|
%		
%			or
%		
%		   		 |f^{eff}(x) - f(x)|
%		s_rel = ---------------------
%		               |f(x)|
%	
%	wherein f^{eff}(x) is the effective function of f(x), which, in this case,
%	is the worst objective value in the neighborhood 
%	
%		N(x) = {x'| x - dx/2 <= x' <= x + dx/2}
%
%	in which dx is a n x 1 vector of uncertainties. s_rel is the typical method,
%	but if f(x) is not guaranteed to be non-zero over all feasible space, it is
%	best to use the absolute version s_abs.
%
%	Computing the maximum of f(x) in a given interval requires another 
%	optimization problem, which is very costly. The two approaches used here are
%
%	- "sample": sample a number Nneigh of neighbors in the interval 
%	[x - dx/2, x + dx/2], and get the worst value. This is simple but requires
%	a lot of neighbors to be reliable, and thus, many objective evaluations;
%	- "linear": the proposed method, which again samples a number of neighbors
%	but uses a linear regression of them, and the maximum will be the solution
%	of the equivalent linear program. This requires way less samples.
%	We set Nneigh = n x neigh, where neigh is a multiplier of the dimension 
%	number.
%	
%	The steps followed in the linear approximation are:
%	For each point in P, compute its sensitivity. It consists on the steps:
%	For each x in P,
%		1- generate n*neigh points in x +- dx/2. n is the dimension of the 
%		   problem and neigh is the multiplier;
%		2- Perform a linear regression in the points in this neighborhood;
%		3- Compute the maximum value of f, fmax, in this region by solving a 
%		   linear programming;
%		4- Calculate the sensitivity of x using
%			s = abs(fmax - fx),         if using absolute differences, or
%			s = abs(fmax - fx)/abs(fx), if using percentages
%       This is performed for EACH objective. Notice that, in the latter 
%       case, fx must not be zero, or else the percentage difference will not
%       make sense.
%
%	Syntax:
%		P = compute_sensitivity(f, P, xrange, options)
%
%	Input arguments:
%		f: the objective function
%		P: a struct with the population, with P.x a (n x mu) matrix with mu 
%		   individuals with n variables each, and P.f a (m x mu) matrix with 
%		   the objective values in each of the m functions
%		xrange: a (n x 2) matrix with the limits of the search space
%		options: the struct with the parameters of the algorithm. 
%		         See demo_robust documentation.
%
%	Output argument:
%		P: the population increased with the field s, a (m x mu) matrix with the 
%		   sensitivity of each point in each objective.

[n, mu] = size(P.x); %dimension and population size
m = size(P.f,1); %number of objectives
s = zeros(m, mu); %initialize the matrix of robustness

% Get the parameters for computing the sensitivity
nneigh = ceil(options.nneigh*n); %number of neighbors for each point
dx = options.dx;

for ii = 1:mu
	x = P.x(:,ii);
	fx = P.f(:,ii);
	
	%% Step 1: get neighbors for x
	% Artificially penalizes the bad points
	% If any x - dx < xmin or x + dx > xmax, then x is too close to a border of 
	% unfeasibility. Therefore, I will artificially give a bad fmax to it
	if any(x - dx/2 < xrange(:,1), 1) || any(x + dx/2 > xrange(:,2), 1)
		s(:,ii) = 10^6*ones(m,1);
		continue %skip the sensitivity computation
	end
	
	% Generates nneigh random neighbors
	noise = bsxfun(@times, dx, rand(n, nneigh) - 0.5); %noise in [-dx/2,dx/2]
	xnoise = bsxfun(@plus, x, noise); %add x to the noise
	fnoise = f(xnoise); %evaluate them
	N.x = [xnoise, x]; 
	N.f = [fnoise, fx];
		
	%% Step 2: perform the linear approximation for each function and solve it
	% Also, compute the sensitivity with this maximum value
	for jj = 1:m
		if strcmp(options.stype, "linear")
			%% Linear regression
			% Prepare the data for the regression. It will return something like
			%	y = c0 + c1x1 + c2x2 + ... cnxn
			% I can despise the first term for the optimization later
			X = [ones(size(N.x,2),1), N.x'];
			y = N.f(jj,:)';
			Beta = ols(y, X);
			c = Beta(2:end);
	
			%% Linear programming
			% Setup for the GLPK procedure
			A = zeros(1,n); % a dummy matrix of coefficients
			b = 0;
			ctype = 'F'; %set "F" to ignore the constraint Ax == b
			vartype = repmat('C', 1, n); %to set all variables continuous
			lb = x - dx/2; %lower bound
			ub = x + dx/2; %upper bound
			sense = -1; %to set the problem to maximization		
			% Run the linear solver
			[xmax, fmax] = glpk(c, A, b, lb, ub, ctype, vartype, sense);		
		
			% Using this xmax, compute its objective value
			ffunc = f(xmax);
			fmax = ffunc(jj); %only the jj-th objective value is useful		
			fmax = max(fmax, max(N.f(jj,:))); %update with the overall worst value
		
		else
			% Get the worst out of all neighbors created		
			fmax = max(N.f(jj,:));
		end
		
		%% Compute the sensitivity with this value of fmax
		% If absdiff is True, the sensitivity will be computed as absolute 
		% difference. Otherwise, as a percentage (assuming fx is not zero)
		if options.absdiff
			s(jj,ii) = abs(fmax - fx(jj));
		else 
			s(jj,ii) = abs(fmax - fx(jj))/abs(fx(jj));
		end
	end	
end

P.s = s; %append the sensitivities
