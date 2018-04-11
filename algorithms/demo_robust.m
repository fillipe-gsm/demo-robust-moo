function Popt = demo_robust(f, xrange, options)
%DEMO_ROBUST: DEMO but restricting the solutions to only the robust ones
%	Differently from the original DEMO_OPT, this algorithm does not normalize
%	the population in [0,1]. This facilitates the sensitivity computation.
%
%	Syntax:
%		[fopt, xopt, sopt, Phist] = demo_robust(f, xrange)
%		[fopt, xopt, sopt, Phist] = demo_robust(f, xrange, options)
%
%	Input arguments:
%		f: the objective function given as a handle, a .m file, inline, or
%			anything that can be computed using the "feval" function and can
%			handle multiple inputs. The output must be, for each point, a 
%			column vector of size m x 1, with m > 1 the number of objectives.
%		xrange: a matrix with the inferior and superior limits of search.
%				If n is the dimension, it will be a n x 2 matrix, such
%				that the first column contains the inferior limit, and the
%				second, the superior one;
%		options: a struct with internal parameters of the algorithm:
%			.F: the scale factor to be used in the mutation (default: 0.5);
%			.CR: the crossover factor used in the recombination (def.: 0.3);
%			.mu: the population size (number of individuals) (def.: 100);
%			.kmax: maximum number of iterations (def.: 300);
%			.display: true to display the population while the algorithm is
%					  being executed, and false to not (default: false);
%			.dx: a n-vector of tolerances in the search space (def.: 0.01);
%			.eta: a m-vector of tolerances in the objective space. It can be 
%			      given as a percentage of variation for each function (assuming
%			      they do not assume the value zero), or in absolute difference.
%			      def.: vector of 1, using absolute difference
%			.absdiff: a boolean with True if eta is in absolute difference, and 
%			          False if as a percentage.
%			.nneigh: number of neighbors for each point in order to compute its
%			         sensitivity, as a multiple of n (def.: 2)
%			If any of the parameters is not set, the default ones are used
%			instead.
%
%	Output arguments:
%		Popt, with the following structures:
%			fopt: the m x mu_opt matrix with the mu_opt best objectives
%			xopt: the n x mu_opt matrix with the mu_opt best individuals
%			sopt: a mu_opt vector with the sensitivity of each individual
%
%	Example: 
%		Solve the f1robust function with alpha = beta = 1 and n = 10 as the 
%		first test in section 3.2 of [1]
%	% Set the objective function
%	f = @(x) deb_robust (x, 1, 1); %x, alpha, beta
%	n = 10; %dimension
%	xrange = deb_range(n); %n = 10 variables
%	% Set parameters of the algorithm
%	options.dx = [0.01; 0.02*ones(n-1,1)]; %vector of uncertainties
%	options.eta = 1; % sensitivity tolerance
%	options.display = true; %to see the population evolving
%	Popt = demo_robust(f, xrange, options);
%	% Compare the results with Figures 4 and 5 from [1].
%
%	Reference:
%	[1] Goulart, Fillipe, et al. "Robust multiobjective optimization using 
%	regression models and linear subproblems." Proceedings of the Genetic and 
%	Evolutionary Computation Conference. ACM, 2017.

% Check the parameters of the algorithm
if nargin < 3 %options was not provided
	options = struct();
end
options = check_input(options);

% Initial considerations
n = size(xrange,1); %dimension of the problem
mu = options.mu; %population size
Xn = rand(n, mu); %normalized initial decision variables
Xmin = repmat(xrange(:,1), 1, mu); %replicates inferior limit
Xmax = repmat(xrange(:,2), 1, mu); %replicates superior limit
P.x = (Xmax - Xmin).*Xn + Xmin; %unnormalized population
P.f = f(P.x); %evaluates initial population
P = compute_sensitivity(f, P, xrange, options); %compute initial sensitivity
m = size(P.f, 1); %number of objectives

% Beginning of the main loop
for k = 1:options.kmax
	% Plot the current population (if desired)
	if options.display
		if m == 2
			plot(P.f(1,:), P.f(2,:), 'o');
			title('Objective values during the execution')
			xlabel('f_1'), ylabel('f_2')
			drawnow
		elseif m == 3
			plot3(P.f(1,:), P.f(2,:), P.f(3,:), 'o');
			title('Objective values during the execution')
			xlabel('f_1'), ylabel('f_2'), zlabel('f_3')
			drawnow
		end
	end
		
	% Perform the variation operation (mutation and recombination)
	O.x = mutation(P.x, options); %mutation
	O.x = recombination(P.x, O.x, options); %recombination
	O.x = repair(O.x, xrange); %assure the offspring do not cross the limits
	O.f = f(O.x); %compute objective functions
	O = compute_sensitivity(f, O, xrange, options);
	
	% Selection and updates
	P = selection(P, O, options);
		
	fprintf('Iteration %d/%d\r', k, options.kmax)
end
fprintf('\n')

% Return the final population
ispar = ndset(P.f);
Popt.f = P.f(:,ispar);
Popt.x = P.x(:,ispar);
Popt.s = P.s(:,ispar);

%=========================== Sub-functions ================================%
function Xo = mutation(Xp, options)
%MUTATION Performs mutation in the individuals
%	The mutation is one of the operators responsible for random changes in
%	the individuals. Each parent x will have a new individual, called trial
%	vector u, after the mutation.
%	To do that, pick up two random individuals from the population, x2 and
%	x3, and creates a difference vector v = x2 - x3. Then, chooses another
%	point, called base vector, xb, and creates the trial vector by
%
%		u = xb + F*v = xb + F*(x2 - x3)
%
%	wherein F is an internal parameter, called scale factor.
%
%	Syntax:
%		Xo = mutation(Xp, options)
%
%	Input arguments:
%		Xp: a n x mu matrix with mu "parents" and of dimension n
%		options: the struct with the internal parameters
%
%	Output arguments:
%		Xo: a n x mu matrix with the mu mutated individuals (of dimension n)

% Creates a mu x mu matrix of 1:n elements on each row
A = repmat((1:options.mu), options.mu, 1);
% Now, as taken at the MatLab Central, one removes the diagonal of
% A, because it contains indexes that repeat the current i-th
% individual
A = A';
A(logical(eye(size(A)))) = []; %removes the diagonal
A = transpose(reshape(A, options.mu-1, options.mu)); %reshapes

% Now, creates a matrix that permutes the elements of A randomly
[~, J] = sort(rand(size(A)),2);
Ilin = bsxfun(@plus,(J-1)*options.mu,(1:options.mu)');
A(:) = A(Ilin);

% Chooses three random points (for each row)
xbase = Xp(:, A(:,1)); %base vectors
v = Xp(:,A(:,2)) - Xp(:,A(:,3)); %difference vector

% Performs the mutation
Xo = xbase + options.F*v;
%--------------------------------------------------------------------------%
function Xo = recombination(Xp, Xm, options)
%RECOMBINATION Performs recombination in the individuals
%	The recombination combines the information of the parents and the
%	mutated individuals (also called "trial vectors") to create the
%	offspring. Assuming x represents the i-th parent, and u the i-th trial
%	vector (obtained from the mutation), the offspring xo will have the
%	following j-th coordinate:
%
%		xo_j = u_j if rand_j <= CR
%				 x_j otherwise
%
%	wherein rand_j is a number drawn from a uniform distribution from 0 to
%	1, and CR is called the crossover factor. To prevent mere copies, at
%	least one coordinate is guaranteed to belong to the trial vector.
%
%	Syntax:
%		Xo = recombination(Xp, Xm, options)
%
%	Input arguments:
%		Xp: a n x mu matrix with the mu parents
%		Xm: a n x mu matrix with the mu mutated points
%		options: the struct with the internal parameters
%
%	Output argument:
%		Xo: a n x mu matrix with the recombinated points (offspring)

% Draws random numbers and checks whether they are smaller or
% greater than CR
n = size(Xp, 1); %dimension of the problem
aux = rand(n, options.mu) <= options.CR;
% Now assures at least one coordinate will be changed, that is,
% there is at least one 'true' in each column
auxs = sum(aux) == 0; %gets the columns with no trues
indc = find(auxs); %get the number of the columns
indr = randi(n, 1, sum(auxs)); %define random indexes of rows
if isempty(indr), indr = []; end
if isempty(indc), indc = []; end
ind = sub2ind([n, options.mu], indr, indc); %converts to indexes
aux(ind) = true;

% Finally, creates the offspring
Xo = Xp;
Xo(aux) = Xm(aux);
%--------------------------------------------------------------------------%
function Xo = repair(Xo, xrange)
%REPAIR Truncates the population to be in the feasible region
%
%	Syntax:
%		Xo = repair(Xo, options)

% This is easy, because the population must be inside the interval [xmin, xmax]
Xo = max(Xo, xrange(:,1)); %corrects inferior limit
Xo = min(Xo, xrange(:,2)); %superior limit

%--------------------------------------------------------------------------%
function Pnew = selection(P, O, options)
%SELECTION Selects the next population
%	Each parent is compared to its offspring. If the parent dominates its 
%	child, then it goes to the next population. If the offspring dominates 
%	the parent, that new member is added. However, if they are incomparable
%	(there is no mutual domination), them both are sent to the next 
%	population. After that, the new set of individuals must be truncated to 
%	mu, wherein mu is the original number of points.
%	This is accomplished by the use of "non-dominated sorting", that is,
%	ranks the individual in fronts of non-domination, and within each
%	front, measures them by using crowding distance. With regard to these
%	two metrics, the best individuals are kept in the new population.
%
%	Syntax:
%		Pnew = selection(P, O, options)
%
%	Input arguments:
%		P: a struct with the parents (x and f)
%		O: a struct with the offspring
%		options: the struct with the algorithm's parameters
%
%	Output argument:
%		Pnew: the new population (a struct with x and f)

% ------ First part: checks robustness between parents and offspring
% This needs to be different from the original selection. Here, if the parent 
% satisfies the robustness criterion (r < eta) and the offspring doesn't, the 
% first enters the next population, and vice-versa. If both satisfies, then both
% enter. If none satisfy, then the one with best value of r enters

% Verifies whether parent dominates offspring
aux1 = all(P.f <= O.f, 1);
aux2 = any(P.f < O.f, 1);
auxp = and(aux1, aux2); %P dominates O
% Now, where offspring dominates parent
aux1 = all(P.f >= O.f, 1);
aux2 = any(P.f > O.f, 1);
auxo = and(aux1, aux2); %O dominates P
auxpo = and(~auxp, ~auxo); %P and O are incomparable

% For now, I will just include a for loop
mu = size(P.x,2);
R.f = [];
R.x = [];
R.s = [];
eta = options.eta;
for ii = 1:mu
	x = [P.x(:,ii), O.x(:,ii)];
	fx = [P.f(:,ii), O.f(:,ii)];
	s = [P.s(:,ii), O.s(:,ii)];
	test_s = all(s <= eta, 1); %are all robustness constraints satisfied?
	% if any test_s is true, then the ones with True go to the intermediate
	% population. If there is no true, then get the one with best r value
	if sum(test_s) == 1 %only one passed
		R.x = [R.x, x(:,test_s)];
		R.f = [R.f, fx(:,test_s)];
		R.s = [R.s, s(:,test_s)];
	elseif sum(test_s) == 2 %both passed
		% If auxp(ii) is true, then the parent dominates the offspring
		% If auxo(ii) is true, then the offspring is the dominant
		% If auxpo(ii) is true, then they are incomparable
		if auxp(ii)
			ind = 1;
		elseif auxo(ii)
			ind = 2;
		elseif auxpo(ii)
			ind = [1,2];
		elseif auxp(ii) & auxpo(ii)
			error('Que porra é essa?!')
		end
		R.x = [R.x, x(:,ind)];
		R.f = [R.f, fx(:,ind)];
		R.s = [R.s, s(:,ind)];
	else %none passed
		if s(1) < s(2)
			ind = 1;
		else
			ind = 2;
		end
		R.x = [R.x, x(:,ind)];
		R.f = [R.f, fx(:,ind)];
		R.s = [R.s, s(:,ind)];
	end
end

% In the end, we have a R population with at least mu points. If there is more
% than required, perform the usual nondominated sorting with crowding distance
mu_r = size(R.x,2);
if mu_r == mu
	Pnew = R; %all .f, .x and .s
else %too many points
	% Perform the non-dominated sorting
	Pnew.x = []; Pnew.f = []; Pnew.s = []; %prepare the new population
	while true
		ispar = ndset(R.f); %gets the non-dominated front
		% If the number of points in this front plus the current size of the new
		% population is smaller than mu, then include everything and keep going.
		% If it is greater, then stop and go to the truncation step
		if size(Pnew.f, 2) + sum(ispar) < options.mu
		  Pnew.f = [Pnew.f, R.f(:,ispar)];
		  Pnew.x = [Pnew.x, R.x(:,ispar)];
		  Pnew.s = [Pnew.s, R.s(:,ispar)];
		  % Remove this front
		  R.f(:,ispar) = []; R.x(:,ispar) = []; R.s(:,ispar) = [];
		else
		  % Gets the points of this front and goes to the truncation part
		  Frem = R.f(:,ispar);
		  Xrem = R.x(:,ispar);
		  Rrem = R.s(:,ispar);
		  break %don't forget this to stop this infinite loop
		end
	end
	
	% Finally, truncates Frem, Xrem and Rrem with the crowding distance
	aux = (size(Pnew.f,2) + size(Frem,2)) - options.mu; %remaining points to fill
	if aux == 0
		Pnew.x = [Pnew.x, Xrem]; Pnew.f = [Pnew.f, Frem]; Pnew.s = [Pnew.s, Rrem];
	elseif aux > 0
		for ii = 1:aux
		  cdist = crowdingdistance(Frem);
		  [~, imin] = min(cdist); %gets the point with smaller crowding distance
		  Frem(:,imin) = []; %and remove it
		  Xrem(:,imin) = [];
		  Rrem(:,imin) = [];
		end
		Pnew.x = [Pnew.x, Xrem]; 
		Pnew.f = [Pnew.f, Frem];
		Pnew.s = [Pnew.s, Rrem];
	else %if there are too few points... well, we're doomed!
		error('Run to the hills! This is not supposed to happen!')
	end

end

% ---------------------------------------------------------------------------- %
function options = check_input(options)
%CHECK_INPUT Checks the parameters of the algorithm before
%	This sub-function checks the endogenous parameters of the algorithm. If
%	they are not set, the default ones are used

if ~isfield(options, 'F') %scale factor
	options.F = 0.5;
end

if ~isfield(options, 'CR') %crossover factor
	options.CR = 0.3;
end

if ~isfield(options, 'kmax') %maximum number of iterations
	options.kmax = 300;
end

if ~isfield(options, 'mu') %population size
	options.mu = 100;
end

if ~isfield(options, 'display') %show or not the population during execution
	options.display = false;
end

if ~isfield(options, 'dx')
	options.dx = 0.01;
end

if ~isfield(options, 'eta')
	options.eta = 1;
end

if ~isfield(options, 'absdiff')
	options.absdiff = false;
end

% Number of neighbors to approximate the worst value of f in N(x, dx)
% this number is given as a MULTIPLE of n!!!!
if ~isfield(options, 'nneigh')
	options.nneigh = 2;
end

% Method to compute the sensitivity: "linear" or "sample"
if ~isfield(options, 'stype') 
	options.stype = "linear";
end
