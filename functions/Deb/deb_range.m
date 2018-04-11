function xrange = deb_range(n)
%DEB_RANGE Get the range of the search space for DEB_ROBUST test function
%	x1 goes from 0 to 1 and x2:xn goes from -1 to 1.
%
%	Syntax:
%		xrange = deb_range(n)
%
%	Input argument:
%		n: dimension of the problem
%
%	Output argument:
%		xrange: a (n x 2) matrix with the search limits
%
%	See also deb_robust, deb_sensitivity


xrange = [0 1; -ones(n-1,1), ones(n-1,1)];
