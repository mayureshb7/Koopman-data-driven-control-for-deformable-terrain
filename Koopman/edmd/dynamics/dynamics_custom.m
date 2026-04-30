function dxdt = dynamics_custom(t,x)
% generic function to represent system dynamics
% Inputs
% x     : system state
% t     : system time
% Outputs
% dxdt  : nonlinear dynmaics in state space form

%%%%%%%%%%%%% nonlinear dynamics function %%%%%%%%%%%%%%%%%%%%
% complete code for represeting nonlinear dynamics in state space
% dxdt = ??;

x1 = x(1);
x2 = x(2);

denominator = (9*x1^2*x2^2 +6*x1^2+3*x2^2 + cos(x2) + 2);
dx1 = ((7.5*x2^2 + 5)*(x1^3 + x1 + sin(x2)) + (-x1+x2^3+2*x2)*cos(x2))/denominator;
dx2 = (2.5*x1^3 + 2.5*x1 - (3.*x1^2 + 1)*(-x1 + x2^3 + 2*x2) + 2.5*sin(x2))/denominator;

dxdt = [dx1;dx2];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end