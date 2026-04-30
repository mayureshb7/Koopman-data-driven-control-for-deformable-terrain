function X_pred = eval_prediction(X,operator,basis,prediction,Flag)
% function to obtain trajectories predicted using Koopman operator
% Inputs
% X             : Validation dataset
% operator      : Koopman matrix approximated using DMD or EDMD
% basis         : structure with properites of basis functions
% prediction    : sturecure with properties of trajectoreis for prediction
% Outputs
% X_pred        : Dataset of trajectories predicted using Koopman
A = operator.A;
step_numbers = prediction.n_steps; %20
traj_numbers = prediction.n_traj; %9
state_numbers = prediction.n; %2
num_trajectory_points = length(X)/traj_numbers;
lift_number = size(A, 1);
X_pred = zeros(step_numbers, traj_numbers*state_numbers);
C = [eye(state_numbers), zeros(state_numbers, lift_number - state_numbers)];
if Flag == 1

    
    for i = 1:traj_numbers
        x_0 = X((i-1)*num_trajectory_points+1, :)';
    
        
            start_column = (i-1)*state_numbers+1;
            end_column = i*state_numbers;
            X_current = x_0;
            X_pred(1,start_column:end_column) = X_current';
    
        for k_t = 2:step_numbers
            X_current = A*X_current;
            % start_column = (i-1)*state_numbers+1;
            % end_column = i*state_numbers;
    
    
            X_pred(k_t,start_column:end_column) = (C * X_current)';
    
        end
    
    end
    % X_pred = X_predction_concat;
    elseif Flag ==2

    exponents = generate_monomial_exp(state_numbers, basis.deg);

    for i = 1:traj_numbers
        x_0 = X((i-1)*num_trajectory_points+1, :)';
        start_column = (i-1)*state_numbers+1;
        end_column = i*state_numbers;
        X_current = [x_0;evaluate_monomials(x_0, exponents)];
        X_pred(1,start_column:end_column) = x_0';
        for k_t = 2:step_numbers
               X_current = A*X_current;
               X_pred(k_t,start_column:end_column) = (C * X_current)';
            
        end
    end

    end
end




function exponents = generate_monomial_exp(n, degree)
    exponents = [];
    for number_total = 2:degree
         exponents = [exponents; current_combinations(number_total, n)];
    end
end

function Comb = current_combinations(number_total, n)
    if n ==1
        Comb = number_total;
        return;
    end
    Comb = [];
    for idx = 0:number_total
        data_end = current_combinations(number_total-idx, n-1);
        Comb = [Comb;[idx*ones(size(data_end,1),1),data_end]];
    end
end

function values = evaluate_monomials(x, exponents)
    values = prod(x(:)' .^ exponents, 2);
end