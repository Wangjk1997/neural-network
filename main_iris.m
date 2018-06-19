Irisdata;
times =500;
W = rand(4,3);
V = rand(4,4);
theta = rand(1,3);
r = rand(1,4);
learning_rate = 0.05;
for j = 1:times
    for i = 1:size(r_training_set,1)
        alpha = r_training_set(i,1:size(r_training_set,2) -1) * V;
        b = sigmoid(alpha - r);
        beta = b * W;
        output = sigmoid(beta - theta);
        if r_training_set(i,size(r_training_set,2)) == 0
            label = [1,0,0];
        end
        if r_training_set(i,size(r_training_set,2)) == 0.5
            label = [0,1,0];
        end
        if r_training_set(i,size(r_training_set,2)) == 1
            label = [0,0,1];
        end
        g = output .* (ones(size(output)) - output) .* (label - output);
        delta_W = learning_rate * b' * g;
        delta_theta = - learning_rate * g;
        e = b.*(1 - b).*(g * W');
        delta_V = learning_rate * r_training_set(i,1:size(r_training_set,2)-1)' *e;
        delta_r = -learning_rate * e;
        %¸üÐÂ
        W = W + delta_W;
        V = V + delta_V;
        r = r + delta_r;
        theta = theta + delta_theta;
    end
end
for k= 1:size(r_validation_set,1)
    alpha_v = r_validation_set(k,1:size(r_validation_set,2) -1) * V;
    b_v = sigmoid(alpha_v - r);
    beta_v = b_v * W;
    output_v = sigmoid(beta_v - theta)
    r_validation_set(k,size(r_validation_set,2))
end
