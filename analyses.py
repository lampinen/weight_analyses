import numpy as np
from scipy.stats import ttest_1samp, ttest_ind, entropy
from orthogonal_matrices import random_orthogonal

num_runs = 5
num_layers = 5 # output layer distributions will be shaped by targets, so skip 

#weights = [[] for i in range(num_layers)]
#biases = [[] for i in range(num_layers)]
output_modes = [[] for i in range(num_layers)]
strengths = [[] for i in range(num_layers)]

#linear_weights = [[] for i in range(num_layers)]
linear_output_modes = [[] for i in range(num_layers)]
linear_strengths = [[] for i in range(num_layers)]

#tanh_weights = [[] for i in range(num_layers)]
#tanh_output_modes = [[] for i in range(num_layers)]
#tanh_strengths = [[] for i in range(num_layers)]

#pre_weights = [[] for i in range(num_layers)]
pre_output_modes = [[] for i in range(num_layers)]
pre_strengths = [[] for i in range(num_layers)]

random_output_modes = [[] for i in range(num_layers)]

input_modes = [[] for i in range(num_layers)]
linear_input_modes = [[] for i in range(num_layers)]
pre_input_modes = [[] for i in range(num_layers)]
random_input_modes = [[] for i in range(num_layers)]

for run_i in range(num_runs):
    for layer in range(num_layers):
        # post
        filename = "./results_all_layers/run%i_nonlinear_post_layer%i_weights.csv" % (run_i, layer)
        these_weights = np.loadtxt(filename, delimiter=',')
#        weights[layer].append(these_weights)
#        filename = "./results_all_layers/run%i_post_first_layer_biases.csv" % run_i
#        these_biases = np.loadtxt(filename, delimiter=',')
#        biases.append(these_biases)

        U, S, V = np.linalg.svd(these_weights, full_matrices=False)

        # output modes are rows of V
        output_modes[layer].append(V)
        strengths[layer].append(S)
        input_modes[layer].append(U)

        # random
        random_output_modes[layer].append(random_orthogonal(V.shape[-1]))
        random_input_modes[layer].append(random_orthogonal(U.shape[0]))

        # pre
        filename = "./results_all_layers/run%i_nonlinear_pre_layer%i_weights.csv" % (run_i, layer)
        these_weights = np.loadtxt(filename, delimiter=',')
#        pre_weights[layer].append(these_weights)

        U, S, V = np.linalg.svd(these_weights, full_matrices=False)
        pre_output_modes[layer].append(V)
        pre_strengths[layer].append(S)
        pre_input_modes[layer].append(U)


        # linear
        filename = "./results_all_layers/run%i_linear_post_layer%i_weights.csv" % (run_i, layer)
        these_weights = np.loadtxt(filename, delimiter=',')
#        linear_weights[layer].append(these_weights)

        U, S, V = np.linalg.svd(these_weights, full_matrices=False)
        # output modes are rows of V
        linear_output_modes[layer].append(V)
        linear_strengths[layer].append(S)
        linear_input_modes[layer].append(U)

#        # tanh
#        filename = "./results_all_layers/run%i_tanh_post_first_layer_weights.csv" % run_i
#        these_weights = np.loadtxt(filename, delimiter=',')
#        tanh_weights[layer].append(these_weights)
#
#        U, S, V = np.linalg.svd(these_weights, full_matrices=False)
#        # output modes are rows of V
#        tanh_output_modes[layer].append(V)
#        tanh_strengths[layer].append(S)

#for layer in range(num_layers):
#    post_simil = np.zeros([num_runs,num_runs])
#    for i in range(num_runs):
#        for j in range(num_runs):
#            this_simil = np.dot(output_modes[layer][i], output_modes[layer][j].transpose())
#            post_simil[i][j] = np.sum(this_simil)
#
#    pre_simil = np.zeros([num_runs,num_runs])
#    for i in range(num_runs):
#        for j in range(num_runs):
#            this_simil = np.dot(pre_output_modes[layer][i], pre_output_modes[layer][j].transpose())
#            pre_simil[i][j] = np.sum(this_simil)
#
##    print(strengths)
##    print(post_simil)
##    print(pre_simil)
#
#    print(np.shape((np.abs(post_simil)-np.abs(pre_simil))[np.where(~np.eye(num_runs,dtype=bool))]))
#    print("T-test between similarities (post - pre, abs. values, diagonals removed)")
#    print(ttest_1samp((np.abs(post_simil)-np.abs(pre_simil))[np.where(~np.eye(num_runs,dtype=bool))], 0))

#om_ents_pre = []
#om_ents_post = []
#
#om_maxs_pre = []
#om_maxs_post = []
with open("./results_all_layers/entropies.csv", "w") as fout:
    fout.write('run, layer, type, mode_type, mode_rank, mode_strength, entropy\n')
    for i in range(num_runs): 
        for layer in range(num_layers):
            this_pom = pre_output_modes[layer][i]
            this_ps = strengths[layer][i]
            this_pim = pre_input_modes[layer][i]
            this_rom = random_output_modes[layer][i]
            this_rim = random_input_modes[layer][i]
            this_om = output_modes[layer][i]
            this_im = input_modes[layer][i]
            this_lom = linear_output_modes[layer][i]
            this_lim = linear_input_modes[layer][i]
#        this_tom = tanh_output_modes[layer][i]
            this_pps = pre_strengths[layer][i]
            this_lps = linear_strengths[layer][i]
#        this_tps = tanh_strengths[layer][i]
#        om_maxs_pre.append(np.amax(this_pom, axis=1))
#        om_maxs_post.append(np.amax(this_om, axis=1))

            for mode_j in range(len(this_om)):
                this_ent = entropy(np.square(this_pom[mode_j, :]))
#            om_ents_pre.append(this_ent)
                fout.write('%i, %i, %s, %s, %i, %f, %f\n' % (i, layer, "pre", "output", mode_j+1, this_pps[mode_j], this_ent))
                this_ent = entropy(np.square(this_om[mode_j, :]))
#            om_ents_post.append(this_ent)
                fout.write('%i, %i, %s, %s, %i, %f, %f\n' % (i, layer, "post", "output", mode_j+1, this_ps[mode_j], this_ent))

                this_ent = entropy(np.square(this_rom[mode_j, :]))
                fout.write('%i, %i, %s, %s, %i, NA, %f\n' % (i, layer, "random_orthogonal", "output", mode_j+1, this_ent))

                this_ent = entropy(np.square(this_lom[mode_j, :]))
                fout.write('%i, %i, %s, %s, %i, %s, %f\n' % (i, layer, "linear_post", "output", mode_j+1, this_lps[mode_j], this_ent))
                
                this_ent = entropy(np.square(this_pim[:, mode_j]))
#            om_ents_pre.append(this_ent)
                fout.write('%i, %i, %s, %s, %i, %f, %f\n' % (i, layer, "pre", "input", mode_j+1, this_pps[mode_j], this_ent))
                this_ent = entropy(np.square(this_im[:, mode_j]))
#            om_ents_post.append(this_ent)
                fout.write('%i, %i, %s, %s, %i, %f, %f\n' % (i, layer, "post", "input", mode_j+1, this_ps[mode_j], this_ent))

                this_ent = entropy(np.square(this_rim[:, mode_j]))
                fout.write('%i, %i, %s, %s, %i, NA, %f\n' % (i, layer, "random_orthogonal", "input", mode_j+1, this_ent))

                this_ent = entropy(np.square(this_lim[:, mode_j]))
                fout.write('%i, %i, %s, %s, %i, %s, %f\n' % (i, layer, "linear_post", "input", mode_j+1, this_lps[mode_j], this_ent))
#                this_ent = entropy(np.square(this_tom[mode_j, :]))
#                fout.write('%i, %i, %s, %i, %s, %f\n' % (i, layer, "tanh_post", mode_j+1, this_tps[mode_j], this_ent))
    
#om_maxs_pre = np.array(om_maxs_pre).flatten()
#om_maxs_post = np.array(om_maxs_post).flatten()
#om_ents_pre = np.array(om_ents_pre).flatten()
#om_ents_post = np.array(om_ents_post).flatten()
#
#print("(2-sample) T-test between post mode maxes and pre mode maxes")
#print(ttest_ind(om_maxs_post, om_maxs_pre, equal_var=False))
#
#
#print("(2-sample) T-test between post mode entropies and pre mode entropies")
#print(ttest_ind(om_ents_post, om_ents_pre, equal_var=False))
