import numpy as np
from scipy.stats import ttest_1samp, ttest_ind, entropy
from orthogonal_matrices import random_orthogonal

num_runs = 10

weights = []
biases = []
output_modes = []
strengths = []

linear_weights = []
linear_output_modes = []
linear_strengths = []

pre_weights = []
pre_output_modes = []
pre_strengths = []

random_output_modes = []

for run_i in range(num_runs):
    # post
    filename = "./results/run%i_post_first_layer_weights.csv" % run_i
    these_weights = np.loadtxt(filename, delimiter=',')
    weights.append(these_weights)
    filename = "./results/run%i_post_first_layer_biases.csv" % run_i
    these_biases = np.loadtxt(filename, delimiter=',')
    biases.append(these_biases)

    U, S, V = np.linalg.svd(these_weights, full_matrices=False)
    # output modes are rows of V
    output_modes.append(V)
    strengths.append(S)

    # pre
    filename = "./results/run%i_pre_first_layer_weights.csv" % run_i
    these_weights = np.loadtxt(filename, delimiter=',')
    pre_weights.append(these_weights)

    U, S, V = np.linalg.svd(these_weights, full_matrices=False)
    pre_output_modes.append(V)
    pre_strengths.append(S)

    # random
    random_output_modes.append(random_orthogonal(len(S)))

    # linear
    filename = "./results/run%i_linear_post_first_layer_weights.csv" % run_i
    these_weights = np.loadtxt(filename, delimiter=',')
    linear_weights.append(these_weights)

    U, S, V = np.linalg.svd(these_weights, full_matrices=False)
    # output modes are rows of V
    linear_output_modes.append(V)
    linear_strengths.append(S)

post_simil = np.zeros([num_runs,num_runs])
for i in range(num_runs):
    for j in range(num_runs):
	this_simil = np.dot(output_modes[i], output_modes[j].transpose())
	post_simil[i][j] = np.sum(this_simil)

pre_simil = np.zeros([num_runs,num_runs])
for i in range(num_runs):
    for j in range(num_runs):
	this_simil = np.dot(pre_output_modes[i], pre_output_modes[j].transpose())
	pre_simil[i][j] = np.sum(this_simil)

print(strengths)
print(post_simil)
print(pre_simil)

print(np.shape((np.abs(post_simil)-np.abs(pre_simil))[np.where(~np.eye(num_runs,dtype=bool))]))
print("T-test between similarities (post - pre, abs. values, diagonals removed)")
print(ttest_1samp((np.abs(post_simil)-np.abs(pre_simil))[np.where(~np.eye(num_runs,dtype=bool))], 0))

om_ents_pre = []
om_ents_post = []

om_maxs_pre = []
om_maxs_post = []
with open("./results/entropies.csv", "w") as fout:
    fout.write('run, type, mode_rank, mode_strength, entropy\n')
    for i in range(num_runs): 
        this_pom = pre_output_modes[i]
        this_ps = strengths[i]
        this_rom = random_output_modes[i]
        this_om = output_modes[i]
        this_lom = linear_output_modes[i]
        this_pps = pre_strengths[i]
        this_lps = linear_strengths[i]
        om_maxs_pre.append(np.amax(this_pom, axis=1))
        om_maxs_post.append(np.amax(this_om, axis=1))

        for mode_j in range(len(this_om)):
            this_ent = entropy(np.square(this_pom[mode_j, :]))
            om_ents_pre.append(this_ent)
            fout.write('%i, %s, %i, %f, %f\n' % (i, "pre", mode_j+1, this_pps[mode_j], this_ent))
            this_ent = entropy(np.square(this_om[mode_j, :]))
            om_ents_post.append(this_ent)
            fout.write('%i, %s, %i, %f, %f\n' % (i, "post", mode_j+1, this_ps[mode_j], this_ent))

            this_ent = entropy(np.square(this_rom[mode_j, :]))
            fout.write('%i, %s, %i, NA, %f\n' % (i, "random_orthogonal", mode_j+1, this_ent))

            this_ent = entropy(np.square(this_lom[mode_j, :]))
            fout.write('%i, %s, %i, %s, %f\n' % (i, "linear_post", mode_j+1, this_lps[mode_j], this_ent))
            
    
om_maxs_pre = np.array(om_maxs_pre).flatten()
om_maxs_post = np.array(om_maxs_post).flatten()
om_ents_pre = np.array(om_ents_pre).flatten()
om_ents_post = np.array(om_ents_post).flatten()

print("(2-sample) T-test between post mode maxes and pre mode maxes")
print(ttest_ind(om_maxs_post, om_maxs_pre, equal_var=False))


print("(2-sample) T-test between post mode entropies and pre mode entropies")
print(ttest_ind(om_ents_post, om_ents_pre, equal_var=False))
