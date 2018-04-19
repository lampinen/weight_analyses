import numpy as np
from scipy.stats import ttest_1samp


weights = []
biases = []
output_modes = []
strengths = []

pre_weights = []
pre_output_modes = []
pre_strengths = []

for run_i in range(10):
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

    filename = "./results/run%i_pre_first_layer_weights.csv" % run_i
    these_weights = np.loadtxt(filename, delimiter=',')
    pre_weights.append(these_weights)

    U, S, V = np.linalg.svd(these_weights, full_matrices=False)
    pre_output_modes.append(V)
    pre_strengths.append(S)

post_simil = np.zeros([10,10])
for i in range(10):
    for j in range(10):
	this_simil = np.dot(output_modes[i], output_modes[j].transpose())
	post_simil[i][j] = np.sum(this_simil)

pre_simil = np.zeros([10,10])
for i in range(10):
    for j in range(10):
	this_simil = np.dot(pre_output_modes[i], pre_output_modes[j].transpose())
	pre_simil[i][j] = np.sum(this_simil)

print(strengths)
print(post_simil)
print(pre_simil)

print(np.shape((np.abs(post_simil)-np.abs(pre_simil))[np.where(~np.eye(10,dtype=bool))]))
print(ttest_1samp((np.abs(post_simil)-np.abs(pre_simil))[np.where(~np.eye(10,dtype=bool))], 0))

