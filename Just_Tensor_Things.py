import tensorflow as tf
from tensorflow import keras
from keras import models
from tkinter import *
from Gene import Gene
from Model import Model
import math


"""
GOALS FOR THIS PROGRAM:

- set and determine back-end values for a NEAT algo
- have a helper method for organizing data into a workable pickle from csv, txt, etc...
- put up an eta window and a speed thing
- have a pretty output window w/ graphs using matplotlib
- set a determined accuracy to quit the function
- type of inputs: continuous like flappy bird or static like imdb - might be difficult to parse variables from inside program
- number of inputs, based upon buffered input length (should change when organizing data, not after / before)





"""

# def set_up_gui():

# 	window = Tk()

# 	window.title("Sentimental Analysis GUI")

# 	window.geometry('600x400')


# 	#CheckBox Variables
# 	RedditVar = IntVar()
# 	TwitterVar = IntVar()
# 	InstagramVar = IntVar()


# 	subRedditkeyphraseBox = Entry(window, width=10,)
# 	subRedditkeyphraseBox.grid(column=2, row=2)

# 	lb2 = Label(window, text="Enter Subreddit:")
# 	lb2.grid(column=1, row=2)


# 	InstaBox = Entry(window, width=10,)
# 	InstaBox.grid(column=2, row=3)
# 	lb3 = Label(window, text="Enter Instagram Handle:")
# 	lb3.grid(column=1, row=3)


# 	lbl = Label(window, text="Enter your key phrase")

# 	lbl.grid(column=0, row=0)

# 	keyphrasetxtBox = Entry(window, width=10)

# 	keyphrasetxtBox.grid(column=1, row=0)

# 	btn = Button(window, text="Search", command=clicked)

# 	btn.grid(column=2, row=0)

# 	#Check Box
# 	Checkbutton(window, text="Twitter", variable=TwitterVar).grid(row=1, sticky=W)
# 	Checkbutton(window, text="Reddit", variable=RedditVar).grid(row=2, sticky=W)
# 	Checkbutton(window, text="Instagram", variable=InstagramVar).grid(row=3, sticky=W)

# 	window.mainloop()




# def clicked():
# 	scrollTextwindow = Tk()

# 	scrollTextwindow.title("CommandLine")

# 	scrollTextwindow.geometry('1000x800')
# 	scrollText = scrolledtext.ScrolledText(scrollTextwindow, width=800, height=600)

# 	scrollText.grid(column=0, row=5)


POPULATION_SIZE = 100
targetSpeciesAmount = 10.2 # target amount of species to have in population to have a decent enough spread of solutions
compatabilityThreshold = 6.0 # averaged how many genes not shared (disjoint) to be considered a different species
compatabilityMod = 0.3 # change in compatability threshold
dropoffAge = 15.0 # after 15 generations with no increase in fitness, penalize the species
survival_threshold = 0.85 # top 100*(1-x) percent survive and are allowed to reproduce
weight_mutation_rate = .80 #mutate weights at 80% rate
node_mutation_rate = .03 # mutate nodes at 3-30% rate
gene_mutation_rate = .05 # mutate links at 5%
mutation_power = .5 # mutate up to 2.5
disjoint_ratio = 1.0 # averaged disjoint between two species; ratio of 1 disjoint worth average of 3 in weight
excessRatio = 1.0
num_species = 1
weightDifferenceRatio = 0.3 * POPULATION_SIZE / 100 # 0.4 -> 3

fitness = []
global_genes = []
current_pop = []
global_num_nodes = 0


def update_compat_threshold(): # DONE
	global num_species, target_species_amount, compatability_mod, compatability_threshold

	if num_species<target_species_amount:
		compatability_threshold-=compatability_mod # compat_mod works well at about 0.3
	else if (num_species>target_species_amount):
		compatability_threshold+=compatability_mod
	if (compatability_threshold<0.3):
		compatability_threshold=0.3

def save(current_pop, num_models): # needs to save more than weights
	for i in range(num_models):
		current_pop[i].save("Current_Model - - model_num_" + str(i))
	print("Saved current pop!")

def load_saved(num_models):
	current_pop = []
	for i in range(num_models):
			current_pop[i] = load_model("Current_Model - - model_num_" + str(i))
	return current_pop

def blank_txt():
	f = open('file.txt', 'r+')
	f.truncate(0) # need '0' when using r+
	f.close()

def two_parent_crossover(model_idx1, model_idx2): # DONE
	
	# matching genes are inherited randomly (including disabled/enabled), and disjoint genes are taken from more fit parent
	# new_genes = []

	if fitness[model_idx1] > fitness[model_idx2]:
		new_model = make_new_model(model_idx1, model_idx2)
	else:
		new_model = make_new_model(model_idx2, model_idx1)
	
	return new_model

def make_new_model(model_idx1, model_idx2): # DONE
	global current_pop, fitness, global_genes

	genes1 = current_pop[model_idx1].get_genes()
	genes2 = current_pop[model_idx2].get_genes()

	enabled1 = current_pop[model_idx1].get_enabled_list()
	enabled2 = current_pop[model_idx2].get_enabled_list()

	weights1 = current_pop[model_idx1].get_weights()
	weights2 = current_pop[model_idx2].get_weights()
	
	new_model = Model() # # (self, gene, enabled, weight)

	count = 0
	for gene in range(len(genes1)):
		inno1 = genes1[gene].get_inno_num()
		inno2 = genes2[count].get_inno_num()
		while inno2 < inno1:
			if count < len(genes2):
				inno2 = genes2[count].get_inno_num()
				count += 1
			else:
				for i in range(gene, len(genes1)):
					new_model.duplicate(genes1[i], enabled1[i], weights1[i])
				return new_model
		if inno1 == inno2:
			if random.uniform(0, 1) > 0.5:
				new_model.duplicate(genes1[gene], enabled1[gene], weights1[gene])
			else:
				new_model.duplicate(genes2[count], enabled2[count], weights2[count])
				count += 1
		else:
			new_model.duplicate(genes1[gene], enabled1[gene], weights1[gene])
	return new_model


def mutate_weights(weight_mutation_rate, mutation_power): # DONE
	global current_pop
	for xi in range(len(current_pop)):
		weights = current_pop[xi].get_weights()
		for yi in range(len(weights[xi])):
			if random.uniform(0, 1) < weight_mutation_rate:
				change = random.uniform(-mutation_power, mutation_power)
				weights[xi][yi] += change
	return weights

def node_already_exists(model, genes, ind): # DONE
	# checks to see if exact node nums exist already
	global global_genes
	# for i in range(len(global_genes)-1): # will check to see if two consecutive (inno nums) genes form a line to the highest node at that point (what happens if I sort it?)
	# 		# if node already exists
	# 			if global_genes[i].get_node_in() == genes[ind].get_node_in():
	# 				if (global_genes[i+1].get_node_out() == genes[ind].get_node_out()) and (global_genes[i+1].get_inno_num() == global_genes[i].get_inno_num() + 1): # if they have consecutive inno nums, and the same output
	one = False
	one_i = -1
	two = False
	two_i = -1
	n_in = genes[ind].get_node_in()
	n_out = genes[ind].get_node_out()
	n_num = model.get_num_nodes()
	for i in range(len(global_genes)):
		if (global_genes[i].get_node_in() == n_in) and (global_genes[i].get_node_out() = n_num):
			one = True
			one_i = i
		elif (global_genes[i].get_node_in() == n_num) and (global_genes[i].get_node_out() = n_out):
			two = True
			two_i = i
	return [one, one_i, two, two_i]

def mutate_nodes(node_mutation_rate): # DONE
	global current_pop, global_genes, global_num_nodes
	for xi in range(len(current_pop)):
		if random.uniform(0, 1) < node_mutation_rate:
			model = current_pop[xi]
			genes = model.get_genes()
			ind = random.randint(0, len(genes))
			num_nodes = current_pop[xi].get_num_nodes() # will use num nodes to set the new node to the new max number of nodes, and then disable one gene, and either duplicate 2 previously made genes, or add two new genes.
			num_nodes += 1
			current_pop[xi].disable_gene(ind)
			# append a new gene to the global genes, and then add to the model using .add() as it is new to the model no matter what
			test = True
			# for i in range(len(global_genes)-1): # will check to see if two consecutive (inno nums) genes form a line to the highest node at that point (what happens if I sort it?)
			# # if node already exists
			# 	if global_genes[i].get_node_in() == genes[ind].get_node_in():
			# 		if (global_genes[i+1].get_node_out() == genes[ind].get_node_out()) and (global_genes[i+1].get_inno_num() == global_genes[i].get_inno_num() + 1): # if they have consecutive inno nums, and the same output
			array = node_already_exists(model, genes, ind):
			if array[0]:
				current_pop[xi].add(global_genes[array[1]])
			else: # creating a new gene because global does not have it
				pre_len = len(global_genes)
				n_num = current_pop[xi].get_num_nodes() + 1
				current_pop[xi].set_num_nodes(n_num) # increases the number of local nodes by 1
				global_genes.append(Gene(genes[ind].get_node_in(), n_num, global_genes[pre_len-1].get_inno_num()+1))
				current_pop[xi].add(global_genes[pre_len])

			if array[2]:
				current_pop[xi].add(global_genes[array[3]])
			else:
				pre_len = len(global_genes)
				global_num_nodes += 1
				global_genes.append(Gene(n_num, genes[ind].get_node_out(), global_genes[pre_len-1].get_inno_num()+1))
				current_pop[xi].add(global_genes[pre_len])



def mutate_connections():
	global current_pop
	pass

def model_mutate():
	global current_pop, weight_mutation_rate, mutation_power, gene_mutation_rate, node_mutation_rate

	mutate_weights(weight_mutation_rate, mutation_power) # DONE
	mutate_nodes(node_mutation_rate)
	mutate_connections()


def predict_action(inputs):
	global current_pop
	# The height, dist and pipe_height must be between 0 to 1 (Scaled by SCREENHEIGHT)
	height = min(SCREENHEIGHT, height) / SCREENHEIGHT - 0.5
	dist = dist / 450 - 0.5 # Max pipe distance from player will be 450
	pipe_height = min(SCREENHEIGHT, pipe_height) / SCREENHEIGHT - 0.5
	# neural_input = np.asarray([height, dist, pipe_height])
	# neural_input = np.atleast_2d(neural_input)
	# output_prob = current_pop[model_num].predict(neural_input, 1)[0]
	if output_prob[0] <= 0.5:
		# Perform the jump action
		return 1
	return 0

# Initialize all models
def init_models(num_models, input_num):
	global global_genes, global_num_nodes, current_pop

	global_num_nodes = input_num + 1

	for i in range(num_models):
		model = Model()
		# model.add(Dense(output_dim=3, input_dim=input_num))
		# model.add(Activation("sigmoid"))
		# model.add(Dense(output_dim=1))
		# model.add(Activation("sigmoid"))
		for j in range(input_num):
			if len(global_genes) < input_num:
				global_genes.append(Gene(j, input_num, j))
			model.add(global_genes[j])
			model.set_num_nodes(input_num+1)


		# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # stochastic gradient descent, (lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		# model.compile(loss="mse", optimizer=sgd, metrics=["accuracy"])
		current_pop.append(model)
		fitness.append(-100)

	if load_current_pool:
		current_pop = load_saved()

	for i in range(num_models):
		print(current_pop[i].get_weights())



def main():
	set_up_gui()



if __name__ == '__main__':
	main()



"""
 can't seem to find much documentation on how to interpret the output of get_weights() when running a neural network in Keras. From what I understand, the output is determined by the structure of the network.

model.add(Dense(5, input_dim=2, activation = linear, use_bias=True, kernel_initializer=Orthogonal))
model.add(Dense(1, use_bias=True))
model.compile(loss='mae', optimizer='adam')


The output of get_weights() after training is:

	 [array([[ 0.79376745,  0.79879117,  1.22406125,  1.07782006,  1.24107373],
			 [ 0.88034034,  0.88281095,  1.13124955,  0.98677355,  1.14481246]], dtype=float32), 
	  array([-0.09109745, -0.09036621,  0.0977743 , -0.07977977,  0.10829113], dtype=float32), 
	  array([[-0.72631335],
			 [-0.38004425],
			 [ 0.62861812],
			 [ 0.10909595],
			 [ 0.30652359]], dtype=float32), 
	  array([ 0.09278722], dtype=float32)]

There are a total of four arrays. What does each represent? Thanks!

Weights for the first layer (2 inputs x 5 units)
Biases for the first layer (5 units)
Weights for the second layer (5 inputs x 1 unit)
Biases for the second layer (1 unit)
You can always get by layer too:

for lay in model.layers:
	print(lay.name)
	print(lay.get_weights())



model = Sequential()
		model.add(Dense(output_dim=3, input_dim=input_num))
		model.add(Activation("sigmoid"))
		model.add(Dense(output_dim=1))
		model.add(Activation("sigmoid"))


# activation layers match the prior layer's node count as it is always a 1:1 node in/out
[array([[ 5.2106876 ,  5.1622434 ,  5.5775213 ,  5.639428  ,  4.620723  , 5.5399513 ,  4.990021  ], # layer 1 with 3 inputs going to 7 nodes (checked)
	   [-0.1801055 ,  0.09985326,  0.04459427,  0.6945113 , -0.39618808,-0.46534306,  0.24852274],
	   [-5.038672  , -4.6839995 , -4.5152817 , -4.1976657 , -4.3997526 ,-3.9397857 , -5.041324  ]], dtype=float32), 
array([-0.15964277, -0.54143775, -1.6091541 , -1.4050983 ,  0.18426257, 1.3009388 , -2.1813276 ], dtype=float32), # layer 2 with 1 input from 7 nodes (added sums) going to 7 nodes
array([[-4.593916  ], # layer 3 with 7 inputs going to 1 node
	   [-1.3583922 ],
	   [-2.4457595 ],
	   [-0.30574194],
	   [ 1.7661164 ],
	   [-1.8115075 ],
	   [-1.5253297 ]], dtype=float32), 
array([3.564565], dtype=float32)] # layer 4 with 1 input goint to 1 node


targetSpeciesAmount = 10.2 # target amount of species to have in population to have a decent enough spread of solutions
compatabilityThreshold = 6.0 # averaged how many genes not shared (disjoint) to be considered a different species
compatabilityMod = 0.3 # change in compatability threshold
dropoffAge = 15.0 # after 15 generations with no increase in fitness, penalize the species
survivalThreshold = 0.85 # top 100*(1-x) percent survive and are allowed to reproduce
mutationRateWeights = .80 #mutate weights at 80% rate
mutationRateNodes = .03 # mutate nodes at 3-30% rate
mutationRateLinks = .05 # mutate links at 5%
mutationPower = 2.5 # mutate up to 2.5
disjointRatio = 1.0 # averaged disjoint between two species ratio of 1 disjoint worth average of 3 in weight
excessRatio = 1.0
numSpecies = 1
weightDifferenceRatio = 0.3 * POPULATION_SIZE / 100 # 0.4 -> 3



"""




