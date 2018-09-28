class Model(object):

	genes = []
	weights = []
	enabled_list = []
	num_nodes = 0

	def add(self, gene): # assume new gene for this model and for global
		self.genes.append(gene)
		self.enabled_list.append(True)
		self.weights.append(random.uniform(0, 1))

	def sigmoid(x):
		return (1 / (1 + math.exp(-x)))

	def duplicate(self, gene, enabled, weight): # assume new gene for this model, BUT (not for global as the values for enabled and weights are already set) AND (you are copying values over from another gene)
		self.genes.append(gene)
		self.enabled_list.append(enabled)
		self.weights.append(weight)

	def get_weights(self):
		return self.weights

	def get_genes(self):
		return self.genes

	def get_enabled_list(self):
		return self.enabled_list

	def get_num_nodes(self):
		return self.num_nodes

	def set_num_nodes(self, num):
		self.num_nodes = num

	def disable_gene(self, ind):
		self.enabled_list[ind] = False

	def predict(self, inputs, num_inputs): # assume pre-ordered genes and weights list based upon input numbers
		sums = [0 for i in range(len(genes))] # each 0 in this list represents a node's output
		for i in range(num_inputs):
			sums[i] += inputs[i]
		for i in range(1, len(genes)): # starts from the first genes and will put all others through a sigmoid function
			if genes[i].get_output_node() > genes[i-1].get_output_node(): # if I have moved onto a gene with the next ordered output node, go back and run the other through a sigmoid before anything else happens
				sums[genes[i-1].get_output_node()] = sigmoid(sums[genes[i-1].get_output_node()])
			if enabled_list[i]:
				sums[genes[i].get_output_node()] += weights[i] * sums[genes[i].get_input_node()]


	"""docstring for ClassName"""
	def __init__(self):
		# super(ClassName, self).__init__()
		pass
		