class Gene(object):
	node_in = 0
	node_out = 1
	innovation_num = 0

	def get_inno_num(self):
		return self.innovation_num

	def get_node_in(self):
		return self.node_in

	def get_node_out(self):
		return self.node_out


	"""docstring for ClassName"""
	def __init__(self, node_in, node_out, innovation_num):
		# super(ClassName, self).__init__()
		self.node_in = node_in
		self.node_out = node_out
		self.innovation_num = innovation_num
		