//神经网络 + 遗传算法调整权重

/**
 * Neuroevolution类
 */
var Neuroevolution = function (options) {
	var self = this;

	// Declaration of module parameters (options) and default values
	self.options = {
		//Logistic activation function.
		activation: function (a) {
			var ap = (-a) / 1;
			return (1 / (1 + Math.exp(ap)));
		},


		// Returns a random value between -1 and 1
		randomClamped: function () {
			return Math.random() * 2 - 1;
		},

		// various factors and parameters (along with default values)
		network: [1, [1], 1], //  network structure ([input_dim, [hidden0_dim, hidden1_dim, ...], output_dim])
		population: 50, // Population by generation
		elitism: 0.2, // Best networks kepts unchanged for the next generation (rate)
		randomBehaviour: 0.2, // New random networks for the next generation(rate)
		mutationRate: 0.1, // Mutation rate on the weights of synapses
		mutationRange: 0.5, // Interval of the mutation changes on the synapse weight
		historic: 0, // Latest generations saved
		lowHistoric: false, // Only save score (not the network)
		scoreSort: -1, // Sort order (-1 = 降序, 1 = 升序)
		nbChild: 1 // Number of children by breeding

	}


	// Override default options
	self.set = function (options) {
		for (var i in options) {
			if (this.options[i] !== undefined) {
				self.options[i] = options[i];
			}
		}
	}

	// Overriding default options with the pass in options
	self.set(options);


	/**
	 * Neuron类
	 */
	var Neuron = function () {
		this.value = 0;
		this.weights = [];
	}

	// 初始化权值矩阵
    // nb : number of inputs
	Neuron.prototype.populate = function (nb) {
		this.weights = [];
		for (var i = 0; i < nb; i++) {
			// 使每个神经元的权重被初始化为[-1,1]间的一个值，防止梯度过慢或梯度爆炸
			this.weights.push(self.options.randomClamped());
		}
	}


	/**
	 * Layer类
	 */
	var Layer = function (index) {
		this.id = index || 0;
		this.neurons = [];
	}

	// 初始化神经元
	// nbNeurons: number of neurons, nbInputs: number of inputs
	Layer.prototype.populate = function (nbNeurons, nbInputs) {
		this.neurons = [];
		for (var i = 0; i < nbNeurons; i++) {
			var n = new Neuron();
			n.populate(nbInputs);
			this.neurons.push(n);
		}
	}


	/**
	 * Neural Network类
	 */
	var Network = function () {
		this.layers = [];
	}

	/**
	 * 输入网络层参数 --> 搭建神经网络
	 */
	Network.prototype.perceptronGeneration = function (input, hiddens, output) {
		var index = 0;
		var previousNeurons = 0;
		var layer = new Layer(index); // Layer类已经被封装好了，直接调用populate方法就可以初始化一个神经元层
		layer.populate(input, previousNeurons); // Number of Inputs will be set to
		// 0 since it is an input layer.
		previousNeurons = input; // number of input is size of previous layer.
		this.layers.push(layer);
		index++;
		for (var i in hiddens) {
			// Repeat same process as first layer for each hidden layer.
			var layer = new Layer(index);
			layer.populate(hiddens[i], previousNeurons);
			previousNeurons = hiddens[i];
			this.layers.push(layer);
			index++;
		}
		var layer = new Layer(index);
		layer.populate(output, previousNeurons); // Number of input is equal to
		// the size of the last hidden
		// layer.
		this.layers.push(layer);
	}


	/**
	 * 保存神经网络参数(layers, weights)
	 */
	Network.prototype.getSave = function () {
		var datas = {
			neurons: [], // Number of Neurons per layer.
			weights: [] // Weights of each Neuron's inputs.
		};

		for (var i in this.layers) {
			datas.neurons.push(this.layers[i].neurons.length);
			for (var j in this.layers[i].neurons) {
				for (var k in this.layers[i].neurons[j].weights) {
					// push all input weights of each Neuron of each Layer into a flat
					// array.
					datas.weights.push(this.layers[i].neurons[j].weights[k]);
				}
			}
		}
		return datas;
	}

	/**
	 * 设置参数为已保存的格式类型(neurons, weights)
	 */
	Network.prototype.setSave = function (save) {
		var previousNeurons = 0;
		var index = 0;
		var indexWeights = 0;
		this.layers = [];
		for (var i in save.neurons) {
			// Create and populate layers.
			var layer = new Layer(index);
			layer.populate(save.neurons[i], previousNeurons);
			for (var j in layer.neurons) {
				for (var k in layer.neurons[j].weights) {
					// Apply neurons weights to each Neuron.
					layer.neurons[j].weights[k] = save.weights[indexWeights];

					indexWeights++; // Increment index of flat array.
				}
			}
			previousNeurons = save.neurons[i];
			index++;
			this.layers.push(layer);
		}
	}


	/**
	 * forward propagation前向传播计算
	 */
	Network.prototype.compute = function (inputs) {
		// Set the value of each Neuron in the input layer.
		for (var i in inputs) {
			if (this.layers[0] && this.layers[0].neurons[i]) {
				this.layers[0].neurons[i].value = inputs[i];
			}
		}

		var prevLayer = this.layers[0]; // Previous layer is input layer.
		// forward propagation
		for (var i = 1; i < this.layers.length; i++) {
			for (var j in this.layers[i].neurons) {
				// For each Neuron in each layer.
				var sum = 0;
				for (var k in prevLayer.neurons) {
					// Every Neuron in the previous layer is an input to each Neuron in the next layer
					// 第i-1层第k个神经元 --> 第i层第j个神经元的映射
					sum += prevLayer.neurons[k].value *
						this.layers[i].neurons[j].weights[k];
				}

			// Compute the activation of the Neuron.
				this.layers[i].neurons[j].value = self.options.activation(sum);
			}
			prevLayer = this.layers[i];
		}

		// All outputs of the Network.
		var out = [];
		var lastLayer = this.layers[this.layers.length - 1];
		for (var i in lastLayer.neurons) {
			out.push(lastLayer.neurons[i].value);
		}
		return out;
	}


	/**
	 * Genome类（基因型类）
	 */
	var Genome = function (score, network) {
		this.score = score || 0;
		this.network = network || null;
	}


	/**
	 * Generation类（当前代）
	 */
	var Generation = function () {
		// 当前代基因型
		this.genomes = [];
	}

	/**
	 * 向该代中添加基因型
	 */
	Generation.prototype.addGenome = function (genome) {
		// Locate position to insert Genome into.
		// The gnomes should remain sorted.
		for (var i = 0; i < this.genomes.length; i++) {
			// Sort in descending order.
			if (self.options.scoreSort < 0) {
				if (genome.score > this.genomes[i].score) {
					break;
				}
				// Sort in ascending order.
			} else {
				if (genome.score < this.genomes[i].score) {
					break;
				}
			}

		}

		// Insert genome into correct position
		this.genomes.splice(i, 0, genome);
	}

	/**
	 * 交配 + 变异
	 */
	Generation.prototype.breed = function (g1, g2, nbChilds) {
		var datas = [];
		for (var nb = 0; nb < nbChilds; nb++) {
			// Deep clone of genome 1.
			var data = JSON.parse(JSON.stringify(g1));
			for (var i in g2.network.weights) {
				// Genetic crossover
				// 0.5 is the crossover factor.
				if (Math.random() <= 0.5) {
					data.network.weights[i] = g2.network.weights[i];
				}
			}

			// Perform mutation on some weights.
			for (var i in data.network.weights) {
				// mutationRate为发生变异的比例
				// mutationRange为发生变异的幅度
				if (Math.random() <= self.options.mutationRate) {
					data.network.weights[i] += Math.random() *
						self.options.mutationRange *
						2 -
						self.options.mutationRange;
				}
			}
			datas.push(data);
		}

		return datas;
	}

	/**
	 * 产生子代
	 */
	Generation.prototype.generateNextGeneration = function () {
		var nexts = [];

		for (var i = 0; i < Math.round(self.options.elitism *
				self.options.population); i++) {
			if (nexts.length < self.options.population) {
				// Push a deep copy of ith Genome's Nethwork.
				nexts.push(JSON.parse(JSON.stringify(this.genomes[i].network)));
			}
		}

		for (var i = 0; i < Math.round(self.options.randomBehaviour *
				self.options.population); i++) {
			var n = JSON.parse(JSON.stringify(this.genomes[0].network));
			for (var k in n.weights) {
				n.weights[k] = self.options.randomClamped();
			}
			if (nexts.length < self.options.population) {
				nexts.push(n);
			}
		}

		var max = 0;
		while (true) {
			for (var i = 0; i < max; i++) {
				// Create the children and push them to the nexts array.
				var childs = this.breed(this.genomes[i], this.genomes[max],
					(self.options.nbChild > 0 ? self.options.nbChild : 1));
				for (var c in childs) {
					nexts.push(childs[c].network);
					if (nexts.length >= self.options.population) {
						// Return once number of children is equal to the
						// population by generatino value.
						return nexts;
					}
				}
			}
			max++;
			if (max >= this.genomes.length - 1) {
				max = 0;
			}
		}
	}


	/**
	 * Generations类
	 * 存储至今产生过的所有代
	 */
	var Generations = function () {
		this.generations = [];
		var currentGeneration = new Generation();
	}

	/**
	 * 产生第一代（初始父代）
	 */
	Generations.prototype.firstGeneration = function () {

		var out = [];
		for (var i = 0; i < self.options.population; i++) {
			// Generate the Network and save it.
			var nn = new Network();
			nn.perceptronGeneration(self.options.network[0],
				self.options.network[1],
				self.options.network[2]);
			out.push(nn.getSave());
		}

		this.generations.push(new Generation());
		return out;
	}

	/**
	 * 基于父代产生子代
	 */
	Generations.prototype.nextGeneration = function () {
		if (this.generations.length == 0) {
			// Need to create first generation.
			return false;
		}

		var gen = this.generations[this.generations.length - 1]
			.generateNextGeneration();
		this.generations.push(new Generation());
		return gen;
	}

	/**
	 * 向最后一代添加基因型
	 */
	Generations.prototype.addGenome = function (genome) {
		// Can't add to a Generation if there are no Generations.
		if (this.generations.length == 0) return false;

		return this.generations[this.generations.length - 1].addGenome(genome);
	}


	self.generations = new Generations();

	/**
	 * 重置所有代
	 */
	self.restart = function () {
		self.generations = new Generations();
	}

	/**
	 * 总体：创建初始父代后每一次迭代
	 */
	self.nextGeneration = function () {
		var networks = [];

		if (self.generations.generations.length === 0) {
			// If no Generations, create first.
			networks = self.generations.firstGeneration();
		} else {
			// Otherwise, create next one.
			networks = self.generations.nextGeneration();
		}

		// Create Networks from the current Generation.
		var nns = [];
		for (var i in networks) {
			var nn = new Network();
			nn.setSave(networks[i]);
			nns.push(nn);
		}

		if (self.options.lowHistoric) {
			// Remove old Networks.
			if (self.generations.generations.length >= 2) {
				var genomes =
					self.generations
					.generations[self.generations.generations.length - 2]
					.genomes;
				for (var i in genomes) {
					delete genomes[i].network;
				}
			}
		}

		if (self.options.historic != -1) {
			// Remove older generations.
			if (self.generations.generations.length > self.options.historic + 1) {
				self.generations.generations.splice(0,
					self.generations.generations.length - (self.options.historic + 1));
			}
		}

		return nns;
	}

	/**
	 * 使用特定的score和基因型构建一代
	 */
	self.networkScore = function (network, score) {
		self.generations.addGenome(new Genome(score, network.getSave()));
	}
}
