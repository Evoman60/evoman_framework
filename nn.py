import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return x.clip(0.)  

def tanh(x):
    return (np.exp(x) + np.exp(-x)) / (np.exp(x) - np.exp(-x)) 

class NN(object):
    def __init__(self):
        pass
    
    def forward(self):
        pass 


class MLP(NN):
    def __init__(self, input_size: int, 
                 hidden_layer_num: int, 
                 hidden_layer_size: int, 
                 output_size: int,
                 init_weight: str | np.ndarray = 'random',
                 activation_function: callable = sigmoid) -> None:
        
        #for future changes might be done, not necessary for now        
        super().__init__()

        self.input_size = input_size
        self.hidden_layer_num = hidden_layer_num
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

        # num_var = input layer weight + hidden layer num weight + output layer weight + input+hidden bias + output bias
        self.num_var = input_size * hidden_layer_size + \
                        (hidden_layer_num - 1) * hidden_layer_size * hidden_layer_size + \
                       hidden_layer_size * output_size + \
                       hidden_layer_num * hidden_layer_size + \
                       output_size if self.hidden_layer_num > 0 else input_size * output_size + output_size
                       
        self.weight_list = []
        self.bias_list = []
        self.activation_function = activation_function
        
        #using kaiming init
        if init_weight == 'random':
            if self.hidden_layer_num <= 0:   
                self.weight_list.append(np.random.normal(0, np.sqrt(2 / input_size), (output_size, input_size)))
                self.bias_list.append(np.zeros((output_size)))
            else:
                self.weight_list.append(np.random.normal(0, np.sqrt(2 / input_size), (hidden_layer_size, input_size)))
                self.bias_list.append(np.zeros((hidden_layer_size)))
                for i in range(hidden_layer_num - 1):
                    self.weight_list.append(np.random.normal(0, np.sqrt(2 / hidden_layer_size), (hidden_layer_size, hidden_layer_size)))
                    self.bias_list.append(np.zeros((hidden_layer_size)))
                self.weight_list.append(np.random.normal(0, np.sqrt(2 / hidden_layer_size), (output_size, hidden_layer_size)))
                self.bias_list.append(np.zeros((output_size)))
        elif isinstance(init_weight, np.ndarray):
            assert init_weight.size == self.num_var
            
            if self.hidden_layer_num <= 0:
                self.weight_list.append(init_weight[ :input_size * hidden_layer_size].reshape(output_size, input_size))
                self.bias_list.append(init_weight[input_size * output_size : ])
            else:
                #init as the order we calculate num_var
                weight_split_points = [input_size * hidden_layer_size]
                weight_split_points.extend([weight_split_points[-1] + i * hidden_layer_size * hidden_layer_size for i in range(1, hidden_layer_num)])
                weight_split_points.append(weight_split_points[-1] + hidden_layer_size * output_size)
                weight_vec_list = np.array_split(init_weight[ : weight_split_points[-1]], weight_split_points[ : -1])
                weight_vec_list[0] = weight_vec_list[0].reshape((hidden_layer_size, input_size))
                weight_vec_list[-1] = weight_vec_list[-1].reshape((output_size, hidden_layer_size))
                weight_vec_list[1 : -1] = list(map(lambda array: array.reshape(hidden_layer_size, hidden_layer_size), weight_vec_list[1 : -1]))
                self.weight_list = weight_vec_list
                
                bias_split_point = [hidden_layer_size for _ in range(hidden_layer_num)]
                self.bias_list = np.array_split(init_weight[weight_split_points[-1] : ], bias_split_point)
        
    def set_weight(self, init_weight: np.ndarray):
        assert init_weight.size == self.num_var
        
        input_size = self.input_size
        hidden_layer_size = self.hidden_layer_size
        hidden_layer_num = self.hidden_layer_num
        output_size = self.output_size
        
        if self.hidden_layer_num <= 0:
            self.weight_list.append(init_weight[ :input_size * hidden_layer_size].reshape(output_size, input_size))
            self.bias_list.append(init_weight[input_size * output_size : ])
        else:
            #init as the order we calculate num_var
            weight_split_points = [input_size * hidden_layer_size]
            weight_split_points.extend([weight_split_points[-1] + i * hidden_layer_size * hidden_layer_size for i in range(1, hidden_layer_num)])
            weight_split_points.append(weight_split_points[-1] + hidden_layer_size * output_size)
            weight_vec_list = np.array_split(init_weight[ : weight_split_points[-1]], weight_split_points[ : -1])
            weight_vec_list[0] = weight_vec_list[0].reshape((hidden_layer_size, input_size))
            weight_vec_list[-1] = weight_vec_list[-1].reshape((output_size, hidden_layer_size))
            weight_vec_list[1 : -1] = list(map(lambda array: array.reshape(hidden_layer_size, hidden_layer_size), weight_vec_list[1 : -1]))
            self.weight_list = weight_vec_list
                
            bias_split_point = [hidden_layer_size for _ in range(hidden_layer_num)]
            self.bias_list = np.array_split(init_weight[weight_split_points[-1] : ], bias_split_point)                   
    
    def set(self, set_arg: NN | np.ndarray):
        if isinstance(set_arg, NN):
            self = set_arg
        elif isinstance(set_arg, np.ndarray):
            self.set_weight(set_arg)
        else:
            raise NotImplementedError
    
    def forward(self, x):
        hidden_state = x
        for weight, bias in zip(self.weight_list, self.bias_list): 
            hidden_state = weight.dot(hidden_state)
            hidden_state += bias
            hidden_state = self.activation_function(hidden_state)
        output = hidden_state
        return output
    
    @classmethod
    def like(cls, mlp):
        return cls(input_size = mlp.input_size, 
                   hidden_layer_num = mlp.hidden_layer_num, 
                   hidden_layer_size = mlp.hidden_layer_size, 
                   output_size = mlp.output_size,
                   init_weight = 'random',
                   activation_function = mlp.activation_function)
    
    def get_weights_as_vec(self):
        return np.concat(list(map(lambda x: x.flatten(),self.weight_list)) + self.bias_list)
    
    
                
            
        
        
        