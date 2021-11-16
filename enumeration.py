import itertools
from sympy import *




class Expression():

    def evaluate(self, environment):
        assert False, "not implemented"

    def arguments(self):
        assert False, "not implemented"

    def cost(self):
        return 1 + sum([0] + [argument.cost() for argument in self.arguments()])

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self): return hash(str(self))

    def __ne__(self, other): return str(self) != str(other)

    def __gt__(self, other): return str(self) > str(other)

    def __lt__(self, other): return str(self) < str(other)

    def free_variables(self):
        return {fv for a in self.arguments() for fv in a.free_variables()  }

class Number(Expression):
    return_type = "int"
    argument_types = []
    
    def __init__(self, n):
        self.n = n

    def __str__(self):
        return f"Number({self.n})"

    def pretty_print(self):
        return str(self.n)

    def evaluate(self, environment):
        return self.n

    def extension(self):
        return [self]

    def arguments(self): return []



class Vector(Expression):
    return_type = "vector"
    argument_types = []
    
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"Vector('{self.name}')"

    def pretty_print(self):
        return self.name

    def evaluate(self, environment):
        if self.name in environment: return environment[self.name]
        
        return symbols(self.name, positive=True, real=True)

    def arguments(self): return []
    
    def free_variables(self): return {self.name}

    

class Matrix(Expression):
    return_type = "matrix"
    argument_types = []
    
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"Matrix('{self.name}')"

    def pretty_print(self):
        return self.name

    def evaluate(self, environment):
        if self.name in environment: return environment[self.name]
        
        return symbols(self.name, positive=True, real=True)

    def arguments(self): return []

    def free_variables(self): return {self.name}


class Plus(Expression):
    return_type = "vector"
    argument_types = ["vector","vector"]
    
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Plus({self.x}, {self.y})"

    def pretty_print(self):
        return f"(+ {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        
        return x + y

    def arguments(self): return [self.x, self.y]

class MatrixMultiply(Expression):
    return_type = "vector"
    argument_types = ["matrix","vector"]
    
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"MatrixMultiply({self.x}, {self.y})"

    def pretty_print(self):
        return f"(@ {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        
        return x * y

    def arguments(self): return [self.x, self.y]

class Sigmoid(Expression):
    return_type = "vector"
    argument_types = ["vector"]
    
    def __init__(self, x):
        self.x=x

    def __str__(self):
        return f"Sigmoid({self.x})"

    def pretty_print(self):
        return f"Sigmoid({self.x.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        if "sympy.core" in str(x.__class__):
            #import pdb; pdb.set_trace()
            
            return symbols('S', cls=Function)(x)
        return x

    def arguments(self): return [self.x]

class Tanh(Expression):
    return_type = "vector"
    argument_types = ["vector"]
    
    def __init__(self, x):
        self.x=x

    def __str__(self):
        return f"Tanh({self.x})"

    def pretty_print(self):
        return f"Tanh({self.x.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        if "sympy.core" in str(x.__class__):
            return tanh(x)
        return x

    def arguments(self): return [self.x]
    
class Minus(Expression):
    return_type = "vector"
    argument_types = ["vector","vector"]
    
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"(- {self.x}, {self.y})"

    def pretty_print(self):
        return f"(- {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        
        return x - y

    def arguments(self): return [self.x, self.y]


class Times(Expression):
    return_type = "vector"
    argument_types = ["vector","vector"]
    
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Times({self.x}, {self.y})"

    def pretty_print(self):
        return f"(* {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        
        return x * y

    def arguments(self): return [self.x, self.y]




def bottom_up_generator(global_bound, operators, constants, behavior):
    """
    global_bound: int. an upper bound on the size of expression
    operators: list of classes, such as [Times, If, ...]
    constants: list of possible leaves in syntax tree, such as [Number(1)]. Variables can also be leaves
    behavior: function mapping program to something that can be hashed/compared
    
    yields: sequence of programs, ordered by expression size, which are semantically distinct according to behavior
    """

    # suggested first thing: variables and constants should be treated the same, because they are both leaves in syntax trees
    # after computing `variables_and_constants`, you should no longer refer to `constants`. express everything in terms of `variables_and_constants`
    # `make_variable` is just a helper function for making variables that smartly wraps the variable name in the correct class depending on the type of the variable
    
    variables_and_constants = constants

    # suggested data structure (you don't have to use this if you don't want):
    # a mapping from a tuple of (type, expression_size) to all of the possible values that can be computed of that type using an expression of that size
    observed_values = set()

    enumerated_expressions = {}
    def record_new_expression(expression, size):
        """Returns True iff the semantics of this expression has never been seen before"""
        nonlocal observed_values

        # calculate what values are produced on these inputs
        values = behavior(expression)

        # is this something we have not seen before?
        if values is not None and values not in observed_values: 
            observed_values.add(values)

            # we have some new behavior
            key = (expression.__class__.return_type, size)

            if key not in enumerated_expressions:
                enumerated_expressions[key] = []

            enumerated_expressions[key].append( (expression, values) )

            return True

        return False
            
    for terminal in variables_and_constants:
        if record_new_expression(terminal, 1): yield terminal
    
    for target_size in range(2, global_bound + 1): # enumerate programs of increasing size
        for operator in operators:
            partitions = integer_partitions(target_size - 1 - len(operator.argument_types),
                                            len(operator.argument_types))
            for argument_sizes in partitions:
                actual_argument_sizes = [sz+1 for sz in argument_sizes]
                candidate_arguments = [enumerated_expressions.get(type_and_size, [])
                                       for type_and_size in zip(operator.argument_types, actual_argument_sizes)]
                for arguments in itertools.product(*candidate_arguments):
                    new_expression = operator(*[e for e,v in arguments ])
                    if record_new_expression(new_expression, target_size):
                        yield new_expression
    return 

def integer_partitions(target_value, number_of_arguments):
    """
    Returns all ways of summing up to `target_value` by adding `number_of_arguments` nonnegative integers
    You may find this useful when implementing `bottom_up_generator`:

    Imagine that you are trying to enumerate all expressions of size 10, and you are considering using an operator with 3 arguments.
    So the total size has to be 10, which includes +1 from this operator, as well as 3 other terms from the 3 arguments, which together have to sum to 10.
    Therefore: 10 = 1 + size_of_first_argument + size_of_second_argument + size_of_third_argument
    Also, every argument has to be of size at least one, because you can't have a syntax tree of size 0
    Therefore: 10 = 1 + (1 + size_of_first_argument_minus_one) + (1 + size_of_second_argument_minus_one) + (1 + size_of_third_argument_minus_one)
    So, by algebra:
         10 - 1 - 3 = size_of_first_argument_minus_one + size_of_second_argument_minus_one + size_of_third_argument_minus_one
    where: size_of_first_argument_minus_one >= 0
           size_of_second_argument_minus_one >= 0
           size_of_third_argument_minus_one >= 0
    Therefore: the set of allowed assignments to {size_of_first_argument_minus_one,size_of_second_argument_minus_one,size_of_third_argument_minus_one} is just the integer partitions of (10 - 1 - 3).
    """

    if target_value < 0:
        return []

    if number_of_arguments == 1:
        return [[target_value]]

    return [ [x1] + x2s
             for x1 in range(target_value + 1)
             for x2s in integer_partitions(target_value - x1, number_of_arguments - 1) ]

def behavior(e):
    global variable_names
    environments=[{n: symbols(n, positive=True, real=True)*s
                   for s,n in zip(signs, variable_names) }
                   for signs in itertools.product(*[[-1,1] for _ in variable_names ]) ]
    
    #print(environments)
    #try:
    b=[simplify(e.evaluate(environment))
           for environment in environments ]
    #except:
    #    return None

    b.sort(key=str)
    b=tuple(b)
    #import pdb; pdb.set_trace()
    arguments=e.arguments()
    if arguments:
        if len({"z"}&e.free_variables())==0:
            return None
    return b

operators=[Times, Plus, Minus, Tanh, Sigmoid, MatrixMultiply]
variables=[Vector("z"), Matrix("m"), Vector("b")]
leaves=variables#+[Number(1)]
variable_names=[v.name for v in variables ]

j=0
for i, e in enumerate(bottom_up_generator(10, operators, leaves, behavior)):
    v = e.evaluate([])
    try:
        variables = v.free_symbols
    except:
        variables = set()
    if set(symbols("z m b", positive=True, real=True)) <= variables:
        j+=1
        print(j, i, e.pretty_print(), v, variables)
        
    if i>1000:
        #import pdb; pdb.set_trace()
        
        break
