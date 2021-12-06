from max_heap import MaxHeap

import copy
import pickle
import torch
import torch.nn as nn
import random
import numpy as np



class Grammar():
    def __init__(self, rules):
        """
        Probabilistic Context Free Grammar (PCFG) over program expressions
        
        rules: mapping from non-terminal symbol to list of productions
        each production is a tuple of (log probability, form)
        where log probability is a float corresponding to the log of the probability that generating from that nonterminal symbol will use that production
        form is either : a tuple of the form (constructor, non-terminal-1, non-terminal-2, ...). `constructor` should be a component in the DSL, such as '+' or '*', which takes arguments
                       : just `constructor`, where `constructor` should be a component in the DSL, such as '0' or 'x', which takes no arguments        
        non-terminals can be anything that can be hashed and compared for equality, such as strings, integers, and tuples of strings/integers
        """
        self.non_terminals = set(rules.keys())
        self.rules = rules

    def pretty_print(self):
        pretty = ""
        for symbol, productions in self.rules.items():
            for probability, form in productions:
                pretty += f"{symbol} ::= "
                if isinstance(form, tuple):
                    pretty += "constructor='"+form[0] + "', args=" + ",".join(map(str,form[1:]))
                else:
                    pretty += "constructor="+form
                pretty += "\tw.p. " + str(probability) + "\n"
        return pretty

    @staticmethod
    def from_components(components, gram):
        """
        Builds and returns a `Grammar` (ie PCFG) from typed DSL components 
        You should initialize the probabilities to be the same for every single rule
        Also takes as input whether we are doing bigrams or unigrams for conditioning the probabilities
        
        gram=1: unigram
        gram=2: bigram
        """

        """
        Suggestion:
        Implement bigrams by defining new nonterminal symbols of the form:
        (parent, argument_index, type)
        where: parent is a component in the DSL, such as '+' or '*'
               argument_index is which argument of the parent is being generated. This is a natural number. For instance, if `parent` is '+', argument_index could be equal to 0 (for the left child of the addition operator) or 1 (for the right child) but could not be equal to 2 (because '+' takes only two arguments)
        
        You still want nonterminal symbols for each type, such as 'int', because these will be the start symbols at the beginning of program generation
        """
        
        assert gram in [1,2]

#starting        

        symbols = { tp for component_type in components.values() for tp in component_type }

        if gram == 1:
            def make_form(name,tp):
                if len(tp) == 1: return name
                assert len(tp) > 1
                return tuple([name] + list(tp[:-1]))
            rules = {symbol: [(0., make_form(component_name,component_type)) for component_name,component_type in components.items() if component_type[-1] == symbol]
                     for symbol in symbols}

        if gram == 2:
            for parent, parent_type in components.items():
                if len(parent_type) == 1: continue # this is not a function, so cannot be a parent
                for argument_index, argument_type in enumerate(parent_type[:-1]):
                    symbols.add( (parent, argument_index, argument_type))

            rules = {}
            for symbol in symbols:
                rules[symbol] = []
                if isinstance(symbol, tuple):
                    return_type = symbol[-1]
                else:
                    return_type = symbol

                for component, component_type in components.items():
                    if component_type[-1] == return_type:
                        if len(component_type) == 1:
                            form = component
                        else:
                            form = tuple([component] + [(component, argument_index, argument_type)
                                                        for argument_index, argument_type in enumerate(component_type[:-1]) ])
                        rules[symbol].append((0., form))
                        
        if True: return Grammar(rules)
#ending
        assert False, "implement as part of homework"

    

    def top_down_generator(self, start_symbol, canonical=None):
        """
        Best-first top-down enumeration of programs generated from the PCFG
        
        start_symbol: a nonterminal in the grammar. Should have: `start_symbol in self.rules.keys()`

        Yields a generator.
        Each generated return value is of the form: (log probability, expression)
        The expressions should be in non-increasing order of (log) probability
        Every expression that can be generated from the grammar should eventually be yielded
        """

        # hint: do best-first search using a priority queue
        # the elements of the priority queue are partially constructed syntax trees.
        # each such partially constructed syntax tree can have nonterminal symbols in it
        # the priority of a partially constructed syntax tree is the log probability of generating it
        # so we begin with the start symbol, which has probability 1, hence log probability 0=log(1).
        
        heap = MaxHeap()
        heap.push(0., start_symbol)
#starting
        def next_non_terminal(syntax_tree):
            for non_terminal in self.rules:
                if non_terminal == syntax_tree:
                    return non_terminal

            if not isinstance(syntax_tree, tuple): # leaf
                return None

            arguments = syntax_tree[1:]
            for argument in arguments:
                argument_next = next_non_terminal(argument)
                if argument_next is not None:
                    return argument_next

            return None # none of the arguments had a next non-terminal symbol to expand
            
        def finished(syntax_tree):
            return next_non_terminal(syntax_tree) is None

        def substitute_next_non_terminal(syntax_tree, expansion):
            for non_terminal in self.rules:
                if non_terminal == syntax_tree:
                    return expansion

            if not isinstance(syntax_tree, tuple): # leaf
                return None # failure

            function = syntax_tree[0]
            arguments = list(syntax_tree[1:])
            performed_substitution = False
            for argument_index, argument in enumerate(arguments):
                argument_new = substitute_next_non_terminal(argument, expansion)
                if argument_new is not None:
                    arguments[argument_index] = argument_new
                    performed_substitution = True
                    break
                
            if performed_substitution:
                return tuple([function] + arguments)
            else:
                return None
#ending
        # keep on removing things from the heap
        # this will give you the highest probability syntax tree that you have built so far
        # yield them if they are a fully completed syntax tree without nonterminals
        # otherwise, we need to fill in more of the syntax tree and add back to the heap
        while not heap.empty():
            log_probability, syntax_tree = heap.pop()
#starting            

            if finished(syntax_tree):
                if canonical is None or canonical(syntax_tree):
                    yield log_probability, syntax_tree
                continue

            non_terminal = next_non_terminal(syntax_tree)

            for production_log_probability, production in self.rules[non_terminal]:
                new_probability = production_log_probability + log_probability
                new_syntax_tree = substitute_next_non_terminal(syntax_tree, production)
                if canonical is None or canonical(new_syntax_tree):
                    assert new_syntax_tree is not None, "should never happen"
                    heap.push(new_probability, new_syntax_tree)

            continue
#ending        
            assert False, "implement as part of homework"


    def normalize(self):
        """
        Destructively modifies grammar so that all of the probabilities sum to one
        Has some extra logic so that if the log probabilities are coming from a neural network, everything is handled properly, but you don't have to worry about that
        """

        def norm(productions):
            z,_ = productions[0]
            if isinstance(z, torch.Tensor):
                z = torch.logsumexp(torch.stack([log_probability for log_probability,_ in productions ]),0)
            else:
                for log_probability,_ in productions[1:]:
                    z = np.logaddexp(z, log_probability)

            return [(log_probability - z, production) for log_probability, production in productions ]
        
        self.rules = {symbol: norm(productions)
                      for symbol, productions in self.rules.items() }
        
        return self

    def jitter(self, j=0.05):
        self.rules = {symbol: [ (lp+random.random()*j, rhs) for lp,rhs in productions ]
                      for symbol, productions in self.rules.items() }
        
        return self.normalize()

    def uniform(self):
        """
        Destructively modifies grammar so that all of the probabilities are uniform across each nonterminal symbol
        """
        self.rules = {symbol: [(0., form) for _, form in productions ]
                      for symbol, productions in self.rules.items() }
        return self.normalize()

    @property
    def n_parameters(self):
        """
        Returns the number of real-valued parameters in the probabilistic grammar
        (technically, this is not equal to the number of degrees of freedom, because we have extra constraints that the probabilities must sum to one across each nonterminal symbol)
        """
        return sum(len(productions) for productions in self.rules.values() )

    def from_tensor(self, tensor):
        """
        Destructively modifies grammar so that the continuous parameters come from the provided pytorch tensor        
        """
        assert tensor.shape[0] == self.n_parameters
        index = 0
        for symbol in sorted(self.rules.keys(), key=str):
            for i in range(len(self.rules[symbol])):
                _, form = self.rules[symbol][i]
                self.rules[symbol][i] = (tensor[index], form)
                index += 1
        assert self.n_parameters == index

    def sample(self, nonterminal):
        """
        Samples a random expression built from the space of syntax trees generated by `nonterminal`
        """

        # productions: all of the ways that we can produce expressions from this nonterminal symbol
        productions = self.rules[nonterminal]

        # sample from multinomial distribution given by log_probabilies in `productions`
        log_probabilities = [log_probability for log_probability, form in productions]
        probabilities = np.exp(np.array(log_probabilities))

        i = np.argmax(np.random.multinomial(1, probabilities))

        _, rule = productions[i]
        if isinstance(rule, tuple):
            # this rule is a function that takes arguments
            constructor, *arguments = rule
            return tuple([constructor] + [self.sample(argument) for argument in arguments ])
        else:
            # this rule is just a terminal symbol
            constructor = rule
            return constructor

    def log_probability(self, nonterminal, expression):
        """
        Returns the logarithm of the probability of sampling `expression` starting from the symbol `nonterminal`
        """
        result=None
        for log_probability, rule in self.rules[nonterminal]:
            if isinstance(expression, tuple) and isinstance(rule, tuple) and expression[0] == rule[0]:
                child_log_probability = sum(self.log_probability(child_symbol, child_expression)
                                            for child_symbol, child_expression in zip(rule[1:], expression[1:]) )
                assert result is None, "ensure that there is always a unique parse"
                result = log_probability + child_log_probability

            if expression == rule:
                assert result is None, "ensure that there is always a unique parse"
                result = log_probability

            if rule in self.non_terminals:
                try:
                    new_result = self.log_probability(rule, expression)
                except: continue
                assert result is None, "ensure that there is always a unique parse"
                result = new_result+log_probability

        # we are going to have to use one of the 
        if result is None:
            assert False, "could not calculate probability of expression giving grammar"
        return result
#starting
    def bottom_up_generator(self, start_symbol):
        non_terminals = set(self.rules.keys())
        heaps = {symbol: MaxHeap() for symbol in non_terminals }
        successors = {symbol: {} for symbol in non_terminals }
        already_seen = {symbol: set() for symbol in non_terminals }

        smallest_programs = {symbol: next(self.top_down_generator(symbol))
                             for symbol in non_terminals }
        for symbol, smallest in smallest_programs.items():
            print("smallest", symbol, "is", smallest)
        print(smallest_programs)
        

        for symbol in non_terminals:
            for rule_probability, rule in self.rules[symbol]:
                if isinstance(rule, tuple):
                    function, *arguments = rule
                    smallest_instantiation = tuple([function] + [smallest_programs[argument][1] for argument in arguments ])
                elif rule in non_terminals:
                    smallest_substantiation = smallest_programs[rule]
                else:
                    smallest_instantiation = rule

                best_probability = self.log_probability(symbol, smallest_instantiation)
                heaps[symbol].push(best_probability, smallest_instantiation)
                already_seen[symbol].add(smallest_instantiation)

        print("successors",successors)
        print("heaps", heaps)
        print()
        print()
        
        def query(symbol, expression):
            nonlocal successors, heaps

            if len(successors[symbol]) == 0 and expression is not None: query(symbol, None)

            if expression in successors[symbol]:
                return successors[symbol][expression]

            if heaps[symbol].empty():
                return None
            
            _, other_expression = heaps[symbol].pop()
            successors[symbol][expression] = other_expression

            if isinstance(other_expression, tuple):
                function, *arguments = other_expression
                argument_symbols = [ p[1:]
                                     for _, p in self.rules[symbol]
                                     if isinstance(p,tuple) and p[0] == function][0]
            else:
                function, arguments, argument_symbols = other_expression, [], []

            for i, (x, argument_symbol) in enumerate(zip(arguments, argument_symbols)):
                assert isinstance(other_expression, tuple)
                new_arguments = list(arguments)
                
                new_arguments[i] = query(argument_symbol, x)
                if new_arguments[i] is None:
                    continue
                
                new_expression = tuple([function] + new_arguments)
                if new_expression not in already_seen[symbol]:
                    already_seen[symbol].add(new_expression)
                    heaps[symbol].push(self.log_probability(symbol, new_expression),
                                       new_expression)
                    #print("\tpopping", other_expression, "forces us to push", new_expression)

            return other_expression

        # compute bottom-up
        # for s in self.non_terminals:
        #     print(s, smallest_programs[s])
        #     print(self.log_probability(s, smallest_programs[s][1]))
        #for symbol in sorted(self.non_terminals,key=lambda s: -smallest_programs[s][0]): query(symbol, None)

        current_program = smallest_programs[start_symbol][1]
        yield self.log_probability(start_symbol, current_program), current_program
        
        while True:
            current_program = query(start_symbol, current_program)
            if current_program is None: return 
            yield self.log_probability(start_symbol, current_program), current_program
#ending

rules={"product": [(1., ("*", "term", "product")),
                   (0., ("I", "term")),],
       "term": [(2., "a"),
                (1., "b")]
}

rules={"product": [(1., ("*", "term", "product")),
                   (0., ("I", "term")),
                   (10., "c")
],
       "term": [(3., ("+", "product", "product")),
                (2., "a"),
                (1., "b")]
}

rules={"summation": [(0., ("+", "product", "summation")),
                      (0., ("I", "product"))],
       "difference": [(0., ("-", "summation", "summation"))],
       "product": [(0., ("*", "term", "product")),
                   (0., ("I", "term")),],
       "term": [(0., ("sigmoid", "leaf")),
                (0., ("tanh", "leaf")),
                (0., ("logSquared", "leaf")),
                (0., ("I", "leaf"))],
       "leaf": [(1., "input"),
                (0., "parameter"),
                (0., "1")]}

g=Grammar(rules).uniform().normalize().jitter(1.)
n=500
st="difference"
for j,(p1, p2) in enumerate(zip(g.bottom_up_generator(st),
                                g.top_down_generator(st))):
    print(j, p1[1], p2[1],
          p1[0], p2[0])
    assert abs(g.log_probability(st, p1[1])-g.log_probability(st, p2[1]))<1e-3
    if j>n:break
print()
assert False
def is_canonical(expression):
    def finished(syntax_tree):
        if isinstance(syntax_tree, tuple):
            return all(finished(a) for a in syntax_tree[1:] )
        return syntax_tree not in g.non_terminals

    
    return True
for j,p in enumerate(g.top_down_generator("product", is_canonical)):
    print("\t", j, p)
    if j>n:break
    
