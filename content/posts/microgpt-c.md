+++
date        = '2026-04-08T06:53:50+05:30'
draft       = false
title       = '`microgpt-c` Part 1: Automatic Differentiation'
tags        = ["llms", "c"] 
description  = "Creating a Backprop Engine for [`microgpt-c`](https://github.com/agastyapatri/microgpt-c/)" 
+++
[microgpt](https://karpathy.github.io/2026/02/12/microgpt/) is a 200 line python implementation of GPT-2. It is trained on a set of names, and attempts to train a model to generate new ones. Following this, I wanted to try porting it to C as an exercise in learning the language. **This post will lay out my approach to implementing an autodifferentiation engine with nothing but the C standard library**.  It's recommended that one reads the original post, along with [growingswe's visual explainer](https://growingswe.com/blog/microgpt) for more background on GPT-2 itself.

The full code for this project can be found on [Github](https://github.com/agastyapatri/microgpt-c) 

##  Scalar Valued [Reverse Mode Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation:~:text=0%3B%20%7D-,Reverse%20accumulation) 
The (admittedly handwavy) recipe for implementing your own backprop engine is simple: 
1.  Wrap your data in a struct. In this case, it is `struct ad_value`:
    
    ```c
    typedef struct ad_value {
        OPTYPE op;
        float data;
        float grad;
        int ref_count;
        bool is_param;
        bool visited;
        struct ad_value* previous[NUM_PREVS]; 
    } ad_value;
    ```
2.  Provide a way to do some math on the `ad_value`s.
3.  Track the operation that was used to create a `ad_value`, along with its parents `previous[0]` and `previous[1]`
4.  Provide a way to calculate local gradients at a node depending on the values of it's child nodes and the operations that yielded them. 
5.  Sort your computational graph, formed after math done sequentially with `ad_value_<math>`.
6.  Step through the sorted graph and propagate the local gradient starting from the root node, which in most cases is a scalar valued loss function. 

Most C specific minutiae crop up from manually having to manage the lifetime and memory of the nodes in the computational graph. Let's talk about it. 

##  Creation and Destruction of `ad_value`
Here's how `ad_value` is created: 

```c
ad_value* ad_value_alloc(float data){
    ad_value* a = (ad_value*)malloc(sizeof(ad_value));
    if(!a)  return NULL;
    a->data = data; 
    a->is_param = false;
    a->grad = 0;
    a->op = NONE;
    a->previous[0] = NULL;
    a->previous[1] = NULL;
    a->ref_count = 1;
    a->visited = false;
    return a;
}
```
You'll notice that **all `ad_value` instances are heap allocated.** This is done for a few reasons: 

*   The computational graph is dynamic. Each document - name instance - is of a different length, meaning that the computational graph constructed for that document is of a different size than the instance before or after it. You can't stack allocate something you don't know the size of. 
*   `ad_value` instances - nodes in the computational graph - need to survive beyond the scope that they were created in. Heap allocated memory lasts until it is freed or the program terminates.
*   Computation graphs require an expression of shared ownership. Heap allocations provide that.

Understandably, having thousands of `malloc` and `free` calls is not something anyone prefers. Arena allocators would speed things up significantly. For simplicity's sake however, I've kept it as is. 

The `ref_count` parameter is my humble attempt at implementing the managing of `ad_value` lifetimes with reference counting. The idea is that each `ad_value` keeps track of how many other `ad_value`s refer to it. When the reference counter hits 0, the `ad_value` memory is freed. The main reason to do this is that accidentally freeing a random `ad_value` in the computational graph and trying to backpropagate through it would be catastrophic; the infamous "use-after-free". This would also be a nightmare to debug for a novice C programmer. This is slightly different from the textbook definition of the technique, in which a resource is shared among multiple objects which point to it, and is freed when the reference counter hits 0. 

`ad_value`s are freed recursively: 
```c
void ad_value_free(ad_value* val){
	val->ref_count--;
	if(val->ref_count == 0){
		if(val->previous[0])
			ad_value_free(val->previous[0]);
		if(val->previous[1])
			ad_value_free(val->previous[1]);
		free(val);
	}
}
```
Simply calling the function `ad_value_free` decrements the target's reference counter - one less ad_value now refers to it. If no other `ad_value`s are referring to it, it is let go. This triggers an `ad_value_free` call for the parent nodes as well, because a reference to the parent is being freed. To understand this better, one needs to understand how a link between two `ad_value`s is formed.

##  `ad_value` Mathematics
One of the primary assumptions of automatic differentiation (autodiff) is that any function represented programmatically can be decomposed into a sequence of more primitive "atomic" calculations. We need to make sure that the library of mathematical functions for `ad_value` enables atomicity and the use of the [chain rule](https://en.wikipedia.org/wiki/Chain_rule). 

To that end, I've chosen to represent these atomic operations as an `enum`: 

```c
typedef enum {
    NONE,
    ADD,
    SUB,
    MUL,
    DIV,
    POW, 
    SIGMOID,
    NEG, 
    TANH,
    RELU,
    SIN,
    COS,
    LOG, 
    EXP, 
} OPTYPE;
```
Any `ad_value` created must be a result of other `ad_value`s being operated on, or be created as a leaf node with `op = NONE`. The atomic operations created for each field in the `enum OPTYPE` follow a very similar pattern: 
```c 
ad_value* ad_value_add(ad_value* inp1, ad_value* inp2){
    // a new heap-allocated `ad_value`
    ad_value* out = ad_value_alloc(inp1->data + inp2->data);        

    // tracking the operation which led to it  
    out->op = ADD;                                                  

    // tracking it's parent ad_values
    out->previous[0] = inp1;                                        
    out->previous[1] = inp2;

    // incrementing the reference counter of both the parents
    out->previous[0]->ref_count++;                                  
    out->previous[1]->ref_count++;
    return out;
}
```
Of course, for additional flexibility in the control of the lifetime of the `ad_value`'s, It would've been prudent to provide functions like `void ad_value_add(ad_value* inp1, ad_value* inp2, ad_value* out)`, but that isn't something I found to be particularly useful at this time. 
The current set of operations in `enum OPTYPE` was more than enough to be able to calculate more complicated functions like softmax or root mean squared norm. More on that later.

##  `ad_matrix` and `state_dict`
Ultimately, neural nets interpret their weights and biases as matrices; `struct ad_matrix` is defined as a thin wrapper around `ad_value`: 

```c
typedef struct ad_matrix{
	uint rows; 			//  rows  
	uint cols;  		//  cols 
	uint size;			//  rows*cols
	uint nbytes;        //  rows*cols*sizeof(ad_value*)
	ad_value** data;    //  flat representation
} ad_matrix;
```
I prefer to think of this simply as an array of `ad_value` instances instead of a matrix-valued autodifferentiation solution. Each element of an `ad_matrix` is a separate parameter in the neural net, which needs to be updated independently. A true matrix implementation would be leagues faster. We're now ready to actually start building the network.


`ad_matrix` instances are created by `ad_matrix_alloc`: 

```c
ad_matrix* ad_matrix_alloc(uint nrows, uint ncols){
	uint size = nrows * ncols; 
	ad_matrix* m = (ad_matrix*)malloc(sizeof(ad_matrix));
	if(!m){
		return NULL;
	}
	AD_MATRIX_SHAPE_INIT(m, nrows, ncols);
	m->nbytes =size * sizeof(ad_value*); 
	m->data = (ad_value**)malloc(size * sizeof(ad_value*));
	if(!m->data){
		free(m);
		return NULL;
	}
	for(uint i = 0; i < size; i++){
		m->data[i] = ad_value_alloc(0);
	} 
	return m;
}
```
`ad_matrix_alloc` creates an initial set of `ad_value`s, all initialized with `value = 0` and `op = NONE`. 

PyTorch's `nn.Module` has parameters which can be accessed via the `nn.Module.parameters()` attribute. PyTorch also provides a `nn.Module.state_dict()` which maps each layer to it's parameter tensor. Karpathy refers to this as the knowledge of the model.
The original microgpt replicates this behaviour with a `state_dict` dictionary with the signature: 
```python
state_dict = {
    "wte":matrix,               #   token embeddings 
    "wpe":matrix,               #   position embeddings
    "lm_head":matrix,           #   language model head
    "layeri.attn_wq":matrix,    #   query matrix of the ith layer
    "layeri.attn_wk":matrix,    #   key matrix of the ith layer
    "layeri.attn_wv":matrix,    #   value matrix of the ith layer  
    "layeri.attn_wo":matrix,    #   output matrix of the ith layer 
    "layeri.mlp_fc1":matrix,    #   fully connected matrix of the ith layer 
    "layeri.mlp_fc2":matrix,    #   fully connected matrix of the ith layer 
    .... }
```
C doesn't have hash tables, we have to fall back to the trusty struct + indexing method. The layer-wise parameters are implemented as arrays of `ad_matrix` pointers. `state_dict["layer0.attn_wv"]` becomes `state_dict.attn_wv[i]`. The layer-wise matrices in `state_dict` are implemented as arrays of `ad_matrix*`, and take the form of `ad_matrix**`. Dereferencing at the desired layer index will yield the desired `ad_matrix*`
```c
typedef struct state_dict{
	size_t 		embd_dim;		//	embedding dimensions 
	size_t 		num_heads;		//	number of attention heads 
	size_t 		num_layers;		//	number of layers 
	size_t 		block_size;		//	maximum sequence length
	size_t 		head_dim;		//	dimension of each head
	ad_matrix*  wte; 			//	weight token embedding; each row is the learned vector representation of one token 
	ad_matrix*  wpe; 			//	tells the model where in the sequence each token is  
	ad_matrix*  lm_head; 		//  language model  head	
	ad_matrix** attn_wq;		//	array of query weight matrices for each layer
	ad_matrix** attn_wk;		// 	array of key weight matrices for each layer	
	ad_matrix** attn_wv;   		// 	array of value weight matrices for each layer	 
	ad_matrix** attn_wo;		// 	array of output matrices for each layer
	ad_matrix** mlp_fc1;		// 	array of fully connected portions of each layer
	ad_matrix** mlp_fc2;		// 	array of fully connected portions of each layer
} state_dict;
```
You'd have noticed the `is_param` flag in the `ad_value` struct; it notes which `ad_value`s are coming from the model's `state_dict` so that they're not freed after the backward pass. 



### `ad_matrix` Mathematics
To build the GPT-2, we need functions that calculate the softmax and RMS Norm of a vector of `ad_value` nodes. This has to be done in a way that makes sure the correct connections are made between new `ad_value`s and the inputs. Consider `ad_matrix_softmax`: 

```c
ad_matrix* ad_matrix_softmax(ad_matrix* x){
	assert(x!=NULL);
	ad_matrix* out = ad_matrix_alloc(x->rows, x->cols);
	ad_value* xmax = ad_value_alloc(ad_matrix_max(x));
	for(uint i = 0; i < x->size; i++){
		ad_value_free(out->data[i]);
		out->data[i] = ad_value_sub(x->data[i], xmax);
	}
	for(uint i = 0; i < x->size; i++){
		out->data[i] = ad_value_exp(out->data[i]);
	}
	ad_value* sum = ad_value_alloc(0.0);
	for(uint i = 0; i < x->size; i++){
		sum = ad_value_add(sum, out->data[i]);
	}
	for(uint i = 0; i < x->size; i++){
		out->data[i] = ad_value_div(out->data[i], sum);
	}

	return out;
}
```
You'll notice something interesting; at every iteration of the first loop, the `ad_value` at index `i` is being freed. This is done to make sure that the assignment `out->data[i] = ad_value_sub(x->data[i], xmax)` results in the elements of `out` holding the correct memory addresses of the new `ad_value`s generated by `ad_value_sub`. Not freeing the `ad_value`s created by `ad_matrix_alloc` before the assignment would immediately orphan the memory, leading to a memory leak.

In the second loop though, this is not done for the opposite reason - we _need_ to track the dependence of the inputs of the expression `out->data[i] = ad_value_exp(out->data[i])`. As this loop is run, ad_value_exp creates a new node whose `previous[0]` points to the old     `out->data[i]`, which is exactly how the computation graph gets built.

This idea is carried forward in other matrix-specific functions: `ad_matrix_matmul` and `ad_matrix_rmsnorm`.


##  Local Gradients and Backpropagation
The gradient of a function with respect to its inputs really boils down to an accumulation of the local gradients over the computational graph. Here's where the design decisions made until now come to a head. 

To calculate the gradient of an `ad_value` with respect to its inputs, `ad_value_backward` is called: 
```c
void ad_value_backward(ad_value* out){
	size_t graph_size = GRAPH_SIZE;
	ad_value** sorted = (ad_value**)malloc(GRAPH_SIZE * sizeof(ad_value*));
	ad_value** visited = (ad_value**)malloc(GRAPH_SIZE * sizeof(ad_value*));
	size_t sorted_size = 0;
	size_t visited_size = 0;
	graph_sort(out, &sorted, &sorted_size, &visited, &visited_size, &graph_size);

	out->grad = 1.0;
	for(int i = sorted_size - 1; i >=0; i--){
		grad(sorted[i]);
	}
	for(int i = 0; i < (int)sorted_size-1; i++){
		sorted[i]->visited = false;
		if(!sorted[i]->is_param){
			ad_value_free(sorted[i]);
		}
	}
	sorted[sorted_size - 1]->visited = false;
	free(sorted);
	free(visited);

}
```
Let's step through this function. 
1.  `GRAPH_SIZE` is a macro set initially as 8192. Overreaching on `GRAPH_SIZE` might lead to a stack overflow later in the execution. 
2. `ad_value** sorted` stores the topologically sorted computational graph. 
3. `ad_value** visited` stores the visited nodes, to prevent a node's local gradient from being accumulated multiple times. This is especially relevant with shared nodes. 
4. `graph_sort` is a recursive function which topologically sorts the initial computational graph. 


5. `out->grad = 1.0` to initiate the backprop
6. For every node in the sorted graph, the local gradient is calculated, accumulated and passed to the next node. 
7. Every non-parameter node - intermediate node not belonging to the state dict - is freed. 
8. Every parameter-node is flagged as not visited to prepare for the next iteration of backprop

The calculation and accumulation of the local gradients makes use of the `enum OPTYPE` of the `ad_value`: 

```c
void grad(ad_value* out){
	switch(out->op){
		case NONE: 
			break; 
		case ADD: 
			out->previous[0]->grad += out->grad; 
			out->previous[1]->grad += out->grad; 
			break; 
		case SUB: 
			out->previous[0]->grad += out->grad; 
			out->previous[1]->grad += -out->grad; 
			break; 
		case MUL: 
			out->previous[0]->grad += out->grad * out->previous[1]->data; 
			out->previous[1]->grad += out->grad * out->previous[0]->data; 
			break; 
		case DIV: 
			out->previous[0]->grad += out->grad * 1 / out->previous[1]->data; 
			out->previous[1]->grad += out->grad * (-out->previous[0]->data / (pow(out->previous[1]->data, 2))); 
			break; 
		case POW:
			out->previous[0]->grad += out->grad * out->previous[1]->data * pow(out->previous[0]->data, out->previous[1]->data - 1) ;
			out->previous[1]->grad += out->grad * (out->data * log(out->previous[0]->data)) ;
			break;
		case SIN: 
			out->previous[0]->grad += out->grad * cos(out->previous[0]->data);
			break; 
		case COS: 
			out->previous[0]->grad += out->grad * -1 * sin(out->previous[0]->data);
			break; 
		case TANH: 
			out->previous[0]->grad += out->grad * (1 - pow(out->data, 2));
			break; 
		case SIGMOID: 
			out->previous[0]->grad += out->grad * out->data * (1 - out->data);
			break; 
		case RELU: 
			out->previous[0]->grad += out->grad * ((out->data > 0 ) ? 1 : 0);
			break; 
		case LOG: 
			out->previous[0]->grad += out->grad * (1.0 / out->previous[0]->data);
			break; 
		case EXP: 
			out->previous[0]->grad += out->grad * (out->data);
			break; 
		case NEG: 
			out->previous[0]->grad += out->grad * (-1);
			break; 
	}
}
```
The initial `GRAPH_SIZE` is set to 8192. If the number of nodes in the computational graph exceeds this, a `realloc` is triggered in `graph_sort`: 
```c
void graph_sort(ad_value* out, ad_value*** sorted, size_t* sorted_size, ad_value*** visited, size_t* visited_size, size_t* capacity){
	if(out->visited) return;
	out->visited = true; 
	if(*visited_size >= *capacity){
		*capacity *= 2; 
		*visited = realloc(*visited, *capacity * sizeof(ad_value*));
		*sorted = realloc(*sorted, *capacity * sizeof(ad_value*));
		assert(*sorted != NULL);
		assert(*visited != NULL);
	}
	(*visited)[*visited_size] = out;
	(*visited_size)++;
	for(int i = 0; i < NUM_PREVS; i++){
		if(!out->previous[i])
			break;
		graph_sort(out->previous[i], sorted, sorted_size, visited, visited_size, capacity);
	}
	(*sorted)[*sorted_size] = out;
	(*sorted_size)++;
}
```

All the pieces for a homegrown scalar-valued backpropagation engine are now in place. The actual GPT-2 forward pass and training code are contained in [utils.c](https://github.com/agastyapatri/microgpt-c/blob/main/utils.c), and follows closely the method described in [the "Architecture" section of the original blog post](https://karpathy.github.io/2026/02/12/microgpt/#:~:text=hundreds%20of%20billions.-,Architecture,-The%20model%20architecture), along with the memory management described earlier. 

##  Performance
microgpt-c was trained for 1000 steps with the following hyperparameters: 
```c
#define N_EMBD 16    						//	embedding dimension 
#define N_HEAD 4							//	number of attention heads 
#define N_LAYER 1							//	number of layers 
#define GPT_BLOCK_SIZE 16					//	maximum sequence length
#define HEAD_DIM (int)(N_EMBD / N_HEAD) 	// dimensionality of each head
#define LEARNING_RATE (float)1.0e-2
#define BETA1 0.85              
#define BETA2 0.99
#define NUM_STEPS 1000
#define EPS_ADAM (float)1e-8
#define TEMPERATURE 0.5
```



The actual training + inference code was compiled with the settings `-std=c17 -Wall -Wextra -O3 -ffast-math -DNDEBUG -march=native`, and profiled with [perf](https://man7.org/linux/man-pages/man1/perf.1.html), a statistical profiler provided by the Linux kernel. 


Running `perf stat` gives:

```
Performance counter stats for './main':

             0      context-switches:u               #      0.0 cs/sec  cs_per_second     
             0      cpu-migrations:u                 #      0.0 migrations/sec  migrations_per_second
       673,913      page-faults:u                    # 216895.2 faults/sec  page_faults_per_second
      3,107.09 msec task-clock:u                     #      1.0 CPUs  CPUs_utilized       
    18,532,638      cpu_core/branch-misses/u         #      0.5 %  branch_miss_rate         (99.81%)
 4,033,483,079      cpu_core/branches/u              #   1298.2 M/sec  branch_frequency     (99.81%)
 6,912,081,802      cpu_core/cpu-cycles/u            #      2.2 GHz  cycles_frequency       (99.81%)
19,537,667,088      cpu_core/instructions/u          #      2.8 instructions  insn_per_cycle  (99.81%)
    10,118,524      cpu_atom/branch-misses/u         #      0.3 %  branch_miss_rate         (0.03%)
 4,632,166,754      cpu_atom/branches/u              #   1490.8 M/sec  branch_frequency     (0.03%)
 6,998,143,681      cpu_atom/cpu-cycles/u            #      2.3 GHz  cycles_frequency       (0.06%)
12,501,881,248      cpu_atom/instructions/u          #      2.2 instructions  insn_per_cycle  (0.10%)
         TopdownL1 (cpu_core)                        #      4.8 %  tma_bad_speculation    
                                                     #     17.2 %  tma_frontend_bound       (99.81%)
                                                     #     32.5 %  tma_backend_bound      
                                                     #     45.5 %  tma_retiring             (99.81%)
         TopdownL1 (cpu_atom)                        #     34.0 %  tma_backend_bound        (0.16%)
                                                     #     13.8 %  tma_frontend_bound       (0.13%)
                                                     #     16.2 %  tma_bad_speculation    
                                                     #     36.0 %  tma_retiring             (0.10%)

   3.231840422 seconds time elapsed

   2.188536000 seconds user
   1.024326000 seconds sys
```



Some things to note: 
-   The program spends nearly 31% of the total execution time executing system calls (`sys` time). This is a surprisingly large number. 

-   There are a total of 673,913 page faults. Let's look a bit deeper into how the memory allocation is actually happening.

    When `./main` is run, the operating system makes a set of contiguous memory locations - called the Virtual Address Space (VAS) - available to it. This memory is initially _unmapped_ to any memory in the physical RAM.
    
    Whenever an `ad_value` is created, `malloc` is called, which requests a pointer to a range of virtual addresses in the VAS. No physical memory is actually allocated yet. Whenever the data in that memory needs to be accessed, say by dereferencing an instance of `ad_value*`, the kernel realizes that there is no physical RAM mapped to it yet. This triggers an exception called a Page Fault. 

    The CPU then pauses execution of `./main` and jumps into [Kernel Mode](https://en.wikipedia.org/wiki/Kernel_(operating_system)#:~:text=is%20an%20exokernel.-,Monolithic%20kernels,-%5Bedit%5D). The kernel fetches a 4KB page of physical memory, updates the page tables to map the virtual address to this physical memory, and hands control back to the program. The instruction is tried again. 

    This process happened _~674,000_ times! 

    `perf` mentions `1.024326000 seconds sys`, meaning that ~1.024 seconds of the total execution time, roughly 31%, was spent by the CPU in performing this juggling of tasks. This number needs to be as low as possible.  



-   [Intel's 12th Generation Core Processors (Alder Lake)](https://en.wikipedia.org/wiki/Alder_Lake) introduced the combined Performance-Core / Efficient-Core architecture, similar to [ARM big.LITTLE](https://en.wikipedia.org/wiki/ARM_big.LITTLE). In the `perf stat` output, `cpu_core` refers to the P-Cores and `cpu_atom` the E-Cores.
    Both `cpu_core` and `cpu_atom` have ~2.2 instructions per cycle, indicating that the CPU is kept busy throughout the program execution, but modern CPUs can reach 4-6 instructions per cycle. 2.2 is middling, but not terrible.

-   Modern CPUs use Speculative Execution. When the CPU encounters an if statement or a for loop, it "guesses" which way the code will go and starts executing those instructions in advance. If the CPU mispredicts the path of the code, then typically around 15-20 CPU cycles are wasted.
    Because these loops repeat thousands of times with a very predictable exit condition, the CPU's Branch Predictor learns the pattern almost immediately. The P-Cores have a 0.5% branch misprediction rate, meaning that only a few misses occur and the amount of wasted CPU time is minimal. The E-Core numbers are less statistically reliable.


The `TopdownL1` statistics are specific to Intel's Topdown Microarchitecture Analysis - a way to understand where the CPU's time goes. I'll cover this in a later blog post, as my understanding of CPU's lower-level minutiae increases. 

##  Looking ahead
As it stands, the profiler data points to a flaw in the `microgpt-c`'s automatic differentiation engine - the thousands of `malloc` calls, thousands of resultant page faults and wasted CPU time. An impactful fix would be to move to [Arena Allocation](https://en.wikipedia.org/wiki/Region-based_memory_management), in which a large chunk of memory would be allocated and mapped upfront. This directly addresses the sys time and the memory bound costs the profiler indicated. 

There is also a critical shortcoming in the design of `ad_matrix`. In the interest of simplicity, I've implemented `ad_matrix` as a collection of several `ad_value` structs, each of which is a dyanmically allocated entity. Operations on `ad_matrix` sometimes replace the elements of the initially contiguous `ad_value**` buffer with other `ad_value*` memory addresses. This would kill the cache locality benefit of a contiguous memory buffer, meaning that the CPU is spending extra cycles fetching from RAM. This _does_ explain some of the `perf stat` outputs which point to the execution being extremely memory-bottlenecked, with the CPU cache sitting idle instead of doing what it was designed to do.

I'll have to address both of these issues in a later post. Stay tuned! 

















