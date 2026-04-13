+++
date        = '2026-04-08T06:53:50+05:30'
draft       = false
title       = 'microgpt-c'
tags        = ["llms", "c"] 
description  = "Training a tiny GPT to generate new names, from scratch, in C." 
+++
[Following Andrej Karpathy's 200 line python implementation of GPT-2](https://karpathy.github.io/2026/02/12/microgpt/), I wanted to try porting it to C as an exercise in learning the language. This post will lay out my approach in re-implementing microgpt with nothing but the C standard library. 

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
6.  Step through the sorted graph and propagate the local gradient starting from the root node, which in most cases is a scalar valued loss function. Et Voila! 

Not quite. Most C specific minutiae crop up from manually having to manage the lifetime and memory of the nodes in the computational graph. Let's talk about it. 

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

The `ref_count` parameter is my humble attempt at implementing a managing `ad_value` lifetimes with reference counting. The idea is that each `ad_value` keeps track of how many other `ad_value`s refer to it. When the reference counter hits 0, the `ad_value` memory is freed. The main reason to do this is that accidentally freeing a random `ad_value` in the computational graph and trying to backpropagate through it would be catastrophic; a problem given the colloquialism of "use-after-free". This would also be a nightmare to debug for a novice C programmer. This is slightly different from the textbook definition of the technique, in which a resource is shared among multiple objects which point to it, and is freed when the reference counter hits 0. 

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
One of the primary assumptions of automatic differentiation (autodiff) is that any function represented programmatically can be decomposed into a sequence of more primitive calculations; a principle known as atomicity. We need to make sure that the library of mathematical functions for `ad_value` enables atomicity and the use of the [chain rule](https://en.wikipedia.org/wiki/Chain_rule). 

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
I prefer to think of this simply as an array of `ad_value` instances instead of a matrix-valued autodifferentiation solution. Each element of an `ad_matrix` is a seperate parameter in the neural net, which needs to be updated independently. A true matrix implementation would be leagues faster. We're now ready to actually start building the network.

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
C doesn't have hash tables, we have to fall back to the trusty struct + indexing method. The layer-wise parameters are impelemented as arrays of `ad_matrix` pointers. `state_dict["layer0.attn_wv"]` becomes `state_dict.attn_wv[i]`
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
	ad_matrix** attn_wo;		// 	array of output?? matrices for each layer
	ad_matrix** mlp_fc1;		// 	array of fully connected portions of each layer
	ad_matrix** mlp_fc2;		// 	array of fully connected portions of each layer
} state_dict;
```
You'd have noticed the `is_param` flag in the `ad_value` struct; it notes which `ad_value`s are coming from the model's `state_dict` so that they're not freed after the backward pass. 



##  Local Gradients and Backpropagation





-   Write about the `ad_value_backward`
-   Write about the performance speedup seen from checking `visited` instead of traversing the visited array and 
-   Should the gpt_forward function be talked about?
-   Add a discussion on the `perf` output
-   Add a discussion on some performance improvements - a future blog post maybe?


