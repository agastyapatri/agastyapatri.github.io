+++
date        = '2026-03-27T06:53:50+05:30'
draft       = true
title       = 'microgpt-c'
tags        = ["llms", "c"] 
description  = "Training a tiny GPT to generate new names. A C port of Andrej Karpathy's Microgpt."
+++
[Following Andrej Karpathy's 200 line python implementation of GPT-2](https://karpathy.github.io/2026/02/12/microgpt/), I wanted to try porting it to C as an exercise in learning the language. This post will lay out my approach in re-implementing microgpt in C with nothing but the standard library. 

``

##  Scalar Valued Automatic Differentiation
The (admittedly handwavy) recipe for implementing your own backprop engine is simple: 
1.  Wrap your data in a `struct ad_value`:
    
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
    
2.  Track the operation that was used to create a `ad_value`, along with its parents `previous[0] and previous[1]`
3.  Calculate its local gradient, store it in `float grad`.
4.  Sort your computational graph, formed after math done sequentially with `ad_value_<op>`.
5.  Step through the sorted graph and propagate the local gradient starting from the root node, which in most cases is a scalar valued loss function.
Et Voila! 

Not quite. Most C specific minutiae crop up from manually having to manage the lifetime and memory of the nodes in the computational graph. 

-   Write about the method to track operations and do math; `enum OPTYPE ` and `ad_value_<op>`
-   Write about reference counting and how that was used to manage lifetimes; the shared node problems and the double free risks. 
-   Write about the performance speedup seen from checking `visited` instead of traversing the visited array and 



