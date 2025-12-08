# Performance estimation and design of first-order methods over Holder smooth convex functions

This repository contains the implementation of a convex PEP framework for the analysis of first-order methods on Hölder-smooth convex functions, together with a numerical scheme for optimizing the step sizes of these algorithms.

---

## Authors
- Yassine Kamri  
- Julien M. Hendrickx  
- François Glineur  

---

## Getting Started
The code is written in Julia and requires the [JuMP](https://jump.dev) optimization toolbox together with the [Mosek](https://www.mosek.com) SDP solver.

---

## Description of the Files

- **opt_step_sizes_memoryless_algo_Holder.jl**  
  code for PEP and design procesure of memoryless first-order methods over Holder smooth convex functions 

-  **opt_step_sizes_full_algo_Holder.jl**   
   code for PEP and design procesure of full first-order methods over Holder smooth convex functions 

