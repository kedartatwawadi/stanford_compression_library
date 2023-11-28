# Implementation of Multiset Compression in SCL

- [ ] Implement MSBST in SCL
- [ ] Create encode/decode functions
  - [ ] Frequency map details
  - [ ] Encode function
  - [ ] Decode function
- [ ] Create JSON/multiset (simple) test cases
- [ ] Incorporate Huffman/other comparisons for benchmarking
- [ ] Create plots/figures to show progress

### Introduction

<!-- We learnt about rANS in class and will learn about bits-back coding in HW2. Recent paper and implementation show that it is possible to have a very general method which allows saving bits by not saving the order in the data (utilizing permutation invariance). This uses bits-back coding underneath. This project will involve understanding and re-implementing this method in SCL. -->

In this work, we aim to understand how to utilize bits-back coding to achieve an improved compressionr ratio in a multiset compression setting. A multiset is a generalization of a set that allows for repetition of elements, where the critical information is contained in the elements and not their ordering (equivalently, the frequency map of the elements is what must be preserved.) 

Any standard method to compress a sequence of `n` elements (such as Huffman coding, rANS, or others) will preserve the order of the input elements. By discarding this information, we save bits corresponding to the element ordering. This improved compression technique can be applied in scenarios where the critical information is not contained in the ordering of the elements, such as JSON maps or unordered dictionaries. In this project, we implement a generalized version of this compressor in SCL. Next, we perform numerous experiments collecting compressor performance and compression ratios on synthetic and real datasets.

### Literature/Code review

Our primary source will be the \href{https://arxiv.org/abs/2107.09202}{paper itself}, which is extensive and provides sufficient depth to fully understand bits-back coding and its application to this problem. A secondary source is the \href{https://arxiv.org/abs/1901.04866}{original paper} on latent variable compression.

### Methods
<!-- What method or algorithm are you proposing? If there are existing implementations and/or theoretical justifications, will you use them and how? How do you plan to improve or modify such implementations? You don't have to have an exact answer at this point, but you should have a general sense of how you will approach the problem you are working on. -->

We plan to implement the multiset compression method described in the paper in a clean, well-documented fashion in the SCL library. To this end, we will implement the encoding and decoding scheme and the accompanying modified BST data structure to support efficient interval insertions and lookups for use therein. While there is a \href{https://github.com/facebookresearch/multiset-compression}{reference implementation} for the paper, we will plan to develop our code independently of theirs.

The algorithms and approach are clearly detailed in the paper, so the primary hurdle lies more in understanding than implementing. To demonstrate our understanding of the compression scheme, we will generalize it to arbitrary latent variable space compression --- instead of solely order-insensitivity for multisets, we will use the method for, \textit{e.g.,} case-insensitivity for compressing text, or left vs. right insensitivity for tree compression. 

Our plan is to develop the code and SCL implementation by the Milestone deadline, so that remaining time can be spent on evaluating, optimizing, and report-writing. Our process will start with a deeper analysis of the mathematics and techniques involved as described in the paper, to understand the core ideas.

### Progress report
<!-- % what have you already finished (please include code link where relevant)? What is the plan for the remaining weeks? -->

#### Progress so far

#### Plan until Dec. 6th presentation

#### Plan until Dec. 15th report

<!-- We will quantify the compression ratio of the input data and the time it takes our algorithm to compress the data (performance). Our initial ideas for the experiments are drawn from the paper; for example, we will write a script to sample JSON dictionaries with very large numbers of entries, of configurable structure. We expect that performing multiset compression on such JSON dictionaries will yield greater compression ratios than other compression techniques (rANS, tANS, Huffman, etc.), with the specific savings dependent on the multiset size. Recall that the multiset size is the information saved, since this is what we remove the need to encode in the "bits-back" process.

We also will develop other case studies to demonstrate the potential for our algorithm to improve compression ratios when other kinds of invariance can be included in "bits-back"; for example, when compressing text, it may be the case that the user is agnostic to the casing of the characters. Such performance tests will allow us to form a Pareto frontier, exhibiting the tradeoff between multiset size and value size. We will also find real datasets that contain varying amounts of multiset repetition to ensure that the results obtained on synthetic data carry over.

Our results will be presented using various plots and figures, such as those included in the results and evaluation section of the original paper (ex: Fig 2, page 20). These will depict the empirical saved bits as a function of multiset size for various datasets. The compression ratios we achieve, along with the associated algorithm runtimes, will be tabulated and presented alongside the results of the aforementioned compression techniques.

Finally, in the process of developing examples and synthetic data, we will experiment with different priors (in addition to the Dirichlet prior used in the paper), such as Gaussian, uniform, dyadic, etc. to observe the impact of multiset structure on performance. Much of our deliverables will be the results of these performance tests, including any adjustments we make to improve performance along the way. -->


