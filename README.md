# Awesome Distributed Deep Learning

<p align="center">
	<img src="https://img.shields.io/badge/stars-0-brightgreen.svg?style=flat"/>
	<img src="https://img.shields.io/badge/forks-0-brightgreen.svg?style=flat"/>
	<img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat">
</p>

A curated list of awesome Distributed Deep Learning resources.

## Table of Contents

### **[Frameworks](#frameworks)** 

### **[Blogs](#blogs)** 

### **[Papers](#papers)**  
<!--
### **[Tutorials](#tutorials)**  

<!--
### **[Miscellaneous](#miscellaneous)**  
<!--
### **[Contributing](#contributing)** -->

## Frameworks

1. [MXNet](https://github.com/dmlc/mxnet) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, Javascript and more.
2. [go-mxnet-predictor](https://github.com/songtianyi/go-mxnet-predictor) - Go binding for MXNet c_predict_api to do inference with pre-trained model.
3. [deeplearning4j](https://github.com/deeplearning4j/deeplearning4j) - Distributed Deep Learning Platform for Java, Clojure, Scala.
4. [Distributed Machine learning Tool Kit (DMTK)](http://www.dmtk.io/) - A distributed machine learning (parameter server) framework by Microsoft. Enables training models on large data sets across multiple machines. Current tools bundled with it include: LightLDA and Distributed (Multisense) Word Embedding.
5. [Elephas](https://github.com/maxpumperla/elephas) - Elephas is an extension of Keras, which allows you to run distributed deep learning models at scale with Spark.
6. [Horovod](https://github.com/uber/horovod) - Distributed training framework for TensorFlow.

## Blogs

1. [Keras + Horovod = Distributed Deep Learning on Steroids](https://medium.com/searchink-eng/keras-horovod-distributed-deep-learning-on-steroids-94666e16673d)
2. [Meet Horovod: Uber’s Open Source Distributed Deep Learning Framework for TensorFlow
](https://eng.uber.com/horovod/)
3. [distributed-deep-learning-part-1-an-introduction-to-distributed-training-of-neural-networks/](https://blog.skymind.ai/distributed-deep-learning-part-1-an-introduction-to-distributed-training-of-neural-networks/)

## Papers

1. [Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis](https://arxiv.org/abs/1802.09941):discusses the different types of concurrency in DNNs; synchronous and asynchronous stochastic gradient descent; distributed system architectures; communication schemes; and performance modeling. Based on these approaches, it also extrapolates the  potential directions for parallelism in deep learning. 
2. [Deep learning with COTS HPC systems](http://ai.stanford.edu/~acoates/papers/CoatesHuvalWangWuNgCatanzaro_icml2013.pdf): Commodity Off-The-Shelf High Performance Computing (COTS HPC) technology, a cluster of GPU servers with Infiniband interconnects and MPI.

**Feedback: If you have any ideas or you want any other content to be added to this list, feel free to contribute.**
