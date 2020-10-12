# IBM Federated Learning

## What is it?
IBM federated learning is a Python framework for federated learning (FL) in an enterprise environment. FL is a distributed machine learning process, in which each participant node (or party) retains data locally and interacts with the other participants via a learning protocol. The main drivers behind FL are privacy and confidentiality concerns, regulatory compliance requirements, as well as the praciticality of moving data to one central learning location.

IBM federated learning provides a basic fabric for FL, to which advanced features can be added. It is not dependent on any specific machine learning framework and supports different learning topologies, e.g., a shared aggregator, and protocols. It supports Deep Neural Networks (DNNs) as well as classic machine learning techniques such as linear regression and k-means. This comprises supervised and unsupervised approaches as well as reinforcemnet learning. The figure below shows a typical configuration of an aggregator based federated learning setup supported by IBM federated learning.

<p align="center">
<img src="docs/floverview.png" width="566">
</p>
  
A set of parties own data and each trains a local model. The parties exchange updates with an aggregrator using a FL protocol. The aggregator fuses (aggregates) the results from the different parties and ships the consolidated results back to the parties. This can go through multiple rounds until a termination criterion is reached. IBM federated learning supports the configuration of these training scenarios.

The key design points of IBM federated learning are the ease of use for the machine learning professional, configurability to different computational environments - from data centers to edge devices - and extensibility. It can be extended to work with different machine learning (ML) libraries, learning protocols, and fusion algorithms. This provides a basic fabric on which FL projects can be run and research in FL learning can take place.

IBM federated learning comes with a large library of fusion algorithms for both DNNs and classic ML approaches, consisting of implementations of both common, published fusion algorithms as well as novel ones we have developed.

## Supported functionality
IBM federated learning supports the following machine learning model types: 

- Neural networks (any neural network topology supported by Keras)
- Decision Tree ID3 
- Linear classifiers/regressions (with regularizer): logistic regression, linear SVM, ridge regression, Kmeans and Naïve Bayes 
- Deep Reinforcement Learning algorithms including DQN, DDPG, PPO and more

IBM federated learning supports multiple state-of-the-art fusion algorithms to combine model updates coming from multiple parties. Changes in this algorithm may speed up the convergence, reduce training time or improve model robustness. 
For a particular ML model, you can select multiple types of fusion algorithms: 

|	*Supported ML Models*	                                    |	*Supported fusion algorithms*	|
|-----------------------------------------------------------|-------------------------------|
|  Neural Networks                    |	Iterative Average             |		          
|                                     | FedAvg  [McMahan et al.](https://arxiv.org/pdf/1602.05629.pdf)  |
|                                     | Gradient Average              |
|                                     | PFNM  [Yurochkin et al.](https://arxiv.org/abs/1905.12022) |
|                                     | Krum [Blanchard et al.](https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf)      |
|                                     | Coordinated median [Yin et al.](https://arxiv.org/pdf/1803.01498.pdf) |
|                                     | Zeno [Xie et al.](https://arxiv.org/abs/1805.10032)  |
|                                     | SPAHM [Yurochkin et al.](https://arxiv.org/abs/1911.00218) |
| ID3 Decision Tree	                  |	ID3 fusion  [Quinlan](https://link.springer.com/article/10.1007/BF00116251)             |
|	Reinforcement Learning RLLib models	|	Iterative Average        |
|                                     |	FedAvg [McMahan et al.](https://arxiv.org/pdf/1602.05629.pdf)  |
|Linear classifiers with SGD | Iterative Average |
|K-means | SPAHM [Yurochkin et al.](https://arxiv.org/abs/1911.00218) |
|Naïve Bayes | Naive Bayes fusion with differential privacy|

## How to get started?

Clone the repository. The main framework runtime is packaged in a [whl file](federated-learning-lib/federated_learning_lib-1.0-py3-none-any.whl). 

Try the [set-up guide](setup.md) for a single-node federated learning setup. 

There are a number of [examples](examples/README.md) with explanation for different federated learning tasks with different model types to get started with.

## How does it work?

There is a [docs folder](./docs) with tutorials and API documentation to learn how to use and extend IBM federated learning.

- [Web Site](https://ibmfl.mybluemix.net/)
- Aggregator and Party [configuration tutorial](docs/tutorials/configure_fl.md)
- [API documentation](http://ibmfl-api-docs.mybluemix.net/index.html)
- [Related publications](docs/papers.md)
- [White paper](https://arxiv.org/abs/2007.10987)

## How to get in touch?

We appreciate feedback and questions. Please post issues when you encounter them. 

We have set up a Slack channel for ongoing discussion. Join the IBM federated learning workspace: https://ibm-fl.slack.com/


## Citing IBM Federated Learning

If you use IBM Federated Learning, please cite the following reference paper:

@article{ibmfl2020ibm,
  title={IBM Federated Learning: an Enterprise Framework White Paper V0. 1},
  author={Ludwig, Heiko and Baracaldo, Nathalie and Thomas, Gegi and Zhou, Yi and Anwar, Ali and Rajamoni, Shashank and Ong, Yuya and Radhakrishnan, Jayaram and Verma, Ashish and Sinn, Mathieu and others},
  journal={arXiv preprint arXiv:2007.10987},
  year={2020}
}

## Ongoing effort 

This is an ongoing effort. We plan to update this repo as new functionality is added frequently.

## License

IBM federated learning is distributed under this [license](LICENSE) for non-commercial and experimental use.

