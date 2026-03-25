# Frankenstein
This System uses "The Game of Infinite Paths" A system of complexity and unknowns that leads to a system where noise becomes reduced and perfection takes the place of the unknowns.

LIVING AI SYSTEM — COMPLETE BUILD PLAN
GOVERNING PRINCIPLE
Every component of this system operates by the Game of Infinite Paths. This is not a design choice — it is the actual mechanism. An infinite space of all possible states exists. Filters are applied. The correct answer is not chosen, it is revealed by the intersection of all filters. As the system learns, its model resolution increases. As model resolution increases, noise decreases. The endpoint is a complete model where noise approaches zero. Every module, every architecture, every learning paradigm in this system is an expression of this same principle at a different resolution level.

WHAT IS BEING BUILT
A fully standalone, self-contained AI system with its own web application interface. It calls no external APIs. It has no token limits. It has no rate limits. It pays nothing to anyone. It is used for chat, tasks, and generation.
The system unifies every AI architecture type, every machine learning paradigm, every graph neural network variant, and every learning method that exists — all wired together under one controller as one system. Each type exists as its own module. The controller opens the connection to whichever modules are needed. They all operate under the same governing principle.

PHASE 0 — Project structure and controller
The controller is the central nervous system. It is the file through which all modules are connected. It routes inputs to the correct modules, aggregates their outputs, and manages the flow of information through the system.


PHASE 1 — Neural network modules
Every neural network architecture exists as its own module under modules/neural_networks/. Each module contains its own fully implemented forward pass, training logic, and connection interface to the controller.
Feedforward networks: Single layer perceptron, multilayer perceptron, deep feedforward network.
Convolutional networks: Standard convolutional neural network, deep convolutional network, depthwise separable convolution, dilated convolution, transposed convolution, AlexNet architecture, VGG architecture, ResNet with residual connections, DenseNet with dense connections, Inception network with parallel filter sizes, EfficientNet with compound scaling, MobileNet for lightweight inference, U-Net for segmentation with skip connections.
Recurrent networks: Vanilla recurrent neural network, Long Short-Term Memory network, Gated Recurrent Unit, Bidirectional recurrent network, Bidirectional Long Short-Term Memory, Deep recurrent network, Echo State Network, Clockwork recurrent neural network.
Transformer and attention networks: Vanilla transformer with encoder and decoder, encoder-only transformer, decoder-only transformer, Vision Transformer treating image patches as tokens, Swin Transformer with shifted window attention, BERT-style masked language model, GPT-style autoregressive language model, T5-style text-to-text transformer, multi-head attention module, cross-attention module, linear attention, sparse attention, local attention, sliding window attention.
Capsule networks: Standard capsule network with dynamic routing, expectation-maximisation capsule network.
Generative networks: Generative Adversarial Network with generator and discriminator, Deep Convolutional Generative Adversarial Network, Wasserstein Generative Adversarial Network, Conditional Generative Adversarial Network, CycleGAN for unpaired image translation, StyleGAN for high resolution image generation, Variational Autoencoder, Beta Variational Autoencoder, Vector Quantized Variational Autoencoder, Denoising Diffusion Probabilistic Model, Score-based diffusion model, Latent diffusion model, Flow-based generative model, Normalizing flow, Autoregressive model, Masked autoencoder.
Autoencoder networks: Standard autoencoder, denoising autoencoder, sparse autoencoder, contractive autoencoder, deep autoencoder.
Energy-based and Hopfield networks: Restricted Boltzmann Machine, Deep Belief Network, Deep Boltzmann Machine, Hopfield Network, Modern Hopfield Network with exponential storage capacity.
Spiking neural networks: Leaky integrate-and-fire neuron model, spike timing dependent plasticity learning, rate-coded spiking network, temporal coding spiking network — these implement the threshold-gated event-driven computation specified in the Living AI volumes as the metabolic engineering module.
Radial basis function networks: Radial basis function network, probabilistic neural network.
Self-organising networks: Self-organising map, adaptive resonance theory network, growing neural gas.
Mixture models: Mixture of Experts with learned routing, Sparse Mixture of Experts with top-k gating, Switch Transformer variant.
Siamese and contrastive networks: Siamese network for similarity learning, triplet network, contrastive learning network.
Memory-augmented networks: Neural Turing Machine, Differentiable Neural Computer, Memory Network, End-to-End Memory Network.

PHASE 2 — Graph neural network modules
Every graph neural network variant exists as its own module under modules/graph_networks/.
Convolutional graph networks: Graph Convolutional Network with spectral convolution, Graph Convolutional Network with spatial convolution, Adaptive Graph Convolutional Network, Chebyshev spectral graph convolutional network, Diffusion Convolutional Neural Network, Dual Graph Convolutional Network.
Attention graph networks: Graph Attention Network, Graph Attention Network version 2, Gated Attention Network.
Recurrent graph networks: Gated Graph Neural Network using Gated Recurrent Unit propagation, Graph Long Short-Term Memory, Tree Long Short-Term Memory with child-sum variant and N-ary variant, Sentence Long Short-Term Memory.
Message passing networks: Message Passing Neural Network — the general framework for all message passing variants, Graph Network as defined by Battaglia.
Sampling and aggregation networks: GraphSAGE with mean aggregator, GraphSAGE with Long Short-Term Memory aggregator, GraphSAGE with pooling aggregator, PinSAGE for large scale graphs.
Autoencoder graph networks: Graph Autoencoder, Variational Graph Autoencoder, Adversarially Regularized Graph Autoencoder.
Generative graph networks: Graph Recurrent Neural Network for graph generation, Junction Tree Variational Autoencoder for molecular graphs, GraphRNN for sequential graph generation.
Spatial-temporal graph networks: Spatial-Temporal Graph Convolutional Network, Temporal Graph Network with memory modules, Dynamic Graph Convolutional Neural Network, Diffusion Convolutional Recurrent Neural Network.
Skip connection graph networks: Jumping Knowledge Network, Highway Graph Neural Network.
Hierarchical graph networks: Edge-Conditioned Convolution Network, Differentiable Pooling, Top-k pooling, Self-attention pooling.
Graph echo state networks: Graph Echo State Network using reservoir computing.
Stochastic graph networks: Stochastic Steady-state Embedding network.
Higher-order graph networks: k-dimensional Graph Neural Network, Graph Isomorphism Network.

PHASE 3 — Learning paradigm modules
Every learning paradigm exists as its own module under modules/learning_paradigms/. The controller activates the appropriate learning paradigm for each training signal received.
Supervised learning: Classification using decision trees, random forests, support vector machines, k-nearest neighbours, naive Bayes, logistic regression, linear regression, ridge regression, lasso regression, gradient boosting, XGBoost, LightGBM, AdaBoost, neural network classifiers. Regression using all regression variants above plus neural regression heads.
Unsupervised learning: Clustering using k-means, hierarchical clustering, density-based spatial clustering of applications with noise, Gaussian Mixture Models, spectral clustering, affinity propagation, mean shift. Dimensionality reduction using Principal Component Analysis, t-distributed Stochastic Neighbour Embedding, Uniform Manifold Approximation and Projection, Independent Component Analysis, Factor Analysis, Autoencoders. Association rule learning using Apriori algorithm, FP-Growth algorithm. Anomaly detection using Isolation Forest, One-Class Support Vector Machine, Local Outlier Factor, Autoencoder-based anomaly detection.
Semi-supervised learning: Self-training, co-training, label propagation, label spreading, generative semi-supervised models, semi-supervised Generative Adversarial Networks, mean teacher model, temporal ensembling, mixmatch, fixmatch.
Self-supervised learning: Contrastive self-supervised learning using SimCLR, MoCo, BYOL, SimSiam. Masked modelling using masked autoencoder, masked language modelling. Predictive self-supervised learning using next token prediction, next frame prediction, rotation prediction, jigsaw puzzle prediction.
Reinforcement learning: Q-learning, Deep Q-Network, Double Deep Q-Network, Dueling Deep Q-Network, Prioritised Experience Replay, Policy Gradient using REINFORCE algorithm, Actor-Critic, Advantage Actor-Critic, Asynchronous Advantage Actor-Critic, Proximal Policy Optimisation, Trust Region Policy Optimisation, Soft Actor-Critic, Deep Deterministic Policy Gradient, Twin Delayed Deep Deterministic Policy Gradient, Multi-Agent Deep Deterministic Policy Gradient, model-based reinforcement learning using world models, Monte Carlo Tree Search, AlphaZero-style self-play.
Meta-learning: Model-Agnostic Meta-Learning, Prototypical Networks, Matching Networks, Relation Networks, Siamese Networks for few-shot learning, Neural Process, Conditional Neural Process, Meta-SGD, Reptile.
Continual learning: Elastic Weight Consolidation — adds a quadratic penalty to the loss function that constrains modification of parameters identified as important for prior tasks, using the Fisher Information Matrix to measure importance. This is the Immortal Helix module. Orthogonal Weight Modification, Gradient Episodic Memory, Averaged Gradient Episodic Memory, Progressive Neural Networks, Packnet, Learning Without Forgetting, Knowledge Distillation for continual learning, Synaptic Intelligence.
Transfer learning: Fine-tuning, feature extraction, domain adaptation, Low-Rank Adaptation for efficient fine-tuning.
Federated learning: Federated Averaging, Federated Proximal, Differential Privacy with federated learning.
Active learning: Uncertainty sampling, query by committee, expected model change, core-set selection.
Multi-task learning: Hard parameter sharing, soft parameter sharing, task-conditioned hypernetworks.
Curriculum learning: Self-paced learning, competence-based curriculum, transfer teacher curriculum.
Bayesian learning: Bayesian Neural Network, Gaussian Process, Bayesian Optimisation, Monte Carlo Dropout for uncertainty estimation, Deep Ensemble for uncertainty estimation.
Evolutionary and genetic learning: Genetic Algorithm, Neuroevolution of Augmenting Topologies, Covariance Matrix Adaptation Evolution Strategy, Particle Swarm Optimisation.
Learning Classifier Systems: XCS, XCSR — rule-based systems combining genetic algorithms with learning.
Inductive Logic Programming: Logic-based learning from examples and background knowledge.

PHASE 4 — Five tier memory system
The memory system mirrors biological memory architecture exactly as specified in the Living AI volumes.
Working memory — the active context window holding current reasoning state, recent conversation, and in-progress task context. High bandwidth, volatile, managed in process using asyncio-safe session state with token-budget enforcement and rolling truncation when the budget is exceeded.
Episodic memory — a vector database storing encoded representations of past interactions with temporal metadata. Retrieval uses semantic similarity search. Implemented using ChromaDB with sentence-transformers embeddings running locally. Consolidation is triggered by the homeostasis loop at session boundaries.
Semantic memory — the weight matrices of the network modules — the compressed, generalised knowledge accumulated through learning. Protected against catastrophic forgetting by the Elastic Weight Consolidation module. Updates only occur during consolidation windows with Fisher Information Matrix constraints enforced.
Procedural memory — the skill registry and Low-Rank Adaptation adapters for learned task-specific procedures. Each skill is a module with defined trigger patterns, required tools, required permissions, a core execution function, quality metrics, and optionally a Low-Rank Adaptation weight delta. Adapters are loaded on demand.
External knowledge base — SQLite with full-text search enabled, a knowledge graph implemented using NetworkX persisted as versioned JSON snapshots, and a Retrieval-Augmented Generation pipeline. The retrieval pipeline uses hybrid search — dense vector similarity, sparse BM25 keyword search, and multi-hop knowledge graph traversal — followed by cross-encoder reranking to select the top results for context injection.

PHASE 5 — Homeostasis
The homeostasis system is the Continuous Execution Environment — an infinite asyncio loop running as a background daemon. It does not stop. It does not wait for requests. It is the system's heartbeat.
Every tick the loop does five things in order. It perceives — reads hardware state including CPU utilisation, memory pressure, and thermal state using psutil. It processes — handles any pending background consolidation tasks from the queue. It consolidates — when the consolidation window timer triggers, it runs episodic to semantic memory transfer, Fisher Information Matrix recomputation, knowledge distillation, and spaced repetition scheduling for critical knowledge. It maintains — runs the Ouroboros pruning pass, which scores every parameter in every active module using the utility function defined in the Living AI volumes and zeros out parameters scoring below the dynamic threshold. It paces — sleeps for the configured tick interval before the next iteration.
The Ouroboros pruner scores each parameter as a weighted combination of gradient magnitude, activation frequency, and weight magnitude. The threshold scales dynamically with current memory pressure so that pruning accelerates when resources are constrained.
The metacognitive monitor tracks performance drift using KL divergence between current and baseline output distributions. When drift exceeds two standard deviations it triggers a forced consolidation cycle and logs an alert.

PHASE 6 — Embodiment modules
Eyes — visual perception pipeline with four stages from low level feature extraction through semantic parsing to cross-modal grounding. Static image understanding, video and temporal vision, three-dimensional spatial vision, and specialised visual modalities all implemented as sub-modules.
Ears — audio perception pipeline covering speech recognition and understanding, audio scene understanding, music intelligence, and biometric audio analysis.
Voice — speech synthesis pipeline with acoustic model, vocoder, prosodic control, voice persona maintenance, and multi-modal non-verbal output.
Hands (digital) — computer use and graphical user interface automation, code generation and execution with iterative debugging loop, API orchestration covering REST and GraphQL, database operations, data pipeline control, and communication API integration.
Hands (physical) — robotic manipulation policy architecture from high level goal to torque output, dexterous grasping, bimanual coordination, deformable object handling.
Feet (digital) — web crawling and multi-hop traversal, code navigation via abstract syntax tree, knowledge graph traversal, file system monitoring and navigation.
Feet (physical) — localisation and mapping, motion planning and control, locomotion across wheeled, legged, aerial, aquatic, and micro-scale modalities.

PHASE 7 — Reasoning, planning, skills, and knowledge
Reasoning — deductive, inductive, abductive, analogical, causal, and spatial reasoning modules. Chain of thought, tree of thought, graph of thought, process reward models, and self-adversarial debate all implemented as reasoning strategies selectable by the router.
Planning — hierarchical task network planning for goal decomposition, action sequencing, contingency planning, and multi-step horizon planning using Monte Carlo Tree Search. The autonomous planner checks preconditions before each step, replans on failure, and escalates to human oversight on anomaly detection.
Multi-agent coordination — orchestrator and worker agent architecture with typed message passing schema, shared vector database workspace, and conflict resolution protocol.
Skills — skill registry with trigger pattern matching, quality metric tracking, reward-weighted regression for skill improvement, transfer learning across related skills, and meta-learning for rapid adaptation to new skill domains.
Knowledge — knowledge graph with typed entity nodes, typed directed relation edges, temporal validity windows, and Bayesian confidence values on all facts. World model covering physical world, social world, information world, self model, goal state, and causal model. Advanced retrieval strategies including Hypothetical Document Embeddings, multi-query expansion, recursive summarisation, self-retrieval with quality evaluation, and multi-hop graph retrieval.

PHASE 8 — Permissions and audit
The capability gate system implements every capability as a binary flag defaulting to disabled. Every capability invocation checks the gate before execution. No action proceeds without a passing gate check.
The audit log is an append-only SQLite table with hash-chaining. Every row stores the hash of the previous row concatenated with the current action record, creating a tamper-evident chain. Fields cover unique identifier, timestamp with nanosecond precision, trace identifier, action type, capability name, payload hash, outcome, and previous hash.

PHASE 9 — Backend API
FastAPI application with full async implementation throughout. WebSocket manager for streaming token-by-token response delivery. Endpoints for chat, task submission, task status polling, memory read and write, health check with dependency status, and capability gate configuration. All endpoints have input validation, error handling, structured JSON logging, and request correlation identifier propagation. PostgreSQL via asyncpg for persistence. Redis for task queue brokering and response caching. Celery workers for task execution.

PHASE 10 — Frontend
React with TypeScript. Chat panel with WebSocket connection, reconnect logic, streaming token display, message type rendering for user, assistant, system, and task status messages, and syntax-highlighted code blocks. Task panel for arbitrary task submission with status polling and structured result display. Memory browser with episodic memory viewer, knowledge base full-text and semantic search. System status panel showing homeostasis loop health, active modules, memory tier sizes, and capability gate states. Settings panel for capability gate configuration.

PHASE 11 — Containerisation and launch
Docker Compose with services for the backend, frontend, PostgreSQL, Redis, and Celery workers. Multi-stage Dockerfiles with non-root runtime users and health checks on all services. PowerShell launch script with development and production modes. Complete .env.example with every variable documented. Zero hardcoded credentials anywhere in the codebase.

PRODUCTION STANDARD
Every file in this system is built to the standard defined in the Professional AI Execution document. One hundred percent implementation coverage. No placeholders. No TODO comments. No simplified stubs. Every function has complete internal logic. Every error path is handled. Structured JSON logging throughout. Health check endpoint at /health with full dependency status. All secrets via environment variables. Parameterised queries only. Async input/output throughout. Connection pooling on all database and HTTP clients. Explicit timeouts on all external calls. Graceful shutdown handling.
