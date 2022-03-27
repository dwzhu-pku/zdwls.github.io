---
title: Fairseq Tutorial 01 Basics
author: David
date: 2022-03-26 16:35:00 +0800
categories: [NLP, Fairseq]
tags: [fairseq]
math: false
mermaid: true
---

This post is an overview of the fairseq toolkit. Some important components and how it works will be briefly introduced.


## Overview
---
Here are some important components in fairseq:
1. Tasks: Tasks are responsible for preparing dataflow, initializing the model, and calculating the loss using the target criterion.
2. Models: A Model defines the neural network’s `forward` method and encapsulates all of the learnable parameters in the network. Each model also provides a set of named architectures that define the precise network configuration (e.g., embedding dimension, number of layers, etc.).
3. Modules: In Modules we find basic components (e.g. transformer_layer, multihead_attention, etc.) of a model.
4. Criterions: Criterions provide several loss functions give the model and batch.
5. Optimizers: Optimizers update the Model parameters based on the gradients.
6. Learning Rate Schedulers: Learning Rate Schedulers update the learning rate over the course of training.

## Workflow
In this part we briefly explain how fairseq works. Getting an insight of its code structure can be greatly helpful in customized adaptations.
The entrance points (i.e. where the `main` function is defined) for training, evaluating, generation and apis like these can be found in folder *fairseq_cli*. For this post we only cover the fairseq-train api, which is defined in `train.py`. Taking this as an example, we'll see how the components mentioned above collaborate together to fulfill a training target.

In `train.py`, we first set up the task and build the model and criterion for training by running following code:
```python
# trainer.py/main
    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    # Build model and criterion
    model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
```
Then, the task, model and criterion above is used to instantiate a `Trainer` object, the main purpose of which is to facilitate parallel training.

That done, we load the latest checkpoint available and restore corresponding parameters using the `load_checkpoint` function defined in module `checkpoint_utils`. After that, we call the `train` function defined in the same file and start training.
## References
1. https://fairseq.readthedocs.io/en/latest/index.html