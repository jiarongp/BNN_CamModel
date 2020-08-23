# Camera Model Identification based on Bayesian Neural Network 

## Week 2

- The take away message from `What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision` is that they have a novel loss function incoperated both aleatoric and epistemic uncertainty, and the loss could be attenuated according to the uncertainty. Aleatoric uncertainty can't be reduced by enlarge the dataset, on the contrary, epistemic can.

- Which variational method in BNN should we use? Bayes by Backprop or using the dropout approximation? I did a little experiment by myself, turns out the dropout has smoother curve. In the paper 'Dropout as a Bayesian Approximation' states that the Bayes by Backprop is slower.

    > We use the stochastic variational inference, bayes by backprop is our focus. To summarize the different methods we talked about:
    >
    > - Softmax statistics, see Hendrycks and Gimpel
    > - Monte-Carlo dropout (MC dropout), see papers by Yarin Gal
    > - Ensembles, see Lakshminarayanan et al.
    > - Stochastic Variational Inference (SVI), e.g., Blundell et al. <--- focus our work
    > - Last layer SVI and Last layer dropout: Approximate Bayesian inference for last layer only.

- Aleatoric and epistemic uncertainty in camera model identification? Should I model it?

- How do you know the modeled uncertainty is correct? Evaluate it's log likelihood, RMSE and std. error?

- In the case of resampling artifacts,  the rescaling factor indicates the distance(dissimilarity) between the closeset samples and the unseen samples. But in camera identification, should we use such a dissimilarity?

## Week 3

- Why the MC Dropout predictiions looked "smoother" than the from the Bayesian neural network example from Martin Krasser. Turns out it is related to the `batch_size` you use during prediciton. When you use `model.predict(X_test)` Keras has a default value for `batch_size=32`. 
Would be great if you could repeat your experiments with `batch_size=1`.

    > `batch_size` means that the maximum number of GPU/CPU can handle at the same time, so if the `batch_size` is set to 32, then the model will wait for 32 inputs then predict 32 predictions all together. If I change the prediction number to 32, using `batch_size` or not present the same result. In the figure, the predictions show very distinct boundaries at the begining of every batch prediction, while inside the batch, the prediction of this points seems very smooth. 
    >
    > This is because the model takes the whole batch as a input, so they share the same random kernel and bias, which will result in linear output, so inside the batch, the output is smooth. When the next batch comes, the model has different random kernel and bias, so the output looks like it has a step.

- To do:
    - [x] Conv2d Bayesian by backprop using [flipout](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/Convolution2DFlipout#args_1) or other [variational 2d](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/conv_variational). 
    - [x] Read the paper of `Evaluating Predictive Uncertainty Under Dataset Shift`, in the Appendix A6 they have some Variational Stochastic Inference setup details. [Github repository](https://github.com/google-research/google-research/tree/master/uq_benchmark_2019).
      > For MNIST we used Flipout (Wen et al., 2018), where we replaced each dense layer and convolutional layer with **mean-field variational dense** and **convolutional Flipout layers** respectively. Variational inference for deep ResNets (He et al., 2016) is non-trivial, so for CIFAR we replaced a single linear layer per residual branch with a Flipout layer, removed batch normalization, added **Selu non-linearities (Klambauer et al., 2017)**, **empirical Bayes for the prior standard deviations** as in Wu et al. (2019) and careful tuning of the initialization via Bayesian optimization.
    - [x] try to use `selu` as the activation function.
      > `selu` needs to be initilized suitably. 
    - [x] This [blog post](https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html) introduce a trainable prior.
        ```python
        def prior_trainable(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            return tf.keras.Sequential([
                tfp.layers.VariableLayer(n, dtype=dtype),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t, scale=1),
                    reinterpreted_batch_ndims=1)),
            ])
        ```
    - [x] [Deterministic Variational Inference for Robust Bayesian Neural Networks](https://arxiv.org/pdf/1810.03958.pdf). This paper that describes how to do Empirical Bayes to learn the variance of the prior from the data.
    - [x] Code for [trainable prior](https://github.com/google-research/google-research/blob/9049acf9246c1b75170f0c6757e62a8f619a9db6/uq_benchmark_2019/imagenet/resnet50_model.py#L316).
        > Watch out, I'm not quite sure if the code matches the description in the paper
    - [ ] [Bayesian Layers: A Module for Neural Network Uncertainty](https://arxiv.org/pdf/1806.05978.pdf) and their [blog](https://medium.com/neuralspace)
    - [ ] [Bayesian Layer](https://arxiv.org/pdf/1812.03973.pdf)



## Week 4

- According to the paper [Weight Uncertainty in Neural Network](https://arxiv.org/abs/1505.05424) by Blundell et al., the empirical Bayes method is not suitable for Bayesian by Backprop, so they select the simple fixed scaled mixture Gaussian as the prior for this algorithm. They argue that using empirical Bayes the prior will quickly learn the randomly initialized weights which are not correct, and once it learns it, the prior will hardly change in the following training so it becomes an obstacle of learning the right posterior. </p> In the blog post [regression with probabilistic layers in tensorflow](https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html), they also mention that:
    > This training corresponds to using **Empirical Bayes** or Type-II Maximum Likelihood. We used this method so that we wouldn’t need to specify the location of the prior for the slope and intercept parameters, which can be tough to get right if we do not have prior knowledge about the problem. Moreover, if you set the priors very far from their true values, then the posterior may be unduly affected by this choice. **A caveat of using Type-II Maximum Likelihood is that you lose some of the regularization benefits over the weights. If you wanted to do a proper Bayesian treatment of uncertainty (if you had some prior knowledge, or a more sophisticated prior), you could use a non-trainable prior (see Appendix B)**.

- How about [local reparameterization](https://github.com/JavierAntoran/Bayesian-Neural-Networks/issues/14#issuecomment-622109141)? 

- What is gradient variance? Is it generated by the monte carlo sampling? Also, in the paper Local Reparameterization Trick and the flipout also mention that by using this trick can reduce this variance.
    > If the variance of the weight is large, then does it mean the  variance of the gradient of the weight is also large, it will leads to slower convergence? So we need to reduce the gradient variance.
    >
    > Stochastic gradient optimization is a class of widely used algorithms for training machine learning models. To optimize an objective, it uses the noisy gradient computed from the random data samples instead of the true gradient computed from the entire dataset. However, when the variance of the noisy gradient is large, the algorithm might spend much time bouncing around, leading to slower convergence and worse performance.

- I use the entropy of the monte carlo predictions to evalute the uncertainty of model's output. The original code base on [tfp example](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/bayesian_neural_network.py) is following.
    ```python
    # probs shape (50, 10000, 10), [# of inferences, # of test object, # of output]
    probs = tf.stack([model.predict(heldout_seq, verbose=0)
                        for _ in range(FLAGS.num_monte_carlo)], axis=0)
    # mean_probs shape (10000, 10), average over all the inferences
    mean_probs = tf.reduce_mean(probs, axis=0)
    # heldout_log_prob is a scalar
    heldout_log_prob = tf.reduce_mean(tf.math.log(mean_probs))
    print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))
    ```
    the output `nats` from the above is a minus number, and further decrease along the training.

    ```python
    probs = tf.stack([model.predict(heldout_seq, verbose=0) for _ in range(num_monte_carlo)], axis=0)
    mean_probs = tf.reduce_mean(probs, axis=0)
    # sum_log_prob shape (10000,), use the definition, sum of p * ln(p) over i (i is 10)
    sum_log_prob = -tf.reduce_sum(tf.math.multiply(mean_probs, tf.math.log(mean_probs)), axis=1)
    # average over the 10000 test entropy
    heldout_log_prob = tf.reduce_mean(sum_log_prob)
    print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))
    ```
    I change it into this, following the definition of natural entropy 
    $H = - \sum_i p_i \ln{p_i}$

    >The metric measured in the paper ***Deterministic Variational Inference for Robust Bayesian Neural Networks*** is in fact the test log-likelihood measured in nats. Lower values are better. Many papers seem to prefer reporting the test log likelihood (instead of flipping the sign by compute the negative log-likelihood), which can be negative. They are working on a regression task, and computing the log-likelihood requires the ground truth label, so this still does not answer what the tfp example is computing.

    I look a the code of the calculation of [categorical_loss](https://github.com/microsoft/deterministic-variational-inference/blob/03a9829038db19dc0c59f5b9b18a4a676c2ce414/bayes_layers.py#L152).

    ```python
    def categorical_loss(logits, target, model, hypers, global_step, MC_samples=-1):
        """
        compute <p(D|w)>_q - lambda KL(q || p)
        """
        lsm = tf.cond(tf.greater(MC_samples, 0),
            lambda: sampled_logsoftmax(logits, MC_samples), # we evaluate the logsoftmax using MC sampling
            lambda: logsoftmax(logits)                      # we evaluate the logsoftmax using the delta approx
        )

        all_surprise = tf.reduce_sum(tf.stack([w.surprise() for w in model.parameters]))
        logprob = tf.reduce_sum(target * lsm, axis=1)
        batch_logprob = tf.reduce_mean(logprob)
        
        lmda = hypers['lambda']
        
        L = lmda * all_surprise / hypers['dataset_size'] - batch_logprob
    return L, batch_logprob, all_surprise
    ```
    Here the `logprob` is the target values multiply by the `logsoftmax` and then sum it up, which is corresponding to the cross-entropy. so the output `batch_logprob` measured in `nats` is the cross-entropy average over the whole batch. (no negative sign)

    > Whatever they compute in the tfp example cannot be the log likelihood, because they are working on a classification task and the ground truth label is required to compute the log likelihood, but they don't use it in their computation.
    >
    > If you want to report the log likelihood (as a measure how well your model explains the validation data) in a classification task: The likelihood function for a categorical variable is given in Bishop's PRML book, Equation 4.89, and the negative log-likelihood is given by Eqn. 4.90. We know this function already, it is also known as the cross-entropy.
    > Apparently the tfp example got changed in January, see [here](https://github.com/Pyrsos/probability/commit/811b06b458a9200391a601a0f605987add0a02c9#diff-f73d46eababcdeb0ae3091cd36dc13a6L323).
    >
    > Before that, it computed something more similar to the log-likelihood (= cross-entropy), but still not exactly the formula. I still don't know what the current version is meant to represent. There is also an [unanswered issue](https://github.com/tensorflow/probability/issues/635) about that. At the moment, I understand that the example computes some sort of certainty of the model in its predictions, where lower values mean higher certainty.

- If I use `SELU`, does that mean I don't need to use `batch_normalization()`?
    > The answer is No.

### Week 5

- Regarding for the `nats` problem, what I need is to evaluate the uncertainty of the results, not only on the test dataset, but also on the Out-of-Distributions(OOD) samples. The following is from the paper [Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift](https://arxiv.org/abs/1906.02530):
  > Negative Log-Likelihood (NLL) ↓ Commonly used to evaluate the quality of model uncertainty on some held out set. Drawbacks: Although a proper scoring rule (Gneiting & Raftery, 2007), it can over-emphasize tail probabilities (Quinonero-Candela et al., 2006).
  >
  > There is no ground truth label for fully OOD inputs. Thus we report **histograms of confidence** and **predictive entropy** on *known and OOD inputs* and **accuracy versus confidence plots** (Lakshminarayanan et al., 2017): Given the prediction $p(y = k|x_n , θ)$, we define the predicted label as $ŷ_n = \argmax_y p(y|x_n , θ)$, and the confidence as $p(y = ŷ|x, θ) = max_k \ p(y = k|x_n , θ)$. We filter out test examples corresponding to a particular confidence threshold $τ ∈ [0, 1]$ and compute the accuracy on this set.

    So I would like to use the accuarcy and cross-entropy to measure the performance of the trained Bayesian Neural Network (if the cross-entropy is large, then it also indicates uncertainty is high), and then use the entropy as one of the evalution method to see the uncertainty. (Maybe I can try [brier score](https://github.com/google-research/google-research/blob/d58326621f40c3ce02dae453fcd737dd13093cac/uq_benchmark_2019/metrics_lib.py#L268) later.)

    [compute_accuracies_at_confidences](https://github.com/google-research/google-research/blob/499fa7141d012c22e6d40898bebeedecee5ce6a7/uq_benchmark_2019/metrics_lib.py#L190)

### Week 6

- understand the empirical bayes method. Maybe it's better to first try using non-learnable prior (standard variance).
- the model is not able to converge and show poor accuaracy.
    1. first try to simplied the model and force it to overfit.
    2. trace the weight and gradient during training using tensorboard.
- Thnik about what's the difference and advantages using different layers (`tfp.layers.Reparameterization`, `tfp.layers.Flipout` etc.)

### Week 7

**To test the uncertainty output**:
- First, output the monte carlo draw of the test set images every 50 steps
- Second, test on the images taken on my phone.
- Adding post processing to images. (Gaussian blur and adjust JPEG quality)

One observation is that, if using a smaller kl weight (divided by a larger number), the model will converge faster, i.e. have higher accuracy with less epochs, but I assume it doesn't learn the prior distribution well.

- using number of training images, the model converge quickly (96% acc after 5 epochs), but the ouput uncertainty on out-of-distribution data is bad.

- using number of batches as kl_weights, the model converge slowly and diverge after 6 epochs, but the ouput uncertainty is not improved.(I expect it would have a good distribution of weight)
  - Result: test loss: 48.905, test accuracy: 79.955% (because its not converging, model using the 6th training epochs, val_loss: 777.0889, val_accuracy: 0.8173), outputs entropy has 0.292 nats. 

- using model.add is not the same as defining the sequential model straightfowardly. (bug on tfp)


Todo:
- [ ] Why my code has different loss, about a half of the loss of keras code?
- [ ] Tweak the kl_weight.
- [ ] set `kl_weight` to 0 to see if the model will converge, if it does, nothing wrong with the initialization.
- [ ] use scalar to describe the variance of prediction.

## Week 8

The default initialization of the `kernel_posterior` is:

```python
tfp.layers.default_mean_field_normal_fn(
    is_singular=False, loc_initializer=tf1.initializers.random_normal(stddev=0.1),
    untransformed_scale_initializer=tf1.initializers.random_normal(mean=-3.0,
    stddev=0.1), loc_regularizer=None, untransformed_scale_regularizer=None,
    loc_constraint=None, untransformed_scale_constraint=None
)
```
The **mean** of the `kernel_posterior` is intialized by a zero mean with 0.1 stddev Gaussian, the **standard deviation** is initialized by a -3.0 mean with 0.1 standard deviation (before softplus, so this is the $\rho$). After softplus it is 0.04858735157374196 (`np.log(1 + np.exp(-3))`).

The `kernel_prior` is a standard normal distribution (0 mean with 1 standarddeviation). After training, the `kernel_posterior` should shift towards the `kernel_prior`.

The dataset has 41,325 training images, 1292 batches.

### Problem
- When set the kl_weight to num_batches, the kl loss becomes negative.
- [x] using the epsitemic and aleatoric decomposition from the [code](https://gitlab.cs.fau.de/snippets/101), it will be a K by K matrix, where K is the number of classes. Does the diagonal of this matrix indicates the uncertainty? (It's a auto covariance matrix to me.) 
  - [answer from kwon](https://github.com/ykwon0407/UQ_BNN/issues/3#issuecomment-452525593)
  - Also, there is a way to calculate the uncertainty [here](https://github.com/ykwon0407/UQ_BNN/issues/3#issuecomment-455529058)


## Week 9

- output score of epistemic uncertainty for the images
- plot ROC curve? (I don't get the idea how to automatically classifiy them)
- calibration, brier score

About aleatoric and epistemic uncertainty
> In a nutshell, viewing a model’s aleatoric uncertainty output should caution us to factor in appropriate deviations when making our predictions, while inspecting epistemic uncertainty should help us re-think the appropriateness of the chosen model.

- [Uncertainty & Out-of-Distribution Robustness in Deep Learning](http://www.gatsby.ucl.ac.uk/~balaji/mluq-talk-balaji.pdf) Good introductory talk for bayesian neural network.

- standard CNN, Dan Hendrys method
- BNN, Dan Hendrys  (for loop outside the test_step)
  - histogram of variance
  - ROC curve

# Week 10
Nikon_D7000_r0bf7f938t.TIF is corrupted.

- [ ] a approach to detect the ood sample is to use generative model to capture the distribution of data. use the captured distribution to distinguish ood samples.
  - [Do Deep Generative Model Know What They Don't Know?](https://arxiv.org/abs/1810.09136)
- [ ] plot histogram of softmax output and entroupy in one plot for different dataset.
- [ ] increas the option to test the model as ood database (is mnist ok?)
- [ ] A thought on the application of uncertainty. Is it possible to incoporate uncertainty in to training? First train the model on BNN, and if in the training the model output high uncertainty during the validation, we can only focus on these examples and if the model has low uncertainty, we could reduce the samples of the corresponding class. It would reduce the amount of training samples that we need. Is it possible? (BNN version of active learning?)
  - the softmax uncertainty is already enough, right?

The ROC curves looks convex, because the true positive has smaller value than the true negative, as the threshold goes up, the true poitive rate will first small, than goes up to 1.

# Week 11

Experiment:
- By setting the kernel and bias posterior like the following:
  ```python
  kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                        loc_initializer=keras.initializers.GlorotUniform(), 
                        # or He normal as initialization
                        untransformed_scale_initializer=tf.random_normal_initializer(mean=-3.0,
                        stddev=0.0))
  bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                      is_singular=True,
                      loc_initializer=tf.random_normal_initializer(stddev=0.0))
  ```
  lr = 0.0001 using RMSprop. kl_weight is train_size. It reaches 89% accuracy.

  The kernel posterior is initialized as a Gaussian, of which the mean is uniformly distributed in range of [-limit, limit], where limit = sqrt(6 / (fan_in + fan_out)) (fan_in is the number of input units in the weight tensor and fan_out is the number of output units). The variance of this Gaussian is near 0, which is 0.04858735157374196 (`np.log(1 + np.exp(-3))`).

  The bias posterior is a deterministic distribution, with the mean of them is 0, variance is near 0.

## Note

### ELBO

$$ ELBO(\theta) = - \int dw Q(w;\theta) log P(Y|X, w) + \int dw Q(w;\theta) log \frac{Q(w;\theta)}{P(w)} $$

ELBO uses three distributions:
- $P(w)$ is the prior over the weights. It is the distribution we assume the weights to follow before we trained the model.
- $Q(w; θ)$ is the variational posterior parameterized by parameters θ. This is an approximation to the distribution of the weights after we have trained the model.
- $P(Y | X, w)$ is the likelihood function relating all inputs $X$, all labels $Y$ and the weights. When used as a probability distribution over $Y$, it specifies the variation in $Y$ given $X$ and the weights.

ELBO is a lower bound on $log P(Y | X)$, i.e. the log-likelihood of the labels given the inputs after marginalizing away the uncertainty over the weights. **ELBO works by trading off the KL divergence of $Q$ with respect to the prior over the weights (the second term), with the ability to predict labels from the inputs (the first term).** When there is little data, the second term dominates and our weights remain close to the prior distribution, which as a side effect helps prevent overfitting.

### Regression

Fit model to the data by maximizing the probability of the labels, or equivalently, minimizing the negative log-likelihood loss: $-log P(y | x)$.

Mean squared error loss for continuous labels, for example, means that $P(y | x, w)$ is a normal distribution with a fixed scale (standard deviation). Cross-entropy loss for classification means that $P(y | x, w)$ is the categorical distribution.

### [Empirical Bayes](https://colab.research.google.com/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter6_Priorities/Ch6_Priors_TFP.ipynb#scrollTo=DD_MDbroh3zQ&line=25&uniqifier=1)

While not a true Bayesian method, *empirical Bayes* is a trick that combines frequentist and Bayesian inference. As mentioned previously, for (almost) every inference problem there is a Bayesian method and a frequentist method. The significant difference between the two is that Bayesian methods have a prior distribution, with hyperparameters $\alpha$, while empirical methods do not have any notion of a prior. Empirical Bayes combines the two methods by using frequentist methods to select $\alpha$, and then proceeds with Bayesian methods on the original problem. 

A very simple example follows: suppose we wish to estimate the parameter $\mu$ of a Normal distribution, with $\sigma = 5$. Since $\mu$ could range over the whole real line, we can use a Normal distribution as a prior for $\mu$. How to select the prior's hyperparameters, denoted ($\mu_p, \sigma_p^2$)? The $\sigma_p^2$ parameter can be chosen to reflect the uncertainty we have. For $\mu_p$, we have two options:

**Option 1**: Empirical Bayes suggests using the empirical sample mean, which will center the prior around the observed empirical mean:

$$ \mu_p = \frac{1}{N} \sum_{i=0}^N X_i $$

**Option 2**: Traditional Bayesian inference suggests using prior knowledge, or a more objective prior (zero mean and fat standard deviation).

Empirical Bayes can be argued as being semi-objective, since while the choice of prior model is ours (hence subjective), the parameters are solely determined by the data.

Personally, I feel that Empirical Bayes is *double-counting* the data. That is, we are using the data twice: once in the prior, which will influence our results towards the observed data, and again in the inferential engine of MCMC. This double-counting will understate our true uncertainty. To minimize this double-counting, I would only suggest using Empirical Bayes when you have *lots* of observations, else the prior will have too strong of an influence. I would also recommend, if possible, to maintain high uncertainty (either by setting a large $\sigma_p^2$ or equivalent.)

Empirical Bayes also violates a theoretical axiom in Bayesian inference. The textbook Bayesian algorithm of:

>*prior* $\Rightarrow$ *observed data* $\Rightarrow$ *posterior* 

is violated by Empirical Bayes, which instead uses 

>*observed data* $\Rightarrow$ *prior* $\Rightarrow$ *observed data* $\Rightarrow$ *posterior*

Ideally, all priors should be specified *before* we observe the data, so that the data does not influence our prior opinions (see the volumes of research by Daniel Kahneman *et. al* about [anchoring](http://en.wikipedia.org/wiki/Anchoring_and_adjustment)).

### Factors that influence the performance of a SVI BNN

- KL-weight, the larger it is, the higher the uncertainty, but the accuracy will decrease.
- The depth of the network (number of variational layers), the more layers it has, the higher the KL-loss.
- The scale and mean of the scale. The weight after softplus $variance = log(1+exp(\rho))$. $\rho = \mu + \epsilon \odot \sigma$, $\mu$ is the mean of untransformed variance, $\epsilon \in \mathcal{N}(\mathbf{0}, \mathbf{I})$. Notice that this is different from the softplus of the weight, in tensorflow probability, the untransformed variance is also controlled by a mean and variance.
- The number of training examples, the more training examples, the more the certain the model is.
- In the notebook `Bayesian linear regression: Impact of prior variance`, it shows that the kl divergence gets smaller when $\alpha$ (controls the magnitude of the prior variance) get larger, but the marginal log likelihood (the overall performance of the model, including accuracy and uncertainty) decrease.

### [ROC Curves and Presicion-Recall Curves](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)
#### ROC Curve

It is a plot of the false positive rate (x-axis) versus the true positive rate (y-axis) for a number of different candidate threshold values between 0.0 and 1.0. Put another way, it plots the false alarm rate versus the hit rate.

The true positive rate is also referred to as **Sensitivity**:
`True Positive Rate = True Positives / (True Positives + False Negatives)`

The false positive rate is called the false alarm rate as it summarizes how often a positive class is predicted when the actual outcome is negative:
`False Positive Rate = False Positives / (False Positives + True Negatives)`

The false positive rate is also referred to as the inverted specificity where specificity is the total number of true negatives divided by the sum of the number of true negatives and false positives:
`Specificity = True Negatives / (True Negatives + False Positives)` where: `False Positive Rate = 1 - Specificity`

![image](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/ROC_curves.svg/600px-ROC_curves.svg.png)

#### Precision-Recall Curves

Precision describes how good the model is at predicting the positive class:
`Precision = True Positives / (True Positives + False Positives)`

Recall is the same as **Sensitivity**:
`Recall = True Positives / (True Positives + False Negatives)`

Reviewing both precision and recall is useful in cases where there is an imbalance in the observations between the two classes. There are many examples of no event (class 0) and only a few examples of an event (class 1). If we only interest in the ability of the model predicting the minority class (class 1), this can help us to avoid the high true negatives affect the evaluation (no true negatives in the formula, right?).

A precision-recall curve is a plot of the precision (y-axis) and the recall (x-axis) for different thresholds, much like the ROC curves.
> While the baseline is fixed with ROC, the baseline of [precision-recall curve] is determined by the ratio of positives (P) and negatives (N) as y = P / (P + N). For instance, we have y = 0.5 for a balanced class distribution …


When to use:
- ROC curves should be used when there are roughly equal numbers of observations for each class. (Too optimistic in class imbalance problem)
- Precision-Recall curves should be used when there is a moderate to large class imbalance.

Summary:
- ROC Curves summarize the trade-off between the true positive rate and false positive rate for a predictive model using different probability thresholds.
- Precision-Recall curves summarize the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds.
- ROC curves are appropriate when the observations are balanced between each class, whereas precision-recall curves are appropriate for imbalanced datasets.

### *A Baseline for Detecting Missclassified and Out-of-Distribution Examples in Neural Network*

Motivation: Detecting if an example is misclassified or out-of-distribution.

Observation: 
- prediction probability from a softmax distribution has a poor direct correspondence to confidence.
- Correctly classified examples tend to have greater maximum softmax probabilityies than erroneously classified and out-of-distribution examples,  allowing for their detection.

Problems:
1. **error and success prediction**: can we predict wether a trained classifier will make an error on a particular held-out test example?
2. **in- and out-of-distribution detection**: can we predict whether a test example is from a different distribution from the training data; can we predict if it is from within the same distribution?

**comparing detectors is not as straightfoward as using accuracy.**

In a class imbalance setting, the classifier may tend to output one of the class, which is misleading. We have to specify a score threshold so that it can classify correctly, but this depends upon the trade-off between false negatives (fn) and false positives(fp).

Two evaluation methods:
- AUROC
  - a threshold-independent performance evaluation.
  - the AUROC can be interpreted as the probability that a positive example has a greater detector score/value than a negative example.
- AUPR
  - for the suitation that positive class negative class have greatly differing base rates (depends on the **true positive rate**)
  - so we have to specify the positive class

**Softmax Prediction Probability As A Baseline**:
1. Separate correctly and incorrectly classified *test set* examples and, for each example, compute the softmax probability of the predicted class, i.e., the maximum softmax probability.
2. obtain AUROC and AUPR. These two areas summarize the performance of a binary classifier discriminating with values/scores across different thresholds.

This means, if we want to know whether an example is classified correctly or not, we can plot the AUROC or AUPR of the example, the AUPR should be very low.

> A debatable, imprecise interpretation of AUROC values may be as follows: 90%—100%: Excellent, 80%—90%: Good, 70%—80%: Fair, 60%—70%: Poor, 50%—60%: Fail.

## How to measure the quality of the preditive uncertainty?

1. Calibration
   - Measure how well the model predictive confidence algin with the observed accuracy