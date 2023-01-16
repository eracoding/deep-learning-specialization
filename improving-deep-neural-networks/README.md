# Course [Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network) 
* Understand industry best-practices for building deep learning applications. 
* Be able to effectively use the common neural network "tricks", including initialization, L2 and dropout regularization, Batch normalization, gradient checking, 
* Be able to implement and apply a variety of optimization algorithms, such as mini-batch gradient descent, Momentum, RMSprop and Adam, and check for their convergence. 
* Understand new best-practices for the deep learning era of how to set up train/dev/test sets and analyze bias/variance
* Be able to implement a neural network in TensorFlow. 

# Week1 notes:
### Train/dev/test sets

Traditionally, you might take all the data you have and carve off some portion of it to be your training set, and some portion of it to be your hold-out cross-validation set; this is sometimes also called the development set. And for brevity, I'm just going to call this the dev set, but all of these terms mean roughly the same thing. And then you might carve out some final portion of it to be your test set. And so the workflow is that you keep on training algorithms on your training set. And use your dev set or your hold-out cross-validation set to see which of many different models performs best on your dev set. And then, after having done this long enough, when you have a final model that you want to evaluate, you can take the best model you have found and evaluate it on your test set in order to get an unbiased estimate of how well your algorithm is doing.

In the modern big data era, where, for example, you might have a million examples in total, then the trend is that your dev and test sets have been becoming a much smaller percentage of the total. Because remember, the goal of the dev set or the development set is that you're going to test different algorithms on it and see which algorithm works better. So the dev set just needs to be big enough for you to evaluate, say, two different algorithms choices or ten different algorithm choices and quickly decide which one is doing better. And you might not need a whole 20% of your data for that. So, for example, if you have a million training examples, you might decide that just having 10,000 examples in your dev set is more than enough to evaluate, you know, which one or two algorithms do better. And in a similar vein, the main goal of your test set is, given your final classifier, to give you a pretty confident estimate of how well it's doing. And again, if you have a million examples, maybe you might decide that 10,000 examples is more than enough in order to evaluate a single classifier and give you a good estimate of how well it's doing. So, in this example, where you have a million examples, if you need just 10,000 for your dev and 10,000 for your test, your ratio will be more like...this 10,000 is 1% of 1 million, so you'll have 98% train, 1% dev, 1% test.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/21b8ed47-dfda-4d14-b175-48ad6a8ef1fd/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/985ef6bf-a4fe-491f-b786-6cb2720bf381/Untitled.png)

Remember, the goal of the test set is to give you a ... unbiased estimate of the performance of your final network, of the network that you selected. But if you don't need that unbiased estimate, then it might be okay to not have a test set. So what you do, if you have only a dev set but not a test set, is you train on the training set and then you try different model architectures. Evaluate them on the dev set, and then use that to iterate and try to get to a good model. Because you've fit your data to the dev set, this no longer gives you an unbiased estimate of performance. But if you don't need one, that might be perfectly fine. In the machine learning world, when you have just a train and a dev set but no separate test set, most people will call this a training set and they will call the dev set the test set. But what they actually end up doing is using the test set as a hold-out cross-validation set. Which maybe isn't completely a great use of terminology, because they're then overfitting to the test set. So when the team tells you that they have only a train and a test set, I would just be cautious and think, do they really have a train dev set? Because they're overfitting to the test set.

So having set up a train dev and test set will allow you to integrate more quickly. It will also allow you to more efficiently measure the bias and variance of your algorithm so you can more efficiently select ways to improve your algorithm.

### Bias and Variance

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/86fb6394-7260-4838-8ca1-eb9815e6bffb/Untitled.png)

And so, let's say, your training set error is 1% and your dev set error is, for the sake of argument, let's say is 11%. So in this example, you're doing very well on the training set, but you're doing relatively poorly on the development set. So this looks like you might have overfit the training set, that somehow you're not generalizing well to this hold-out cross-validation set with the development set. And so if you have an example like this, we would say this has high variance. So by looking at the training set error and the development set error, you would be able to render a diagnosis of your algorithm having high variance. Now, let's say, that you measure your training set and your dev set error, and you get a different result. Let's say, that your training set error is 15%. I'm writing your training set error in the top row, and your dev set error is 16%. In this case, assuming that humans achieve, you know, roughly 0% error, that humans can look at these pictures and just tell if it's cat or not, then it looks like the algorithm is not even doing very well on the training set. So if it's not even fitting the training data as seen that well, then this is underfitting the data. And so this algorithm has high bias. But in contrast, this actually generalizing a reasonable level to the dev set, whereas performance in the dev set is only 1% worse than performance in the training set. So this algorithm has a problem of high bias, because it was not even fitting the training set. Well, this is similar to the leftmost plots we had on the previous slide. Now, here's another example. Let's say that you have 15% training set error, so that's pretty high bias, but when you evaluate to the dev set it does even worse, maybe it does, you know, 30%. In this case, I would diagnose this algorithm as having high bias, because it's not doing that well on the training set, and high variance. So this has really the worst of both worlds. And one last example, if you have, you know, 0.5 training set error, and 1% dev set error, then well maybe our users are quite happy, that you have a cat classifier with only 1% error, then this will have low bias and low variance.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f2635dad-3ed9-46ad-b3fe-7bdcea1177a5/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a65ddf1f-27e2-42c2-9721-3f0224ce23f7/Untitled.png)

### **Basic Recipe for Machine Learning**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/40c7d94d-284a-47a5-aa09-1098b22c61f2/Untitled.png)

### **“Bias variance tradeoff” -**

In the earlier era of machine learning, there used to be a lot of discussion on what is called the bias variance tradeoff. And the reason for that was that, for a lot of the things you could try, you could increase bias and reduce variance, or reduce bias and increase variance. But, back in the pre-deep learning era, we didn't have many tools, we didn't have as many tools that just reduce bias, or that just reduce variance without hurting the other one. But in the modern deep learning, big data era, so long as you can keep training a bigger network, and so long as you can keep getting more data, which isn't always the case for either of these, but if that's the case, then getting a bigger network almost always just reduces your bias, without necessarily hurting your variance, so long as you regularize appropriately.

## Regularization

### Frobenius norm formula:

$||w^{[l]}||^{2} = \sum_{i=1}^{n^{[l]}} \sum_{j=1}^{n^{[l-1]}} (w_{i,j}^{[l]})^{2}$

L2 is almost always used than L1.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/070084b0-7542-40ee-9c3c-9a82c77e5a96/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0350b800-ca6f-4972-b3d8-9341f258e4cd/Untitled.png)

L2 is also called weight decay.

### Why regularization reduces overfitting?

One piece of intuition is that if you, you know, crank your regularization lambda to be really, really big, that'll be really incentivized to set the weight matrices, W, to be reasonably close to zero. So one piece of intuition is maybe it'll set the weight to be so close to zero for a lot of hidden units that's basically zeroing out a lot of the impact of these hidden units. And if that's the case, then, you know, this much simplified neural network becomes a much smaller neural network. In fact, it is almost like a logistic regression unit, you know, but stacked multiple layers deep. And so that will take you from this overfitting case, much closer to the left, to the other high bias case. But, hopefully, there'll be an intermediate value of lambda that results in the result closer to this "just right" case in the middle. But the intuition is that by cranking up lambda to be really big, it'll set W close to zero, which, in practice, this isn't actually what happens. We can think of it as zeroing out, or at least reducing, the impact of a lot of the hidden units, so you end up with what might feel like a simpler network, that gets closer and closer as if you're just using logistic regression. The intuition of completely zeroing out a bunch of hidden units isn't quite right. It turns out that what actually happens is it'll still use all the hidden units, but each of them would just have a much smaller effect. But you do end up with a simpler network, and as if you have a smaller network that is, therefore, less prone to overfitting. So I'm not sure if this intuition helps, but when you implement regularization in the program exercise, you actually see some of these variance reduction results yourself.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/db7ee959-8f86-4349-abd5-b019b5202c9d/Untitled.png)

if lambda, the regularization parameter is large, then you have that your parameters will be relatively small, because they are penalized being large in the cost function. And so if the weights, W, are small, then because z is equal to W, right, and then technically, it's plus b. But if W tends to be very small, then z will also be relatively small. And in particular, if z ends up taking relatively small values, just in this little range, then g of z will be roughly linear. So it's as if every layer will be roughly linear, as if it is just linear regression. And we saw in course one that if every layer is linear, then your whole network is just a linear network. And so even a very deep network, with a deep network with a linear activation function is, at the end of the day, only able to compute a linear function. So it's not able to, you know, fit those very, very complicated decision, very non-linear decision boundaries that allow it to, you know, really overfit.

So to summarize, if the regularization parameters are very large, the parameters W very small, so z will be relatively small, kind of ignoring the effects of b for now, but so z is relatively, so z will be relatively small, or really, I should say it takes on a small range of values. And so the activation function if it's tan h, say, will be relatively linear. And so your whole neural network will be computing something not too far from a big linear function, which is therefore, pretty simple function, rather than a very complex highly non-linear function.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8e874ea0-e99c-4642-81fd-aa67e3a87742/Untitled.png)

### Dropout regularization

Let's say that for each of these layers, we're going to- for each node, toss a coin and have a 0.5 chance of keeping each node and 0.5 chance of removing each node. So, after the coin tosses, maybe we'll decide to eliminate those nodes, then what you do is actually remove all the outgoing things from that no as well. So you end up with a much smaller, really much diminished network. And then you do back propagation training. There's one example on this much diminished network. And then on different examples, you would toss a set of coins again and keep a different set of nodes and then dropout or eliminate different than nodes. And so for each training example, you would train it using one of these neural based networks.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3ef18209-53f1-47e1-bb2d-d354e83bbbe2/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3f3a3e35-4e29-43ee-ba45-1059e703a3ac/Untitled.png)

### Why drop-out work?

Drop out randomly knocks out units in your network. So it's as if on every iteration you're working with a smaller neural network. And so using a smaller neural network seems like it should have a regularizing effect.

Here's the second intuition which is, you know, let's look at it from the perspective of a single unit. Right, let's say this one. Now for this unit to do his job has four inputs and it needs to generate some meaningful output. Now with drop out, the inputs can get randomly eliminated. You know, sometimes those two units will get eliminated. Sometimes a different unit will get eliminated. So what this means is that this unit which I'm circling purple. It can't rely on anyone feature because anyone feature could go away at random or anyone of its own inputs could go away at random. So in particular, I will be reluctant to put all of its bets on say just this input, right. The ways were reluctant to put too much weight on anyone input because it could go away. So this unit will be more motivated to spread out this ways and give you a little bit of weight to each of the four inputs to this unit. And by spreading out the weights this will tend to have an effect of shrinking the squared norm of the weights, and so similar to what we saw with L2 regularization. The effect of implementing dropout is that its strength the ways and similar to L2 regularization, it helps to prevent overfitting, but it turns out that dropout can formally be shown to be an adaptive form of L2 regularization, but the L2 penalty on different ways are different depending on the size of the activation is being multiplied into that way. But to summarize it is possible to show that dropout has a similar effect to. L2 regularization.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/45e2d796-4bb5-48d9-ae7a-235f4050733e/Untitled.png)

So just to summarize if you're more worried about some layers of fitting than others, you can set a lower key prop for some layers than others. The downside is this gives you even more hyper parameters to search for using cross validation. One other alternative might be to have some layers where you apply dropout and some layers where you don't apply drop out and then just have one hyper parameter which is a key prop for the layers for which you do apply drop out and before we wrap up just a couple implantation all tips. Many of the first successful implementations of dropouts were to computer vision, so in computer vision, the input sizes so big in putting all these pixels that you almost never have enough data. And so drop out is very frequently used by the computer vision and there are some common vision research is that pretty much always use it almost as a default.

One big downside of drop out is that the cost function J is no longer well defined on every iteration. You're randomly, calling off a bunch of notes. And so if you are double checking the performance of great inter sent is actually harder to double check that, right? You have a well defined cost function J. That is going downhill on every elevation because the cost function J. That you're optimizing is actually less. Less well defined or it's certainly hard to calculate. So you lose this debugging tool to have a plot a draft like this. So what I usually do is turn off drop out or if you will set keep-propped = 1 and run my code and make sure that it is monitored quickly decreasing J. And then turn on drop out and hope that

### Other Regularization Methods

Let's say you fitting a CAD crossfire. If you are over fitting getting more training data can help, but getting more training data can be expensive and sometimes you just can't get more data. But what you can do is augment your training set by taking image like this. And for example, flipping it horizontally and adding that also with your training set. So now instead of just this one example in your training set, you can add this to your training example. So by flipping the images horizontally, you could double the size of your training set. Because you're training set is now a bit redundant this isn't as good as if you had collected an additional set of brand new independent examples.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c9e4535b-5ebd-48c8-970d-9bc465ad3d3a/Untitled.png)

There's one other technique that is often used called early stopping. So what you're going to do is as you run gradient descent you're going to plot your, either the training error, you'll use 01 classification error on the training set. Or just plot the cost function J optimizing, and that should decrease monotonically, like so, all right? Because as you trade, hopefully, you're trading around your cost function J should decrease. So with early stopping, what you do is you plot this, and you also plot your dev set error. And again, this could be a classification error in a development sense, or something like the cost function, like the logistic loss or the log loss of the dev set. Now what you find is that your dev set error will usually go down for a while, and then it will increase from there. So what early stopping does is, you will say well, it looks like your neural network was doing best around that iteration, so we just want to stop trading on your neural network halfway and take whatever value achieved this dev set error.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b10941d8-2ee9-444d-88e2-ba8f7590fd51/Untitled.png)

So why does this work?

Well when you've haven't run many iterations for your neural network yet your parameters w will be close to zero. Because with random initialization you probably initialize w to small random values so before you train for a long time, w is still quite small. And as you iterate, as you train, w will get bigger and bigger and bigger until here maybe you have a much larger value of the parameters w for your neural network. So what early stopping does is by stopping halfway you have only a mid-size rate w. And so similar to L2 regularization by picking a neural network with smaller norm for your parameters w, hopefully your neural network is over fitting less. And the term early stopping refers to the fact that you're just stopping the training of your neural network earlier.

But it does have one downside, let me explain. I think of the machine learning process as comprising several different steps. One, is that you want an algorithm to optimize the cost function j and we have various tools to do that, such as grade intersect. And then we'll talk later about other algorithms, like momentum and RMS prop and Atom and so on. But after optimizing the cost function j, you also wanted to not over-fit. And we have some tools to do that such as your regularization, getting more data and so on. Now in machine learning, we already have so many hyper-parameters it surge over. It's already very complicated to choose among the space of possible algorithms. And so I find machine learning easier to think about when you have one set of tools for optimizing the cost function J, and when you're focusing on authorizing the cost function J. All you care about is finding w and b, so that J(w,b) is as small as possible. You just don't think about anything else other than reducing this. And then it's completely separate task to not over fit, in other words, to reduce variance. And when you're doing that, you have a separate set of tools for doing it. And this principle is sometimes called orthogonalization. And there's this idea, that you want to be able to think about one task at a time. I'll say more about orthorganization in a later video, so if you don't fully get the concept yet, don't worry about it. But, to me the main downside of early stopping is that this couples these two tasks. So you no longer can work on these two problems independently, because by stopping gradient decent early, you're sort of breaking whatever you're doing to optimize cost function J, because now you're not doing a great job reducing the cost function J. You've sort of not done that that well. And then you also simultaneously trying to not over fit. So instead of using different tools to solve the two problems, you're using one that kind of mixes the two.

**And the advantage of early stopping is that running the gradient descent process just once, you get to try out values of small w, mid-size w, and large w, without needing to try a lot of values of the L2 regularization hyperparameter lambda.**

## Setting up optimization problem

### Normalizing inputs:

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/12783be9-d716-421f-978b-0453861efb52/Untitled.png)

### Why normalize inputs?

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c8062d9e-50a5-4bab-ab80-4d04664b542d/Untitled.png)

Recall that the cost function is defined as written on the top right. It turns out that if you use unnormalized input features, it's more likely that your cost function will look like this, like a very squished out bar, very elongated cost function where the minimum you're trying to find is maybe over there. But if your features are on very different scales, say the feature x_1 ranges from 1-1,000 and the feature x_2 ranges from 0-1, then it turns out that the ratio or the range of values for the parameters w_1 and w_2 will end up taking on very different values. Maybe these axes should be w_1 and w_2, but the intuition of plot w and b, cost function can be very elongated bow like that. If you plot the contours of this function, you can have a very elongated function like that. Whereas if you normalize the features, then your cost function will on average look more symmetric. If you are running gradient descent on a cost function like the one on the left, then you might have to use a very small learning rate because if you're here, the gradient decent might need a lot of steps to oscillate back and forth before it finally finds its way to the minimum. Whereas if you have more spherical contours, then wherever you start, gradient descent can pretty much go straight to the minimum. You can take much larger steps where gradient descent need, rather than needing to oscillate around like the picture on the left.

That by just setting all of them to zero mean and say variance one like we did on the last slide, that just guarantees that all your features are in a similar scale and will usually help you learning algorithm run faster.

### Vanishing and Exploring gradients

Weights initialization for deep networks:

So in practice, what you can do is set the weight matrix W for a certain layer to be np.random.randn you know, and then whatever the shape of the matrix is for this out here, and then times square root of 1 over the number of features that I fed into each neuron in layer l. So there's going to be n(l-1) because that's the number of units that I'm feeding into each of the units in layer l. It turns out that if you're using a ReLu activation function that, rather than 1 over n it turns out that, set in the variance of 2 over n works a little bit better. So you often see that in initialization, especially if you're using a ReLu activation function.

**Xavier Initialization**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fd5fadf0-8c14-4b7e-ae05-da1ec852bdd2/Untitled.png)

But in practice I think all of these formulas just give you a starting point. It gives you a default value to use for the variance of the initialization of your weight matrices. If you wish the variance here, this variance parameter could be another thing that you could tune with your hyperparameters. So you could have another parameter that multiplies into this formula and tune that multiplier as part of your hyperparameter surge. Sometimes tuning the hyperparameter has a modest size effect. It's not one of the first hyperparameters I would usually try to tune, but I've also seen some problems where tuning this helps a reasonable amount.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/83e34f6e-93df-440a-8886-d7620c63bd96/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f93030e3-924b-41f2-a934-00efb41a9049/Untitled.png)

### Lab Assignment:
#1 Initialization: 

[Initialization.ipynb](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6762e877-35a6-4edf-8277-5f55de350e45/Initialization.ipynb)

**Observations**:

- Poor initialization can lead to vanishing/exploding gradients, which also slows down the optimization algorithm.
- If you train this network longer you will see better results, but initializing with overly large random numbers slows down the optimization.
- Initializing weights to very large random values doesn't work well.

Difference:

- numpy.random.rand() produces numbers in a [uniform distribution](https://raw.githubusercontent.com/jahnog/deeplearning-notes/master/Course2/images/rand.jpg).
- numpy.random.randn() produces numbers in a [normal distribution](https://raw.githubusercontent.com/jahnog/deeplearning-notes/master/Course2/images/randn.jpg).

## **He Initialization**

Finally, try "He Initialization"; this is named for the first author of He et al., 2015. (If you have heard of "Xavier initialization", this is similar except Xavier initialization uses a scaling factor for the weights 𝑊[𝑙]W[l] of `sqrt(1./layers_dims[l-1])` where He initialization would use `sqrt(2./layers_dims[l-1])`.)

Here's a quick recap of the main takeaways:

• Different initializations lead to very different results
• Random initialization is used to break symmetry and make sure different hidden units can learn different things
• Resist initializing to values that are too large!
• He initialization works well for networks with ReLU activations

#2 **Regularization**

[Regularization.ipynb](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b0d3280f-0a8f-4c98-8b46-e0369df7f0aa/Regularization.ipynb)

**Observations**:

- The value of  is a hyperparameter that you can tune using a dev set.
- L2 regularization makes your decision boundary smoother. If  is too large, it is also possible to "oversmooth", resulting in a model with high bias.

**What is L2-regularization actually doing?**:

L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes.`

```python
X = (X < keep_prob).astype(int)

# is conceptually the same as this if-else statement 
# (for the simple case of a one-dimensional array) :

for i,v in enumerate(x):
    if v < keep_prob:
        x[i] = 1
    else: # v >= keep_prob
        x[i] = 0
```

Backpropagation with dropout is actually quite easy. You will have to carry out 2 Steps:

1. You had previously shut down some neurons during forward propagation, by applying a mask  to `A1`. In backpropagation, you will have to shut down the same neurons, by reapplying the same mask  to `dA1`.
2. During forward propagation, you had divided `A1` by `keep_prob`. In backpropagation, you'll therefore have to divide `dA1` by `keep_prob` again (the calculus interpretation is that if  is scaled by `keep_prob`, then its derivative  is also scaled by the same `keep_prob`).

**Note**:

- A **common mistake** when using dropout is to use it both in training and testing. You should use dropout (randomly eliminate nodes) only in training.
- Deep learning frameworks like [tensorflow](https://www.tensorflow.org/api_docs/python/tf/nn/dropout), [PaddlePaddle](http://doc.paddlepaddle.org/release_doc/0.9.0/doc/ui/api/trainer_config_helpers/attrs.html), [keras](https://keras.io/layers/core/#dropout) or [caffe](http://caffe.berkeleyvision.org/tutorial/layers/dropout.html) come with a dropout layer implementation. Don't stress - you will soon learn some of these frameworks.

**What you should remember about dropout:**
• Dropout is a regularization technique.
• You only use dropout during training. Don't use dropout (randomly eliminate nodes) during test time.
• Apply dropout both during forward and backward propagation.
• During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations. For example, if keep_prob is 0.5, then we will on average shut down half the nodes, so the output will be scaled by 0.5 since only the remaining half are contributing to the solution. Dividing by 0.5 is equivalent to multiplying by 2. Hence, the output now has the same expected value. You can check that this works even when keep_prob is other values than 0.5.

Note that regularization hurts training set performance! This is because it limits the ability of the network to overfit to the training set. But since it ultimately gives better test accuracy, it is helping your system.

**What we want you to remember from this notebook**:

- Regularization will help you reduce overfitting.
- Regularization will drive your weights to lower values.
- L2 regularization and Dropout are two very effective regularization techniques.

#3 **Gradient Checking**

[Gradient_Checking.ipynb](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/26330d68-1b19-43ef-bfcb-510271293c90/Gradient_Checking.ipynb)

**What you should remember from this notebook**:

• Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient (computed using forward propagation).
• Gradient checking is slow, so you don't want to run it in every iteration of training. You would usually run it only to make sure your code is correct, then turn it off and use backprop for the actual learning process.


# Week2 notes:
With the implementation of gradient descent on your whole training set, what you have to do is, you have to process your entire training set before you take one little step of gradient descent. And then you have to process your entire training sets of five million training samples again before you take another little step of gradient descent. So, it turns out that you can get a faster algorithm if you let gradient descent start to make some progress even before you finish processing your entire, your giant training sets of 5 million examples.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/950a84f2-42dc-4e49-b42b-9ed8f61eb17b/Untitled.png)

Let's say that you split up your training set into smaller, little baby training sets and these baby training sets are called mini-batches. And let's say each of your baby training sets have just 1,000 examples each. So, you take X1 through X1,000 and you call that your first little baby training set, also call the mini-batch. And then you take home the next 1,000 examples. X1,001 through X2,000 and the next X1,000 examples and come next one and so on.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cb044057-3fb0-4f34-a350-2ad29d5d0bf0/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3db7b03c-8e92-4ab3-90db-1170b9b3866d/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c9299afe-88ae-4705-878a-e9c9342e7900/Untitled.png)

**Because mini-batch gradient descent makes a parameter update after seeing just a subset of examples, the direction of the update has some variance, and so the path taken by mini-batch gradient descent will "oscillate" toward convergence. Using momentum can reduce these oscillations.**

Now one of the parameters you need to choose is the size of your mini batch. So m was the training set size on one extreme, if the mini-batch size = m, then you just end up with batch gradient descent.

Alright, so in this extreme you would just have one mini-batch X{1}, Y{1}, and this mini-batch is equal to your entire training set. So setting a mini-batch size m just gives you batch gradient descent. The other extreme would be if your mini-batch size, Were = 1.

This gives you an algorithm called stochastic gradient descent. And here every example is its own mini-batch.

So what you do in this case is you look at the first mini-batch, so X{1}, Y{1}, but when your mini-batch size is one, this just has your first training example, and you take derivative to sense that your first training example. And then you next take a look at your second mini-batch, which is just your second training example, and take your gradient descent step with that, and then you do it with the third training example and so on looking at just one single training sample at the time.

As stochastic gradient descent won't ever converge, it'll always just kind of oscillate and wander around the region of the minimum. But it won't ever just head to the minimum and stay there. In practice, the mini-batch size you use will be somewhere in between.

If you have a small training set then batch gradient descent is fine. If you go to the opposite, if you use stochastic gradient descent,

Then it's nice that you get to make progress after processing just tone example that's actually not a problem. And the noisiness can be ameliorated or can be reduced by just using a smaller learning rate. But a huge disadvantage to stochastic gradient descent is that you lose almost all your speed up from vectorization. Because, here you're processing a single training example at a time. 

The way you process each example is going to be very inefficient. So what works best in practice is something in between where you have some, mini-batch size not to big or too small.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c3c932e5-dd04-4ab4-a0fb-4d678d352e71/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2ad9b503-133c-4147-a9d3-394ea3970dd7/Untitled.png)

**I want to show you a few optimization algorithms. They are faster than gradient descent. In order to understand those algorithms, you need to be able they use something called exponentially weighted averages. Also called exponentially weighted moving averages in statistics.**

## Exponentially Weighted Averages

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/82e107a8-b7ac-48ce-8690-92b154a1f0fc/Untitled.png)

It turns out that for reasons we are going to later, when you compute this you can think of VT as approximately averaging over, something like one over one minus beta, day's temperature.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/99ef75b0-d68a-4c0f-be7d-8d4eaddc75d1/Untitled.png)

B = 0.9 - red curve

B = 0.98 - green curve

B = 0.5 - yellow curve

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e1efcbb8-c313-4e52-aee0-3b85edc15629/Untitled.png)

### **Understanding Exponentially Weighted Averages**

$***v_t = \beta \cdot v_{t-1} + (1-\beta)\cdot \theta_t***$

So you have this exponentially decaying function. And the way you compute V100, is you take the element wise product between these two functions and sum it up. So you take this value, theta 100 times 0.1, times this value of theta 99 times 0.1 times 0.9, that's the second term and so on. So it's really taking the daily temperature, multiply with this exponentially decaying function, and then summing it up. And this becomes your V100. It turns out that, up to details that are for later. But all of these coefficients, add up to one or add up to very close to one, up to a detail called bias correction which we'll talk about in the next video. But because of that, this really is an exponentially weighted average. And finally, you might wonder, how many days temperature is this averaging over. Well, it turns out that 0.9 to the power of 10, is about 0.35 and this turns out to be about one over E, one of the base of natural algorithms. And, more generally, if you have one minus epsilon, so in this example, epsilon would be 0.1, so if this was 0.9, then one minus epsilon to the one over epsilon. This is about one over E, this about 0.34, 0.35. And so, in other words, it takes about 10 days for the height of this to decay to around 1/3 already one over E of the peak. So it's because of this, that when beta equals 0.9, we say that, this is as if you're computing an exponentially weighted average that focuses on just the last 10 days temperature. Because it's after 10 days that the weight decays to less than about a third of the weight of the current day.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c08b663f-4bc9-4793-b702-ddc9abe175a1/Untitled.png)

### **Bias Correction in Exponentially Weighted Averages**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/21f7a0ee-c41d-4145-9282-33d44f280dd7/Untitled.png)

### **Gradient Descent with Momentum**

There's an algorithm called momentum, or gradient descent with momentum that almost always works faster than the standard gradient descent algorithm. In one sentence, the basic idea is to compute an exponentially weighted average of your gradients, and then use that gradient to update your weights instead.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fa4557cd-6979-43db-9114-4421b84aaf34/Untitled.png)

Beta, which controls your exponentially weighted average. The most common value for Beta is 0.9. We're averaging over the last ten days temperature. So it is averaging of the last ten iteration's gradients. And in practice, Beta equals 0.9 works very well.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bccbea85-06fa-48ef-98e2-4adc1804f731/Untitled.png)

Finally, I just want to mention that if you read the literature on gradient descent with momentum often you see it with this term omitted, with this 1 minus Beta term omitted. So you end up with vdW equals Beta vdw plus dW. And the net effect of using this version in purple is that vdW ends up being scaled by a factor of 1 minus Beta, or really 1 over 1 minus Beta. And so when you're performing these gradient descent updates, alpha just needs to change by a corresponding value of 1 over 1 minus Beta. In practice, both of these will work just fine, it just affects what's the best value of the learning rate alpha. But I find that this particular formulation is a little less intuitive. Because one impact of this is that if you end up tuning the hyperparameter Beta, then this affects the scaling of vdW and vdb as well. And so you end up needing to retune the learning rate, alpha, as well, maybe.

### **RMSprop algorithm**

You've seen how using momentum can speed up gradient descent. There's another algorithm called RMSprop, which stands for root mean square prop, that can also speed up gradient descent.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1666f62c-2d3a-4e20-ae18-fee8ef809352/Untitled.png)

So let's gain some intuition about how this works. Recall that in the horizontal direction or in this example, in the W direction we want learning to go pretty fast. Whereas in the vertical direction or in this example in the b direction, we want to slow down all the oscillations into the vertical direction. So with this terms SdW an Sdb, what we're hoping is that SdW will be relatively small, so that here we're dividing by relatively small number. Whereas Sdb will be relatively large, so that here we're dividing yt relatively large number in order to slow down the updates on a vertical dimension. And indeed if you look at the derivatives, these derivatives are much larger in the vertical direction than in the horizontal direction. So the slope is very large in the b direction, right? So with derivatives like this, this is a very large db and a relatively small dw. Because the function is sloped much more steeply in the vertical direction than as in the b direction, than in the w direction, than in horizontal direction. And so, db squared will be relatively large. So Sdb will relatively large, whereas compared to that dW will be smaller, or dW squared will be smaller, and so SdW will be smaller. So the net effect of this is that your up days in the vertical direction are divided by a much larger number, and so that helps damp out the oscillations. Whereas the updates in the horizontal direction are divided by a smaller number. So the net impact of using RMSprop is that your updates will end up looking more like this.

One fun fact about RMSprop, it was actually first proposed not in an academic research paper, but in a Coursera course that Jeff Hinton had taught on Coursera many years ago.

### **Adam Optimization Algorithm**

ADAM stands for Adaptive Moment Estimation

The Adam optimization algorithm is basically taking momentum and RMSprop, and putting them together.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/af6f8065-6788-4b2c-86e1-909a72b380a4/Untitled.png)

We did a default choice for Beta _1 is 0.9, so this is the weighted average of dw. This is the momentum-like term. The hyperparameter for Beta_2, the authors of the Adam paper inventors the Adam algorithm recommend 0.999. Again, this is computing the moving weighted average of dw squared as was db squared. The choice of Epsilon doesn't matter very much, but the authors of the Adam paper recommend a 10^minus 8, but this parameter, you really don't need to set it, and it doesn't affect performance much at all.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4fd37119-09a8-4367-855c-4fd87f59ecd3/Untitled.png)

### **Learning Rate Decay**

One of the things that might help speed up your learning algorithm is to slowly reduce your learning rate over time. We call this learning rate decay.

$\alpha = \frac{1}{1 + decayRate \times epochNumber} \alpha_{0}$

1 epoch = 1 pass through the data (whole data: including all mini-batches)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6259d174-0b83-47dd-a5a4-f905796486df/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c5b07629-55fd-4303-bbd1-15d483b7346c/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/df1b608d-db6d-4875-8bf5-75de65fb46fd/Untitled.png)

### **The Problem of Local Optima**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/42c18eb0-94f2-4117-858a-ae19fa2e8e23/Untitled.png)

Maybe you are trying to optimize some set of parameters, we call them W1 and W2, and the height in the surface is the cost function. In this picture, it looks like there are a lot of local optima in all those places. And it'd be easy for grading the sense, or one of the other algorithms to get stuck in a local optimum rather than find its way to a global optimum. It turns out that if you are plotting a figure like this in two dimensions, then it's easy to create plots like this with a lot of different local optima. And these very low dimensional plots used to guide their intuition. But this intuition isn't actually correct. It turns out if you create a neural network, most points of zero gradients are not local optima like points like this. Instead most points of zero gradient in a cost function are saddle points. So, that's a point where the zero gradient, again, just is maybe W1, W2, and the height is the value of the cost function J. But informally, a function of very high dimensional space, if the gradient is zero, then in each direction it can either be a convex light function or a concave light function. And if you are in, say, a 20,000 dimensional space, then for it to be a local optima, all 20,000 directions need to look like this. And so the chance of that happening is maybe very small, maybe two to the minus 20,000. Instead you're much more likely to get some directions where the curve bends up like so, as well as some directions where the curve function is bending down rather than have them all bend upwards.

Problem of plateuas:

If local optima aren't a problem, then what is a problem? It turns out that plateaus can really slow down learning and a plateau is a region where the derivative is close to zero for a long time. So if you're here, then gradient descents will move down the surface, and because the gradient is zero or near zero, the surface is quite flat. You can actually take a very long time, you know, to slowly find your way to maybe this point on the plateau. And then because of a random perturbation of left or right, maybe then finally I'm going to search pen colors for clarity. Your algorithm can then find its way off the plateau.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/15eb6c2a-f83f-4e56-bf4e-012618966f27/Untitled.png)

So the takeaways from this video are, first, you're actually pretty unlikely to get stuck in bad local optima so long as you're training a reasonably large neural network, save a lot of parameters, and the cost function J is defined over a relatively high dimensional space. But second, that plateaus are a problem and you can actually make learning pretty slow. And this is where algorithms like momentum or RmsProp or Adam can really help your learning algorithm as well. And these are scenarios where more sophisticated observation algorithms, such as Adam, can actually speed up the rate at which you could move down the plateau and then get off the plateau.

### Lab Assignment
Task: 

[Optimization_methods.ipynb](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/55d1615c-4b0c-46be-ba72-4b258bd7c436/Optimization_methods.ipynb)

Until now, you've always used Gradient Descent to update the parameters and minimize the cost. In this notebook, you'll gain skills with some more advanced optimization methods that can speed up learning and perhaps even get you to a better final value for the cost function. Having a good optimization algorithm can be the difference between waiting days vs. just a few hours to get a good result.

By the end of this notebook, you'll be able to:

- Apply optimization methods such as (Stochastic) Gradient Descent, Momentum, RMSProp and Adam
- Use random minibatches to accelerate convergence and improve optimization

Gradient descent goes "downhill" on a cost function 𝐽J. Think of it as trying to do this:

A variant of Gradient Descent is Stochastic Gradient Descent (SGD), which is equivalent to mini-batch gradient descent, where each mini-batch has just 1 example. The update rule that you have just implemented does not change. What changes is that you would be computing gradients on just one training example at a time, rather than on the whole training set. The code examples below illustrate the difference between stochastic gradient descent and (batch) gradient descent.

- **(Batch) Gradient Descent**:

```
X= data_input
Y= labels
m= X.shape[1]# Number of training examplesparameters= initialize_parameters(layers_dims)
for iin range(0, num_iterations):
# Forward propagationa, caches= forward_propagation(X, parameters)
# Compute costcost_total= compute_cost(a, Y)# Cost for m training examples# Backward propagationgrads= backward_propagation(a, caches, parameters)
# Update parametersparameters= update_parameters(parameters, grads)
# Compute average costcost_avg= cost_total/ m

```

- **Stochastic Gradient Descent**:

```
X= data_input
Y= labels
m= X.shape[1]# Number of training examplesparameters= initialize_parameters(layers_dims)
for iin range(0, num_iterations):
    cost_total= 0
for jin range(0, m):
# Forward propagationa, caches= forward_propagation(X[:,j], parameters)
# Compute costcost_total+= compute_cost(a, Y[:,j])# Cost for one training example# Backward propagationgrads= backward_propagation(a, caches, parameters)
# Update parametersparameters= update_parameters(parameters, grads)
# Compute average costcost_avg= cost_total/ m
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/65014592-ecd0-404f-8822-07aec2980127/Untitled.png)

There are two steps to do mini-batch:

- **Shuffle**: Create a shuffled version of the training set (X, Y) as shown below. Each column of X and Y represents a training example. Note that the random shuffling is done synchronously between X and Y. Such that after the shuffling the column of X is the example corresponding to the label in Y. The shuffling step ensures that examples will be split randomly into different mini-batches.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ceacc2ec-96ac-482a-97be-eb300a661438/Untitled.png)
    
    • **Partition**: Partition the shuffled (X, Y) into mini-batches of size `mini_batch_size` (here 64). Note that the number of training examples is not always divisible by `mini_batch_size`. The last mini batch might be smaller, but you don't need to worry about this. When the final mini-batch is smaller than the full `mini_batch_size`, it will look like this:
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ffeddc02-1a68-487a-9be4-6b220fc4f995/Untitled.png)
    
    **What you should remember**:
    
    - Shuffling and Partitioning are the two steps required to build mini-batches
    - Powers of two are often chosen to be the mini-batch size, e.g., 16, 32, 64, 128.
    
    **Because mini-batch gradient descent makes a parameter update after seeing just a subset of examples, the direction of the update has some variance, and so the path taken by mini-batch gradient descent will "oscillate" toward convergence. Using momentum can reduce these oscillations.**
    
    Momentum takes into account the past gradients to smooth out the update. The 'direction' of the previous gradients is stored in the variable 𝑣v. Formally, this will be the exponentially weighted average of the gradient on previous steps. You can also think of 𝑣v
     as the "velocity" of a ball rolling downhill, building up speed (and momentum) according to the direction of the gradient/slope of the hill.
    
    **What you should remember**:
    
    - Momentum takes past gradients into account to smooth out the steps of gradient descent. It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.
    - You have to tune a momentum hyperparameter B and a learning rate .
    
    Some advantages of Adam include:
    
    - Relatively low memory requirements (though higher than gradient descent and gradient descent with momentum)
    - Usually works well even with little tuning of hyperparameters (except α)
    
    During the first part of training, your model can get away with taking large steps, but over time, using a fixed value for the learning rate alpha can cause your model to get stuck in a wide oscillation that never quite converges. But if you were to slowly reduce your learning rate alpha over time, you could then take smaller, slower steps that bring you closer to the minimum. This is the idea behind learning rate decay.
    
    **References**:
    
    - Adam paper: [https://arxiv.org/pdf/1412.6980.pdf](https://arxiv.org/pdf/1412.6980.pdf)

# Week 3 notes:
## Hyperparameters

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/96d57908-b775-49ae-9dc5-1704e257cb3a/Untitled.png)

In deep learning, what we tend to do, and what I recommend you do instead, is choose the points at random. So go ahead and choose maybe of same number of points, right? 25 points, and then try out the hyperparameters on this randomly chosen set of points. And the reason you do that is that it's difficult to know in advance which hyperparameters are going to be the most important for your problem.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b66aadad-3ac3-4abe-bd9a-83c65e232e8d/Untitled.png)

### **Using an Appropriate Scale to pick Hyperparameters**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7f954cbf-d696-4872-b682-5e2fc47228e4/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f0a22775-31dd-4051-ac43-9c0488cc987f/Untitled.png)

### **Hyperparameters Tuning in Practice: Pandas vs. Caviar**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1b2246c6-7b3f-4741-80cf-930ae1929d2b/Untitled.png)

### Batch Normalization

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0a3f8749-c2a4-479a-8263-3c825c7ca44a/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ecfb74ef-9eae-4e21-8954-d14230dd10e3/Untitled.png)

### **Fitting Batch Norm into a Neural Network**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f5c91fc0-2a59-497c-9f4e-85cc7282e3fb/Untitled.png)

Now, you've done the computation for the first layer, where this Batch Norms that really occurs in between the computation from Z and A. Next, you take this value A1 and use it to compute Z2, and so this is now governed by W2, B2. And similar to what you did for the first layer, you would take Z2 and apply it through Batch Norm, and we abbreviate it to BN now. This is governed by Batch Norm parameters specific to the next layer. So Beta 2, Gamma 2, and now this gives you Z tilde 2, and you use that to compute A2 by applying the activation function, and so on. So once again, the Batch Norms that happens between computing Z and computing A. And the intuition is that, instead of using the un-normalized value Z, you can use the normalized value Z tilde, that's the first layer. The second layer as well, instead of using the un-normalized value Z2, you can use the mean and variance normalized values Z tilde 2.

The authors of the Adam paper use Beta on their paper to denote that hyperparameter, the authors of the Batch Norm paper had used Beta to denote this parameter, but these are two completely different Betas. I decided to stick with Beta in both cases, in case you read the original papers. But the Beta 1, Beta 2, and so on, that Batch Norm tries to learn is a different Beta than the hyperparameter Beta used in momentum and the Adam and RMSprop algorithms.

Same as we did on the previous slide using the parameters W1, B1 and then you take just this mini-batch and computer mean and variance of the Z1 on just this mini batch and then Batch Norm would subtract by the mean and divide by the standard deviation and then re-scale by Beta 1, Gamma 1, to give you Z1, and all this is on the first mini-batch, then you apply the activation function to get A1, and then you compute Z2 using W2, B2, and so on. So you do all this in order to perform one step of gradient descent on the first mini-batch and then goes to the second mini-batch X2, and you do something similar where you will now compute Z1 on the second mini-batch and then use Batch Norm to compute Z1 tilde. And so here in this Batch Norm step, You would be normalizing Z tilde using just the data in your second mini-batch, so does Batch Norm step here. Let's look at the examples in your second mini-batch, computing the mean and variances of the Z1's on just that mini-batch and re-scaling by Beta and Gamma to get Z tilde, and so on. And you do this with a third mini-batch, and keep training.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/69fe370a-3e9e-4d6e-83fe-6e4de3be93ef/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/15da51fc-fbe9-4ee3-b409-ae76d2bc62f9/Untitled.png)

COVARIATE SHIFT - So, this idea of your data distribution changing goes by the somewhat fancy name, covariate shift. And the idea is that, if you've learned some X to Y mapping, if the distribution of X changes, then you might need to retrain your learning algorithm. And this is true even if the function, the ground true function, mapping from X to Y, remains unchanged, which it is in this example, because the ground true function is, is this picture a cat or not. And the need to retain your function becomes even more acute or it becomes even worse if the ground true function shifts as well.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5c9ff150-252c-4c26-988b-876b991ccfe0/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ff615527-292c-449a-b0f5-ca8fa251b2a6/Untitled.png)

## Softmax Regression

$a^{[L]} = \frac{e^{Z^{[L]}}}{\sum_{i=1}^{4} t_{i}}$

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a4c958a8-bf26-444d-ae78-dfe53bb37dc4/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0f99aa77-124a-4545-a29a-c60456c55532/Untitled.png)

### **Training a Softmax Classifier**

The name softmax comes from contrasting it to what's called a hard max which would have taken the vector Z and matched it to this vector. So hard max function will look at the elements of Z and just put a 1 in the position of the biggest element of Z and then 0s everywhere else. And so this is a very hard max where the biggest element gets a output of 1 and everything else gets an output of 0. Whereas in contrast, a softmax is a more gentle mapping from Z to these probabilities. So, I'm not sure if this is a great name but at least, that was the intuition behind why we call it a softmax, all this in contrast to the hard max.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/60a341ec-ba09-43d7-a135-596f314dc7b8/Untitled.png)

## **Deep Learning Frameworks**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/09b8896c-dad5-4399-9b76-552ed1bd16ab/Untitled.png)

### Tensorflow

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cdec95a4-4c0c-4109-8adc-dbb33217f15e/Untitled.png)

### Lab Assignment
Description: In this notebook, you'll explore TensorFlow, a deep learning framework that allows you to build neural networks more easily, and use it to build a neural network and train it on a TensorFlow dataset.

Notebook:

[Tensorflow_introduction.ipynb](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/936ca49b-f4ec-4982-8b46-684c37a0eea3/Tensorflow_introduction.ipynb)

References:

Introduction to Gradients and Automatic Differentiation: [https://www.tensorflow.org/guide/autodiff](https://www.tensorflow.org/guide/autodiff)

GradientTape documentation: [https://www.tensorflow.org/api_docs/python/tf/GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape)