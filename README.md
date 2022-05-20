# Multilayer_Perceptron by Rust

## TODO

* 增加矩阵广播

Relu:
$$f(x) =\begin{cases} x&x>0\\0 & x\le0\end{cases}$$
$$f\prime(x) =\begin{cases} 1&x>0\\0&x\le0\end{cases}$$
Sigmoid:
$$\sigma(x) = \frac{1}{1+e^x}$$
$$\sigma\prime(x)=\sigma(x)(1-\sigma(x))$$
Softmax:
$$f(x) = \frac{e^{x_i}}{\sum_{j}e^{x_j}}$$
$$f\prime(x)= f(x_i)(1 - f(x_i)) + \sum_{j\neq i} f(x_j) f(x_i) $$
Cross Entropy:
$$f(\hat{y}, y) = -\frac{1}{n}\sum^n_{i=1}y_i\ln\hat{y}_i$$
$$f_i\prime(\hat{y}, y) = \hat{y_i} - y_i$$
