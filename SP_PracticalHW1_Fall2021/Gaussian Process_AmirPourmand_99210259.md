### Required imports


```python
from math import exp
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
```

### Set your student number as random seed


```python
np.random.seed(99210259)
```


```python
%config InlineBackend.figure_formats = ['svg']
np.set_printoptions(suppress=True)
```

# Gaussian Process

1. Generate 10 Gaussian processes with $\mu(t)=2t^2$ and rbf kernel: $e^{-(x_1-x_2)^2/{2\sigma^2}}$ with $\sigma^2$=0.05, in the interval [-2, 2].
2. Show the processes as well as $\mu(t)$ in a single plot. (set the number of samples as large as the graphs seem smooth)


```python
def generate_gaussian_process(time_span,no_of_functions,mean_function,kernel_function):
    
    mean_vector = mean_function(time_span)
    covariance=kernel_function(time_span[:,None], time_span[None,:])

    return np.random.multivariate_normal(mean=mean_vector,cov=covariance,size=no_of_functions,)
    
def plot_functions(X,list_of_functions):
    plt.figure(figsize=(11,5))
    for item in range(len(list_of_functions)):
        plt.plot(X,list_of_functions[item],'-')
```


```python
mean_function = lambda x: 2*(x**2)
kernel_function= lambda x1,x2: np.exp((-1/2) * ((x1-x2)**2) / 0.05)

time_span = np.linspace(-2,2,num=500)
gaussian_process_list=generate_gaussian_process(time_span,no_of_functions=10,
                         mean_function=mean_function, kernel_function=kernel_function)


plot_functions(time_span,gaussian_process_list)
plt.plot(time_span,mean_function(time_span),'*')
```




    [<matplotlib.lines.Line2D at 0x7effaa617ee0>]




    
![svg](Gaussian%20Process_AmirPourmand_99210259_files/Gaussian%20Process_AmirPourmand_99210259_8_1.svg)
    


# Brownian Motion

Brownian motion is a type of Gaussian process in which the change of its value through time completely random.  
A standard Brownian motion as a Gaussian process has zero mean, and kernel $k(s, t) = \min{(s, t)}$

1. Generate 10 Brownian motions in the interval [0, 1].
2. Draw the processes in a single plot with different colors.


```python
time_span = np.linspace(start=0,stop=1,num=500)

mean_function =lambda x: np.zeros(shape=x.shape)
kernel_function = lambda x1,x2: np.where(x1-x2>0,x2,x1) #implementation of min in numpy

brownian_motion_list=generate_gaussian_process(time_span,no_of_functions=10,
                         mean_function=mean_function, kernel_function=kernel_function)


plot_functions(time_span,brownian_motion_list)

plt.plot(time_span,mean_function(time_span),'*')
```




    [<matplotlib.lines.Line2D at 0x7effa9f44c40>]




    
![svg](Gaussian%20Process_AmirPourmand_99210259_files/Gaussian%20Process_AmirPourmand_99210259_12_1.svg)
    


### Shifting property

* One of the properties of Brownian motions is memorylessness (like poisson processes).
* They are also time and space homogenous which means change in value is independent of current time and value. 
* For any $t_1 < t_2$, the distribution of $X(t_2)-X(t_1)$ is independent of $t_1$ and $X(t)$ for $t \leq t_1$. This means starting from any point in a Brownian motion, what we observe afterwards is also a Brownian motion.
* To see this, extract the second half of all 10 processes and visualize them after translating their starting points to the center.


```python
half = len(time_span)//2
plt.figure(figsize=(11,5))
for i in range(len(brownian_motion_list)):
    plt.plot(time_span[half:]-time_span[half],brownian_motion_list[i,half:]-brownian_motion_list[i,half])
    
plt.plot(time_span[half:]-time_span[half],mean_function(time_span[half:]-time_span[half]),'*')
```




    [<matplotlib.lines.Line2D at 0x7effa9e73550>]




    
![svg](Gaussian%20Process_AmirPourmand_99210259_files/Gaussian%20Process_AmirPourmand_99210259_15_1.svg)
    


### Scaling property

* Brownian motion has scaling property. It means that for if $B_t$ is a Brownian motion and $s > 0$, the process $\frac{1}{\sqrt{s}}B_{st}$ is also a Brownian motion. 
* Generate scaled Brownian process from the initially generated 10 processes with $s = 5$ and plot it. 


```python
# TODO
plt.figure(figsize=(11,5))
for i in range(len(brownian_motion_list)):
    plt.plot(time_span, brownian_motion_list[i] / np.sqrt(5))
    
plt.plot(time_span, mean_function(time_span) / np.sqrt(5),'*')
```




    [<matplotlib.lines.Line2D at 0x7effa9df50a0>]




    
![svg](Gaussian%20Process_AmirPourmand_99210259_files/Gaussian%20Process_AmirPourmand_99210259_18_1.svg)
    


### Inversion property

Brownian motions have inversion property, for both muliplication and addition:
1. Additive inversion: $B_{1-t} - B_1$ is also a Brownian motion.
2. Multiplicative inversion: $tB(\frac{1}{t}), t > 0$ is also a Brownian motion.

Plot inversed (both additive and multiplicative) of original 10 Brownian motions to observe this


```python
# TODO
# TODO
plt.figure(figsize=(11,5))
for i in range(len(brownian_motion_list)):
    plt.plot(time_span[::-1], brownian_motion_list[i,::-1] - brownian_motion_list[i,-1:])
    
plt.plot(time_span[::-1], mean_function(time_span),'*')
```




    [<matplotlib.lines.Line2D at 0x7effa9d7a370>]




    
![svg](Gaussian%20Process_AmirPourmand_99210259_files/Gaussian%20Process_AmirPourmand_99210259_21_1.svg)
    


# Gaussian process with different kernels


```python

```


```python
def draw_zero_mean(kernel_function,title):
    time_span = np.linspace(0,1,num=500)
    mean_function = lambda x: np.zeros(x.shape[0])
    gaussian_process_list1=generate_gaussian_process(time_span,no_of_functions=3,
                             mean_function=mean_function, kernel_function=kernel_function)
    plt.figure()
    for item in range(len(gaussian_process_list1)):
        plt.plot(time_span,gaussian_process_list1[item],'-')
        plt.title(title,fontsize=15)

        
draw_zero_mean(lambda x1,x2: np.arcsin(x1*x2), 
               '1. $k_{w, a}(s, t) = \\arcsin\\left(\\frac{w(s-a)(t-a)}{\\sqrt{(w(s-a)(s-a)+1)(w(t-a)(t-a)+1)}}\\right)$')
draw_zero_mean(lambda x1,x2: np.where(x1==x2,1,0),'2. $k(s, t) = \\delta(s-t)$')
draw_zero_mean(lambda x1,x2: x1*x2, '3. $k_a(s, t) = (s-a)(t-a)$')
draw_zero_mean(lambda x1,x2: np.where(x1-x2<0,x1,x2), '4. $k(s, t) = \min(s, t)$')
draw_zero_mean(lambda x1,x2: np.exp(-1*(x1-x2)**2/2), '5. $k_{\\sigma}(s, t) = e^{\\frac{-(s-t)^2}{2\\sigma^2}}$\n')
draw_zero_mean(lambda x1,x2: np.where(x1-x2<0,x1,x2) - x1*x2, '6. $k(s, t) = \\min(s, t) - st$')
draw_zero_mean( lambda x1,x2: np.exp(-1*np.abs(x1-x2)/1),'7. $k_l(s, t) = e^{-\\frac{|s-t|}{l}}$')

```


    
![png](Gaussian%20Process_AmirPourmand_99210259_files/Gaussian%20Process_AmirPourmand_99210259_24_0.png)
    



    
![png](Gaussian%20Process_AmirPourmand_99210259_files/Gaussian%20Process_AmirPourmand_99210259_24_1.png)
    



    
![png](Gaussian%20Process_AmirPourmand_99210259_files/Gaussian%20Process_AmirPourmand_99210259_24_2.png)
    



    
![png](Gaussian%20Process_AmirPourmand_99210259_files/Gaussian%20Process_AmirPourmand_99210259_24_3.png)
    



    
![png](Gaussian%20Process_AmirPourmand_99210259_files/Gaussian%20Process_AmirPourmand_99210259_24_4.png)
    



    
![png](Gaussian%20Process_AmirPourmand_99210259_files/Gaussian%20Process_AmirPourmand_99210259_24_5.png)
    



    
![png](Gaussian%20Process_AmirPourmand_99210259_files/Gaussian%20Process_AmirPourmand_99210259_24_6.png)
    


Suppose all following processes are Gaussian with zero mean. Choose their corresponding kernel from the following options (one option is extra).

Write the results in a $3\times2$ numpy array and store it as 'gaussian_choice.npy', e.g. [[1, 2], [3, 4], [5, 7]]

1. $k_{w, a}(s, t) = \arcsin\left(\frac{w(s-a)(t-a)}{\sqrt{(w(s-a)(s-a)+1)(w(t-a)(t-a)+1)}}\right)$
2. $k(s, t) = \delta(s-t)$
3. $k_a(s, t) = (s-a)(t-a)$
4. $k(s, t) = \min(s, t)$
5. $k_{\sigma}(s, t) = e^{\frac{-(s-t)^2}{2\sigma^2}}$
6. $k(s, t) = \min(s, t) - st$
7. $k_l(s, t) = e^{-\frac{|s-t|}{l}}$




```python
plt.figure(figsize=(25, 25))
plt.imshow(Image.open('gaussian.png'))
```




    <matplotlib.image.AxesImage at 0x7f0344bcdc40>




    
![png](Gaussian%20Process_AmirPourmand_99210259_files/Gaussian%20Process_AmirPourmand_99210259_27_1.png)
    



```python
# TODO
np.save('gaussian_choice.npy',np.array([[5,6],[4,2],[3,1]]))
```


```python
np.load('gaussian_choice.npy')
```




    array([[5, 6],
           [4, 2],
           [3, 1]])



# Inference

In this section you are given samples from a Gaussian process and are supposed to infer the value of the process in other points using the available samples.

The samples are available in 'gaussian_infer_samples.npy' as a $2\times 50$ array, first row are time values and second row are the process's values.

Samples are generated from a Gaussian process with constant mean $\mu=2$ and rbf kernel: 
$k(s, t) = e^{\frac{-(s-t)^2}{\sigma^2}}$ with $\sigma^2=0.01$ plus a white noise with variance $\delta^2=0.01$.

1. Estimate process value at all points from 0 to 10 with step 0.01.
2. Store the estimated values in a numpy array (in the same form as samples) and save them as 'gaussian_infer_result.npy'
3. Plot the inferred Gaussian process as well as the input samples


```python
# TODO
samples = np.load('gaussian_infer_samples.npy')
```


```python
def GP(X1, y1, X2, kernel_func,noise):
    Σ11 = kernel_func(X1[:,None], X1[None,:])+((noise) * np.eye(X1.shape[0]))
    Σ12 = kernel_func(X1[:,None], X2[None,:])
    solved = (np.linalg.inv(Σ11) @ Σ12).T
    μ2 = solved @ y1
    Σ22 = kernel_func(X2[None,:], X2[:,None])
    Σ2 = Σ22 - (solved @ Σ12)
    return μ2, Σ2
```


```python
kernel_function= lambda x1,x2: np.exp((-1) * ((x1-x2)**2) / 0.01)
time = np.arange(start=0,stop=10,step=0.01)
mean,cov=GP(samples[0],samples[1],time,kernel_function,0.01)
variance = np.sqrt(np.diag(cov))

result=np.random.multivariate_normal(mean=mean,cov=cov)
fig,ax =plt.subplots(figsize=(15,5))
ax.fill_between(time.flat, mean-2*variance, mean+2*variance, color='blue', alpha=0.1)
ax.plot(time,result)
ax.plot(samples[0],samples[1],'.')
```




    [<matplotlib.lines.Line2D at 0x7fda781d0ee0>]




    
![png](Gaussian%20Process_AmirPourmand_99210259_files/Gaussian%20Process_AmirPourmand_99210259_36_1.png)
    



```python
np.save('gaussian_infer_result.npy',(time,result))
```


```python
np.load('gaussian_infer_result.npy')
```




    array([[ 0.        ,  0.01      ,  0.02      , ...,  9.97      ,
             9.98      ,  9.99      ],
           [-0.50462431, -0.64668425, -0.7790506 , ..., -0.8491067 ,
            -1.02993568, -1.18138433]])




```python

```
