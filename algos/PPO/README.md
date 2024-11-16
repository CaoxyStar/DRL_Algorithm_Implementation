# PPO Algo

I tend to think of the PPO as an on-policy algorithm, although this is a debate. Because its critic network need to fit V-function rather than Q-function, so it cannot exploit the experience from the earlier past. And the importance sampling is just used to alleviate this problem, enabling a batch of experience could be trained multiple times.

Overall, this is a implementation of PPO with Monte Carlo, which is unbiased but has high variance.