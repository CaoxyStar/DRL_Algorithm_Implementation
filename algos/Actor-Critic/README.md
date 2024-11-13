# Actor-Critic Algo

The actor-critic algorithm can be implemented by various versions, such as off-policy or on-policy, TD or MC and online or offline.

For the basic actor-critic algorithm, the common way is fitting the V-function by TD(low variance) or MC(unbiased), and then computing action advantage based on V-function to train actor. However, this version is a on-policy paradigm lacking data efficiency.

Thus this is a off-policy implementation with TD. To let latest actor can train with old experience in this off-policy version, we need to fit Q-function instead of V-function and train actor with Q directly. Because the Q-function can eliminate the influence of different actions from old and latest actors.