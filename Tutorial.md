## ECAP Python Tutorial

ecap: Excess Certainty Adjusted Probability Estimate

The "ecap" package mplements the Excess Certainty Adjusted Probability adjustment procedure as described in the paper "Irrational Exuberance: Correcting Bias in Probability Estimates" by Gareth James, Peter Radchenko, and Bradley Rava (Journal of the American Statistical Association, 2020; <doi:10.1080/01621459.2020.1787175>). The package includes a function that preforms the ECAP adjustment and a function that estimates the parameters needed for implementing ECAP. 

For an in-depth discussion of the method. Please review the paper at http://faculty.marshall.usc.edu/gareth-james/Research/Probs.pdf. 

The package can be downloaded via pip install.

```python
pip install ecap
```

## Implimenting the ECAP procedure
To impliment the ecap procedure, we first must train the model with a set of probabilities and their eventual outcomes.

For simplicity, here I have generated probabilities from a uniform distribution between 0 & 1 and generated eventual outcomes through random binomial trials with the probabilities generated previously. 
```python
unadjusted_prob = np.random.uniform(0,1,1000)
win_var = np.random.binomial(1, unadjusted_prob, 1000)
```

This is enough to call the package.

We can specify a number of options. The most important are supplying the ecap function with the probabilities and their outcomes. However, we also can specify if we want the biased or unbiased method of ecap to run through bias_indicator.

win_var is just a variable that tells the function what a "win" looks like. Typically, 1 means win and 0 means a loss, however we will let the user specify this.

```python
import ecap
ecap_fit = ecap.ecap(unadjusted_prob, win_var=1, 1, bias_indicator=True)
```

If needed, there are also other arguments that the function can take in. Namely, we can adjust the grids that the ecap algorithm searches over when picking the optimal parameters. There are already grids included by default, however the user might desire to refine these for better estimations.

If so, you can adjust the arguments in ecap.
- lambda_grid (flexibility in the ecap estimate)
- gamma_grid (corruption levels in ones probability estimates)
- theta_grid (overall bias in ones probability estimates)

To test the estimation procedure, ecap_fit contains a lot of diagnostic information. Namely,
- the optimal choice of lambda (flexibility determined by a cross-validated risk)
- the optimal choice of gamma (corruption in ones estimates)
- the optimal choice of theta (overall bias in ones estimates)
- estimate of the function g on 0-0.5 (see the paper for more of a discussion on this function). 

You can access these values through the dictionary output:
```python
ecap_fit['lambda_opt']
ecap_fit['gamma_opt']
ecap_fit['theta_opt']
ecap_fit['g_hat']
```

With a new set of probability estimates, we can use the model defined above to make an ecap adjustment.

```python
p_new = np.random.uniform(0,1,1000)
ecap_probs = ecap.predict(ecap_fit, p_new)
```

ecap_probs contains the adjusted new ecap adjusted probability estimates.

This package is always evolving. Please email me if you notice any issues or have any questions. brava@marshall.usc.edu