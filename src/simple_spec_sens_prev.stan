//
// This Stan program defines a simple model 
// that will examine specificity, sensitivy, and prevalence 
// on a basic example disease problem.
// 
// guided by model from https://www.medrxiv.org/content/10.1101/2020.05.22.20108944v1.full
// modified for specificity and sensitivity priors by beta distribution.



// The input data assumes global pooling, with some number of positive results y from sample of size n.
data {
  int y_sample; // number of positives
  int n_sample; // number of samples
  // params for prior
  real<lower=0> alpha_sens;
  real<lower=0> beta_sens;
  real<lower=0> alpha_spec;
  real<lower=0> beta_spec;
}
parameters {
  real<lower=0, upper=1> prev;
  real<lower=0, upper=1> spec;
  real<lower=0, upper=1> sens;
}
model {
  real p_sample;
  p_sample=prev*sens+(1-prev)*(1-spec);
  y_sample ~ binomial(n_sample, p_sample);
  spec ~ beta(alpha_spec, beta_spec);
  sens ~ beta(alpha_sens, beta_sens);
}

