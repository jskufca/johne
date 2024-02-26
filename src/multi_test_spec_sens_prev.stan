//
// This Stan program defines a simple model 
// that will examine specificity, sensitivy, and prevalence 
// on a basic example disease problem.
// 
// guided by model from
// https://mc-stan.org/docs/2_21/stan-users-guide/data-coding-and-diagnostic-accuracy-models.html
// modified for specificity and sensitivity priors by beta distribution.



// The input data assumes global pooling, with some number of positive results y from sample of size n.
data {
  // K=2 classes, negative (1) and positive (2)
  int<lower=1> I; //number of animals tested
  int<lower=1> J; // number of tests - 3 for our work
  int<lower=1,upper=2> y[I,J]; // test resuls  1 for neg, 2 for pos
  // params for prior
  row_vector<lower=0>[J] alpha_sens;
  row_vector<lower=0>[J] beta_sens;
  row_vector<lower=0>[J] alpha_spec;
  row_vector<lower=0>[J] beta_spec;
}
parameters {
  simplex[2] pi;
  real<lower=0, upper=1> spec[J];
  real<lower=0, upper=1> sens[J];
}
transformed parameters{
  simplex[2] theta[J,2];
  vector[2] log_q_z[I];
  for (j in 1:J){
    theta[j,1]=[spec[j],1-spec[j]]';
    theta[j,2]=[sens[j],1-sens[j]]';
  }
  for (i in 1:I){
       log_q_z[i]=log(pi);
       for (j in 1:J)
         for (k in 1:2)
           log_q_z[i,k]=log_q_z[i, k]+log(theta[j, k, y[i, j]]);
  }
}
model {
  for (j in 1:J){
       spec[j] ~ beta(alpha_spec[j], beta_spec[j]);
       sens[j] ~ beta(alpha_sens[j], beta_sens[j]);
  }     
  for (i in 1:I)
    target += log_sum_exp(log_q_z[i]);
}
generated quantities {
  vector[2] z[I];
  for (i in 1:I)
    z[i]=softmax(log_q_z[i]);
}
