//
// This Stan program defines a simple model 
// that will examine specificity, sensitivy, and prevalence by farm
//
// allowing that some farms are disease free.
// 
// guided by model from
// Ozsvari et al
// 



// The input data assumes global pooling, with some number of positive results y from sample of size n.
data {
  // K=2 classes, negative (0) and positive (1)
  int<lower=1> F; //number of farms
  int<lower=1> I; //total number of animals tested
  int<lower=1> I_s[F]; //starting index for animals from from f
  int<lower=1> I_e[F]; //ending index for animals from from f

  int<lower=0,upper=1> y[I]; // test resuls  0 for neg, 1 for pos
  // params for prior for sensitivities and specificities
  real<lower=0> alpha_sens;
  real<lower=0> beta_sens;
  real<lower=0> alpha_spec;
  real<lower=0> beta_spec;
  // more prior for heirarchical and for prev and none mixture
}
parameters {
  real beta1;  // population level parameter
  real eta[F]; // Herd level random effects
  real<lower=0> sigmasq; // Variance of the herd level random effects
  real<lower=0.001, upper=0.999> HTP; // Herd true prevalence
  
  real<lower=0, upper=1> Sp;
  real<lower=0, upper=1> Se;
}
transformed parameters {
  real<lower=0> sigma; // Standard deviation of the herd level random effects

  sigma = sqrt(sigmasq);
}


model {
  vector[F] CWHP1; // Conditional within-herd animal-level prevalence 
  
  real pi1;    // Apparent prevalence 
  real t1;
  real t2;
  

  Sp ~ beta(alpha_spec, beta_spec);
  Se ~ beta(alpha_sens, beta_sens);
  beta1  ~ normal(0, 10);
  sigmasq ~ inv_gamma(0.1, 0.1);
  eta ~ normal(0, 3); // Vectorized

  // for (n in 1:F) {
  //   CWHP1[n] = inv_logit(beta1+sigma*eta[n]);
  // }

  // Herd true prevalence, beta prior 
  HTP ~ beta(3, 5); // 

  // The components of the loglikelihood are calculated for each herd 
  for (n in 1:F) {
    t1 = 0; // Loglikelihood component if herd n is supposed to be infected
    t2 = 0; // Loglikelihood component if herd n is not supposed to be infected
    CWHP1[n] = inv_logit(beta1+sigma*eta[n]);


    // for each animal in the herd
    for (k in I_s[n]:I_e[n]) {

      // Apparent prevalence
      pi1   = Se*CWHP1[n] + (1-Sp)*(1-CWHP1[n]);

      t1 += bernoulli_lpmf( y[k] |  pi1);
      t2 += bernoulli_lpmf( y[k] |  1-Sp);
    }
    

    // The unconditional loglikelihood component of herd n is added to the loglikelihood total
    target += log_mix(HTP, t1, t2);
  }
}
generated quantities {
     vector[F] CWHP; //conditional within herd prevalence
     
     real pi1;
     real t1;
     real t2;
     
  vector[2] HID[F]; //Herd Infection distribution
  vector[2] thismix;
  
    for (n in 1:F) {
         CWHP[n] = inv_logit(beta1+sigma*eta[n]);
         
         t1 = 0; // Loglikelihood component if herd n is supposed to be infected
         t2 = 0; // Loglikelihood component if herd n is not supposed to be infected


    // for each animal in the herd
    for (k in I_s[n]:I_e[n]) {

      // Apparent prevalence
      pi1   = Se*CWHP[n] + (1-Sp)*(1-CWHP[n]);

      t1 += bernoulli_lpmf( y[k] |  pi1);
      t2 += bernoulli_lpmf( y[k] |  1-Sp);
    }
         
         
         thismix=[HTP * t1, (1-HTP) * t2 ]';
    HID[n] = softmax(thismix);
  }
  
}

