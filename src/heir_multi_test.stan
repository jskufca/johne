//
// This Stan program defines a simple model 
// that will examine specificity, sensitivy, and prevalence by farm
// with multiple tests  
//
// allowing that some farms are disease free.
// 
// guided by model from
// https://mc-stan.org/docs/2_21/stan-users-guide/data-coding-and-diagnostic-accuracy-models.html
// modified for specificity and sensitivity priors by beta distribution.
// along with model from
//
// Ottawa paper
// and 
// model from Ozsvari et al



// The input data assumes global pooling, with some number of positive results y from sample of size n.
data {
  // K=2 classes, negative (0) and positive (1)
  int<lower=1> F; //number of farms
  int<lower=1> I; //total number of animals tested
  int<lower=1> I_s[F]; //starting index for animals from from f
  int<lower=1> I_e[F]; //ending index for animals from from f
  
  int<lower=1> J; // number of tests - 3 for our work
  int<lower=0,upper=1> y[I,J]; // test resuls  0 for neg, 1 for pos
  // params for prior for sensitivities and specificities
  row_vector<lower=0>[J] alpha_sens;
  row_vector<lower=0>[J] beta_sens;
  row_vector<lower=0>[J] alpha_spec;
  row_vector<lower=0>[J] beta_spec;
  real htp_alpha;
  real htp_beta;
  real b1_mu;
  real b1_sd;
  real eta_mu;
  real eta_sd;
  // more prior for heirarchical and for prev and none mixture
  
  
  
}
parameters {
  real beta1;  // population level parameter
  real eta[F]; // Herd level random effects
//  real<lower=0> sigmasq; // Variance of the herd level random effects
  real<lower=0.001, upper=0.999> HTP; // Herd true prevalence  
     
  real<lower=0, upper=1> spec[J];
  real<lower=0, upper=1> sens[J];
}



// transformed parameters {
//   real<lower=0> sigma; // Standard deviation of the herd level random effects
// 
//   sigma = sqrt(sigmasq);
// }


// transformed parameters{
//   simplex[2] theta[J,2];
//   vector[2] log_q_z[I];
//   for (j in 1:J){
//     theta[j,1]=[spec[j],1-spec[j]]';
//     theta[j,2]=[sens[j],1-sens[j]]';
//   }
//   for (i in 1:I){
//        log_q_z[i]=log(pi);
//        for (j in 1:J)
//          for (k in 1:2)
//            log_q_z[i,k]=log_q_z[i, k]+log(theta[j, k, y[i, j]]);
//   }
// }
model {
     vector[F] CWHP1; // Conditional within-herd animal-level prevalence 
  
  real pi1;    // Apparent prevalence 
  real t1;
  real t2;  
   
  // priors 
  for (j in 1:J){
       spec[j] ~ beta(alpha_spec[j], beta_spec[j]);
       sens[j] ~ beta(alpha_sens[j], beta_sens[j]);
  }
  beta1  ~ normal(b1_mu, b1_sd);
  eta ~ normal(eta_mu, eta_sd); // Vectorized
  
  HTP ~ beta(htp_alpha, htp_beta); // 
  
  
  // primary loop through farms
  
  for (n in 1:F) {
    t1 = 0; // Loglikelihood component if herd n is supposed to be infected
    t2 = 0; // Loglikelihood component if herd n is not supposed to be infected
    CWHP1[n] = inv_logit(beta1+eta[n]);


    // for each animal in the herd
    for (k in I_s[n]:I_e[n]) {
         
         for (j in 1:J) {     // for each test
              pi1   = sens[j]*CWHP1[n] + (1-spec[j])*(1-CWHP1[n]);// Apparent prevalence
              t1 += bernoulli_lpmf( y[k,j] |  pi1);
              t2 += bernoulli_lpmf( y[k,j] |  1-spec[j]);
         }
    }

    // The unconditional loglikelihood component of herd n is added to the loglikelihood total
    target += log_mix(HTP, t1, t2);
  }
  
  
}
generated quantities {
     vector[F] CWHP; //conditional within herd prevalence
     vector[F] EWHP; // expectation of within herd prevalece
     real pi1;
     real t1;
     real t2;
     
  vector[2] HID[F]; //Herd Infection distribution
  vector[2] thismix;
  
  for (n in 1:F) {
    t1 = 0; // Loglikelihood component if herd n is supposed to be infected
    t2 = 0; // Loglikelihood component if herd n is not supposed to be infected
    CWHP[n] = inv_logit(beta1+eta[n]);


    // for each animal in the herd
    for (k in I_s[n]:I_e[n]) {
         
         for (j in 1:J) {     // for each test
              pi1   = sens[j]*CWHP[n] + (1-spec[j])*(1-CWHP[n]);// Apparent prevalence
              t1 += bernoulli_lpmf( y[k,j] |  pi1);
              t2 += bernoulli_lpmf( y[k,j] |  1-spec[j]);
         }
    }

    thismix=[HTP * t1, (1-HTP) * t2 ]';
    HID[n] = softmax(thismix);
    EWHP[n]=HID[n,1]*CWHP[n];
  }

}
