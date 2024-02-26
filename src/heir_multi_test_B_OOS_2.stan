//
// This Stan program defines a simple model 
// that will examine specificity, sensitivy, and prevalence by farm
// with multiple tests
// 
// Version B use parameterization of between farm variation using beta distribution
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
// model of between farm variabiliy as in Liapi et all (2011)
//
// OOS - OUT OF SAMPLE
//    Modifies the base procedure as follows:
//    The last farm is "Fake" and may have missing data on some tests, 
//    Encoded as y_ij = -1
//



// The input data assumes global pooling, with some number of positive results y from sample of size n.
data {
     // K=2 classes, negative (0) and positive (1)
     int<lower=1> F; //number of farms
     int<lower=1> I; //total number of animals tested
     int<lower=1> I_s[F]; //starting index for animals from from f
     int<lower=1> I_e[F]; //ending index for animals from from f
     
     int<lower=1> J; // number of tests - 3 for our work
     int<lower=0,upper=1> y[I,J]; // test resuls  0 for neg, 1 for pos
     int<lower=0,upper=1> v[I,J]; // validity 1 for valid, 0 for invalid
     // params for prior for sensitivities and specificities
     row_vector<lower=0>[J] alpha_sens;
     row_vector<lower=0>[J] beta_sens;
     row_vector<lower=0>[J] alpha_spec;
     row_vector<lower=0>[J] beta_spec;
     real htp_alpha; // prior  (HTP is tau from Liapi)
     real htp_beta; // prior
     real mu_alpha;// prior
     real mu_beta;// prior
     real psi_a;// prior
     real psi_b;// prior
     // more prior for heirarchical and for prev and none mixture
     
     
}
parameters {
     real<lower=0, upper=1> CWHP[F]; // equivalent of pi_star[k]
     real<lower=0, upper=1> mu; // mean true prevalence of infected herds
     real<lower=0> psi; //variability parameter for prev
     //  real<lower=0> sigmasq; // Variance of the herd level random effects
     real<lower=0, upper=1> HTP; // Herd true prevalence  
     
     real<lower=0, upper=1> spec[J];
     real<lower=0, upper=1> sens[J];
}


model {
     real pi1;    // Apparent prevalence 
     real t1;
     real t2;  
     
     // priors 
     for (j in 1:J){
          spec[j] ~ beta(alpha_spec[j], beta_spec[j]);
          sens[j] ~ beta(alpha_sens[j], beta_sens[j]);
     }
     HTP ~ beta(htp_alpha, htp_beta); 
     mu ~ beta(mu_alpha, mu_beta);
     psi ~ gamma(psi_a, psi_b);
     CWHP ~ beta(mu*psi,psi*(1-mu));
     //  eta ~ normal(eta_mu, eta_sd); // Vectorized
     
     
     // primary loop through farms
     
     for (n in 1:F) {
          t1 = 0; // Loglikelihood component if herd n is supposed to be infected
          t2 = 0; // Loglikelihood component if herd n is not supposed to be infected
          
          // for each animal in the herd
          for (k in I_s[n]:I_e[n]) {
               
               for (j in 1:J) {     // for each test
                    if (v[k,j] ==1) {
                         pi1   = sens[j]*CWHP[n] + (1-spec[j])*(1-CWHP[n]);// Apparent prevalence
                         t1 += bernoulli_lpmf( y[k,j] |  pi1);
                         t2 += bernoulli_lpmf( y[k,j] |  1-spec[j]);
                    }               
               }
          }
          
          // The unconditional loglikelihood component of herd n is added to the loglikelihood total
          target += log_mix(HTP, t1, t2);
     }
     
     
}
generated quantities {
     
     real InfP=0; // average prev in an infected herd
     
     real pi1;
     real t1;
     real t2;
     real d1=0; // normalizing constant
     
     vector[2] this_hid; //   
     real HIP[F]; //Herd Infection distribution
     vector[2] thismix;
     
     for (n in 1:F) {
          t1 = 0; // Loglikelihood component if herd n is supposed to be infected
          t2 = 0; // Loglikelihood component if herd n is not supposed to be infected
          
          // for each animal in the herd
          for (k in I_s[n]:I_e[n]) {
               
               for (j in 1:J) {     // for each test
                    if (v[k,j] ==1) {
                         pi1   = sens[j]*CWHP[n] + (1-spec[j])*(1-CWHP[n]);// Apparent prevalence
                         t1 += bernoulli_lpmf( y[k,j] |  pi1);
                         t2 += bernoulli_lpmf( y[k,j] |  1-spec[j]);
                    }
               }
          }
          
          thismix=[log(HTP) + t1, log(1-HTP) + t2 ]';
          this_hid = softmax(thismix);
          InfP += this_hid[1]*CWHP[n];
          d1 += this_hid[1];
          HIP[n]=this_hid[1];
     }
     InfP = InfP/d1;
}
