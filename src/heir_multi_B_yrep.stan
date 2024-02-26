//
// This Stan program generates replicate data



// The input data assumes global pooling, with some number of positive results y from sample of size n.
data {
  // K=2 classes, negative (0) and positive (1)
  int<lower=1> F; //number of farms
  int<lower=1> I; //total number of animals tested
  int<lower=1> I_s[F]; //starting index for animals from from f
  int<lower=1> I_e[F]; //ending index for animals from from f
  
  int<lower=1> J; // number of tests - 3 for our work
  
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



generated quantities {
     
     real pi0;
     real pi1;
     int<lower=0,upper=1> y_rep[I,J];
  
  for (n in 1:F) {
    pi0 = bernoulli_rng(HTP);
   
    // for each animal in the herd
    for (k in I_s[n]:I_e[n]) {
         
         for (j in 1:J) {     // for each test
              if (pi0==0)
                 pi1=1-spec[j];
               else
                 pi1=sens[j]*CWHP[n] + (1-spec[j])*(1-CWHP[n]);
          
         y_rep[k,j]=bernoulli_rng(pi1);

         }
    }


  }

}
