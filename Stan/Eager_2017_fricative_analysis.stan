// Eager, Christopher D. (2017). Contrast preservation and constraints on
//   individual phonetic variation. Doctoral thesis. University of Illinois
//   at Urbana-Champaign.
//
// Fricative analysis Stan code (Section A.3)

functions {
  matrix vec_to_mat_by_row(int R, int C, vector v) {
    matrix[R, C] m;
    for(r in 1:R) m[r] = v[(C * (r - 1) + 1):(C * r)]';
    return m;
  }
}

data {
  int<lower=0> N;  // number of observations
  int<lower=0> K;  // number of coefficients

  int<lower=0> nz;  // num non-zero elements in model matrix
  vector[nz] w;  // non-zero elements in model matrix
  int<lower=0> v[nz];  // column indices for w
  int<lower=0> u[N + 1];  // row-start indices for non-zero elements

  vector[N] y;  // scaled response

  int<lower=0> P;  // number of fixed effects
  int<lower=0> G;  // number of random effect groups
  int<lower=0> cindx[G, 2];  // coefficient index for random effects
  int<lower=0> M_1;  // number of speaker members
  int<lower=0> Q_1;  // number of speaker effects per member

  // (hyper) priors
  real<lower=0> scale_beta;  // prior scale for betas
  real<lower=0> nu_beta;  // degrees of freedom for beta t-dist prior
  real<lower=0> sc_q0;  // prior scale for random intercept sds
  real<lower=0> sc_qs;  // prior scale for random slope sds
  real<lower=0> eta_q;  // shape for LKJ prior on random effects correlations
  real<lower=0> sc_res;  // prior scale for sd of the residuals
}

parameters {
  // all parameters sampled on unit scale or with cholesky factors
  // (as applicable) and reparameterized

  vector[P] beta_raw;

  matrix[Q_1, M_1] gamma_1_raw;
  vector<lower=0>[Q_1] sigma_1_raw;
  cholesky_factor_corr[Q_1] omega_1_raw;

  real<lower=0> sigma_res_raw;
}

transformed parameters {
  vector<lower=0>[Q_1] sigma_1;  // sd in the speaker effects
  real<lower=0> sigma_res;  // sd of the residuals

  vector[K] coef;  // all coefficients
  vector[N] y_hat;  // fitted values

  coef[1:P] = scale_beta * beta_raw;

  sigma_1[1] = sc_q0 * sigma_1_raw[1];
  sigma_1[2:Q_1] = sc_qs * sigma_1_raw[2:Q_1];
  coef[cindx[1, 1]:cindx[1, 2]]
    = to_vector(rep_matrix(sigma_1, M_1)
      .* (omega_1_raw * gamma_1_raw));

  sigma_res = sc_res * sigma_res_raw;

  y_hat = csr_matrix_times_vector(N, K, w, v, u, coef);
}

model {
  beta_raw ~ student_t(nu_beta, 0, 1);

  to_vector(gamma_1_raw) ~ normal(0, 1);
  sigma_1_raw ~ normal(0, 1);
  omega_1_raw ~ lkj_corr_cholesky(eta_q);

  sigma_res_raw ~ normal(0, 1);
  y ~ normal(y_hat, sigma_res);
}

generated quantities {
  vector[N] log_lik;  // log-likelihod
  vector[P] beta;  // fixed effects
  matrix[M_1, Q_1] gamma_1;  // speaker effects
  matrix[Q_1, Q_1] omega_1;  // correlation in the speaker effects

  for(n in 1:N) log_lik[n] = normal_lpdf(y[n] | y_hat[n], sigma_res);
  beta = coef[1:P];
  gamma_1 = vec_to_mat_by_row(M_1, Q_1, coef[cindx[1, 1]:cindx[1, 2]]);
  omega_1 = tcrossprod(omega_1_raw);
}
