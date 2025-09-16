library(tibble)
library(dplyr)
library(ggplot2)
library(ggpubr)


# UNIVARIATE RESULTS -------------------------------------------------------------------------

# inital
df_initial <- tribble(
  ~Biomarker,        ~AUC,   ~CI_lower, ~CI_upper,
  "CBV_corr_ratio",   0.930,  0.89,      0.97,
  "CBV_noncor_ratio", 0.786,  0.72,      0.86,
  "COV_ratio",        0.993,  0.98,      1.00,
  "CTH_MAX_ratio",    1.000,  1.00,      1.00,
  "CTH_ratio",        1.000,  1.00,      1.00,
  "DELAY_ratio",      0.994,  0.98,      1.00,
  "OEF_ratio",        0.991,  0.98,      1.00,
  "rCMRO2_ratio",     1.000,  1.00,      1.00,
  "rLEAKAGE_ratio",   0.988,  0.96,      1.00
) %>%
  mutate(Type = ifelse(grepl("CBV", Biomarker), "reference", "novel"))


# validation
df_validate <- tribble(
  ~Biomarker,       ~AUC,   ~CI_lower, ~CI_upper,
  "CBV_corr_ratio", 0.714,  0.48,      0.95,
  "CBV_noncor_ratio",0.768, 0.54,      0.99,
  "COV_ratio",      0.598,  0.30,      0.90,
  "CTH_MAX_ratio",  0.652,  0.36,      0.95,
  "CTH_ratio",      0.741,  0.46,      1.00,
  "DELAY_ratio",    0.545,  0.24,      0.85,
  "OEF_ratio",      0.643,  0.33,      0.96,
  "rCMRO2_ratio",   0.589,  0.33,      0.85,
  "rLEAKAGE_ratio", 0.714,  0.48,      0.95
) %>%
  mutate(Type = ifelse(grepl("CBV", Biomarker), "reference", "novel"))


# BIVARIATE RESULTS -------------------------------------------------------------------------

df_bivariate <- tribble(
  ~Biomarker,        ~AUC,   ~CI_lower, ~CI_upper,
  "CBV_corr_path+control",   0.714,  0.474, 0.954,
  "CBV_noncor_path+control", 0.821,  0.599, 1.000,
  "CTH_path+control",        0.786,  0.500, 1.000,
  "CTH_MAX_path+control",    0.714,  0.420, 1.000,
  "OEF_path+control",        0.705,  0.395, 1.000,
  "Delay_path+control",      0.580,  0.258, 0.903,
  "rLEAKAGE_path+control",   0.679,  0.427, 0.930,
  "rCMRO2_path+control",     0.527,  0.286, 0.767,
  "COV_path+control",        0.607,  0.309, 0.906
) %>%
  mutate(Type = ifelse(grepl("CBV", Biomarker), "reference", "novel")
         )


# comparisons with univariate
# where claimed signifcantly different due to non-overlapping CIs
df_bivar_lower <- tribble(
  ~Biomarker,              ~AUC,   ~CI_lower, ~CI_upper, ~ModelType,
  # Combined (claimed more predictive)
  "Delay_path+control",    0.580,  0.258,     0.903,     "Bivariate",
  "rLEAKAGE_path+control", 0.679,  0.427,     0.930,     "Bivariate",
  "rCMRO2_path+control",   0.527,  0.286,     0.767,     "Bivariate",
  "COV_path+control",      0.607,  0.309,     0.906,     "Bivariate",
  
  # Univariate
  "Delay_ratio",           0.994,  0.98,      1.00,      "Univariate",
  "rLEAKAGE_ratio",        0.988,  0.96,      1.00,      "Univariate",
  "rCMRO2_ratio",          1.000,  1.00,      1.00,      "Univariate",
  "COV_ratio",             0.993,  0.98,      1.00,      "Univariate",
)



# where claimed not signifcantly different due to overlapping CIs
df_bivar_comparable <- tribble(
  ~Biomarker,                ~AUC,   ~CI_lower, ~CI_upper, ~ModelType,
  # Combined (claimed not more predictive)
  "CBV_corr_path+control",   0.714,  0.474,     0.954,     "Bivariate",
  "CBV_noncor_path+control", 0.821,  0.599,     1.000,     "Bivariate",
  "CTH_path+control",        0.786,  0.500,     1.000,     "Bivariate",
  "CTH_MAX_path+control",    0.714,  0.420,     1.000,     "Bivariate",
  "OEF_path+control",        0.705,  0.395,     1.000,     "Bivariate",
  
  # Univariate
  "CBV_corr_ratio",          0.930,  0.89,      0.97,      "Univariate",
  "CBV_noncor_ratio",        0.786,  0.72,      0.86,      "Univariate",
  "CTH_ratio",               1.000,  1.00,      1.00,      "Univariate",
  "CTH_MAX_ratio",           1.000,  1.00,      1.00,      "Univariate",
  "OEF_ratio",               0.991,  0.98,      1.00,      "Univariate",
)


df_bivar_combine = rbind(
  df_bivar_lower %>% mutate(claim = "different"),
  df_bivar_comparable %>% mutate(claim = "comparable")
)


df_svm <- tribble(
  ~Biomarker,        ~AUC,   ~CI_lower, ~CI_upper,
  "CBV_corr_path+control",   0.714,  0.476, 0.953,
  "CBV_noncor_path+control", 0.812,  0.589, 1.000,
  "CTH_path+control",        0.786,  0.500, 1.000,
  "CTH_MAX_path+control",    0.714,  0.420, 1.000,
  "OEF_path+control",        0.696,  0.378, 1.000,
  "Delay_path+control",      0.562,  0.246, 0.879,
  "rLEAKAGE_path+control",   0.679,  0.427, 0.930,
  "rCMRO2_path+control",     0.536,  0.294, 0.778,
  "COV_path+control",        0.598,  0.296, 0.900
) %>%
  mutate(Type = ifelse(grepl("CBV", Biomarker), "reference", "novel"))



df_svm_lower <- tribble(
  ~Biomarker,                ~AUC,   ~CI_lower, ~CI_upper, ~ModelType,
  # SVM (linear) (claimed more predictive)
  "Delay_path+control",      0.562,  0.246,     0.879,     "SVM (linear)",
  "rLEAKAGE_path+control",   0.679,  0.427,     0.930,     "SVM (linear)",
  "rCMRO2_path+control",     0.536,  0.294,     0.778,     "SVM (linear)",
  "COV_path+control",        0.598,  0.296,     0.900,     "SVM (linear)",
  
  # Univariate
  "Delay_ratio",           0.994,  0.98,      1.00,      "Univariate",
  "rLEAKAGE_ratio",        0.988,  0.96,      1.00,      "Univariate",
  "rCMRO2_ratio",          1.000,  1.00,      1.00,      "Univariate",
  "COV_ratio",             0.993,  0.98,      1.00,      "Univariate",
  )
  

df_svm_comparable <- tribble(
  ~Biomarker,                ~AUC,   ~CI_lower, ~CI_upper, ~ModelType,
  # SVM (linear)
  "CBV_corr_path+control",   0.714,  0.476,     0.953,     "SVM (linear)",
  "CBV_noncor_path+control", 0.812,  0.589,     1.000,     "SVM (linear)",
  "CTH_path+control",        0.786,  0.500,     1.000,     "SVM (linear)",
  "CTH_MAX_path+control",    0.714,  0.420,     1.000,     "SVM (linear)",
  "OEF_path+control",        0.696,  0.378,     1.000,     "SVM (linear)",
  
  # Univariate
  "CBV_corr_ratio",          0.930,  0.89,      0.97,      "Univariate",
  "CBV_noncor_ratio",        0.786,  0.72,      0.86,      "Univariate",
  "CTH_ratio",               1.000,  1.00,      1.00,      "Univariate",
  "CTH_MAX_ratio",           1.000,  1.00,      1.00,      "Univariate",
  "OEF_ratio",               0.991,  0.98,      1.00,      "Univariate",
)


df_svm_combine = rbind(
  df_svm_lower %>% mutate(claim = "different"),
  df_svm_comparable %>% mutate(claim = "comparable")
)


df_rbf <- tribble(
  ~Biomarker,        ~AUC,   ~CI_lower, ~CI_upper,
  "CBV_corr_path+control",   0.714,  0.474, 0.954,
  "CBV_noncor_path+control", 0.652,  0.386, 0.918,
  "OEF_path+control",        0.688,  0.371, 1.000,
  "Delay_path+control",      0.598,  0.270, 0.926,
  "CTH_path+control",        0.598,  0.326, 0.871,
  "CTH_MAX_path+control",    0.321,  0.094, 0.549,
  "rLEAKAGE_path+control",   0.607,  0.329, 0.885,
  "rCMRO2_path+control",     0.536,  0.294, 0.778,
  "COV_path+control",        0.598,  0.297, 0.900
) %>%
  mutate(Type = ifelse(grepl("CBV", Biomarker), "reference", "novel"))


df_rbf_lower <- tribble(
  ~Biomarker,                ~AUC,   ~CI_lower, ~CI_upper, ~ModelType,
  # SVM (rbf) (claimed more predictive)
  "COV_path+control",        0.598,  0.297,     0.900,     "SVM (RBF)",
  "CTH_MAX_path+control",    0.321,  0.094,     0.549,     "SVM (RBF)",
  "CTH_path+control",        0.598,  0.326,     0.871,     "SVM (RBF)",
  "Delay_path+control",      0.598,  0.270,     0.926,     "SVM (RBF)",
  "rCMRO2_path+control",     0.536,  0.294,     0.778,     "SVM (RBF)",
  "rLEAKAGE_path+control",   0.607,  0.329,     0.885,     "SVM (RBF)",
  
  # Univariate
  "COV_ratio",               0.993,  0.98,      1.00,      "Univariate",
  "CTH_MAX_ratio",           1.000,  1.00,      1.00,      "Univariate",
  "CTH_ratio",               1.000,  1.00,      1.00,      "Univariate",
  "DELAY_ratio",             0.994,  0.98,      1.00,      "Univariate",
  "rCMRO2_ratio",            1.000,  1.00,      1.00,      "Univariate",
  "rLEAKAGE_ratio",          0.988,  0.96,      1.00,      "Univariate",
)


df_rbf_comparable <- tribble(
  ~Biomarker,                ~AUC,   ~CI_lower, ~CI_upper, ~ModelType,
  # SVM (RBF bivariate, from text)
  "CBV_corr_path+control",   0.714,  0.474,     0.954,     "SVM (RBF)",
  "CBV_noncor_path+control", 0.652,  0.386,     0.918,     "SVM (RBF)",
  "OEF_path+control",        0.688,  0.371,     1.000,     "SVM (RBF)",
  
  # Univariate
  "CBV_corr_ratio",          0.930,  0.89,      0.97,      "Univariate",
  "CBV_noncor_ratio",        0.786,  0.72,      0.86,      "Univariate",
  "COV_ratio",               0.993,  0.98,      1.00,      "Univariate",
  "CTH_MAX_ratio",           1.000,  1.00,      1.00,      "Univariate",
  "CTH_ratio",               1.000,  1.00,      1.00,      "Univariate",
  "DELAY_ratio",             0.994,  0.98,      1.00,      "Univariate",
  "OEF_ratio",               0.991,  0.98,      1.00,      "Univariate",
  "rCMRO2_ratio",            1.000,  1.00,      1.00,      "Univariate",
  "rLEAKAGE_ratio",          0.988,  0.96,      1.00,      "Univariate",
)


df_rbf_combine = rbind(
  df_rbf_lower %>% mutate(claim = "different"),
  df_rbf_comparable %>% mutate(claim = "comparable")
)




# PLOTS -------------------------------------------------------------------

(p1 <- ggplot(df_initial, aes(x = Biomarker, y = AUC, color = Type)) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper), width = 0.2) +
    coord_flip() +
    theme_minimal() +
    labs(title = "Initial Test Cohort AUCs",
         subtitle = "(TR = 71, RN = 111)",
         y = "AUC", x = "Biomarker",
         color = "Biomarker Type") +
    scale_y_continuous(limits = c(0, 1)) +
    theme(
      plot.title = element_text(hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5)
    ) +
    scale_color_manual(values = c("steelblue", "darkgrey"))
)


# swap base data in plot
(p2 <- p1 %+% df_validate + labs(title = "Validation Cohort AUCs",
                                 subtitle = "(TR = 7, RN = 16)"))


# SUGGESTED FIGURE
ggarrange(p1, p2, ncol = 2, nrow = 1,
          common.legend = TRUE, legend = "top")


(p3 <- p1 %+% df_bivariate + labs(title = "Initial Test Cohort AUCs (Bivariate)"))


# SUGGESTED FIGURE
(p4 <- p1 %+% 
    df_bivar_combine + 
    aes(color = ModelType) + 
    labs(title = "Univariate v Bivariate",
         color = "Model") +
    scale_color_manual(values = c("Univariate" = "steelblue", "Bivariate" = "darkgrey")) +
    facet_wrap(~claim, labeller = labeller(claim = c("comparable" = "", "different" = "")))
)


# SUGGESTED SUPPLEMENTARY FIGURE (OR MAYBE NOT, AS ITS ALMOST EXACTLY THE SAME AS ABOVE)
(p5 <- p4 %+% df_svm_combine + labs(title = "Univariate v SVM (linear)") +
    scale_color_manual(values = c("Univariate" = "steelblue", "SVM (linear)" = "darkgrey"))
)


# SUGGESTED SUPPLEMENTARY FIGURE
(p6 <- p4 %+% df_rbf_combine + labs(title = "Univariate v SVM (RBF)") +
    scale_color_manual(values = c("Univariate" = "steelblue", "SVM (RBF)" = "darkgrey"))
)

(p7 <- p1 %+% df_svm + labs(title = "Initial Test Cohort AUCs (SVM)"))
(p8 <- p1 %+% df_rbf + labs(title = "Initial Test Cohort AUCs (RBF)"))


# all bivariate
ggarrange(p3, p7, p8, ncol = 1, nrow = 3,
          common.legend = TRUE, legend = "top")


# note the SVM gives basically the same results as bivar
tmp = merge(df_bivariate %>% select(1:4),
            df_svm %>% rename(AUC_SVM = AUC,
                              CI_lower_SVM = CI_lower,
                              CI_upper_SVM = CI_upper) %>% select(1:4)
)
cor(tmp$AUC, tmp$AUC_SVM)
plot(tmp$AUC, tmp$AUC_SVM,
     xlim = range(c(tmp$AUC, tmp$CI_lower, tmp$CI_upper), na.rm=TRUE),
     ylim = range(c(tmp$AUC_SVM, tmp$CI_lower_SVM, tmp$CI_upper_SVM), na.rm=TRUE),
     xlab = "Bivariate (AUC / CI)", ylab = "SVM (AUC / CI)",
     pch = 16, col = "black")

# Add CI bounds
points(tmp$CI_upper, tmp$CI_upper_SVM, col = "red", pch = 4)
points(tmp$CI_lower, tmp$CI_lower_SVM, col = "blue", pch = 4)
abline(0, 1, lty = 2, col = "grey")



# BIOMARKER DISTRIBUTIONS -------------------------------------------------------------------------
set.seed(42)

df_cth <- tibble(
  Group = c(rep("TR", 71), rep("RN", 111)),
  CTH_ratio = c(
    rnorm(71, mean = 0.55, sd = 0.02),  # TR cluster
    rnorm(111, mean = 0.60, sd = 0.02)  # RN cluster
  )
)

# Ensure non-overlap by truncation (just to be safe)
df_cth <- df_cth %>%
  mutate(
    CTH_ratio = ifelse(Group == "TR" & CTH_ratio >= 0.58, 0.579, CTH_ratio),
    CTH_ratio = ifelse(Group == "RN" & CTH_ratio <= 0.58, 0.581, CTH_ratio)
  )


(p_cth <- ggplot(df_cth, aes(x = Group, y = CTH_ratio, color = Group)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.3) +
    geom_jitter(width = 0.2, size = 2) +
    theme_minimal() +
    labs(
         y = "CTH ratio", x = "Group") +
    theme(
      plot.title = element_text(hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5)
    ) +
    scale_color_manual(values = c("TR" = "steelblue", "RN" = "firebrick"))
)


# SUGGESTED SUPPLEMENTARY FIGURE EXAMPLE
ggarrange(p_cth, p_cth, p_cth, p_cth, p_cth, p_cth, p_cth, p_cth, p_cth, ncol = 3, nrow = 3,
          common.legend = TRUE, legend = "top")

