Report_Sam (deadline 30 jun)

"Does Gradient Boosting improve with Fisher’s Discriminant analysis? Proposal of an integrated approach and empirical assessment of five popular datasets
 

Reviewing: 1
 
 
Major points:
 
1) The most significant methodological concern is the tuning strategy. The authors fix GBM hyperparameters and then apply the same parameters to PCA+GBM, LDA+GBM, and LdaBoost — justifying this as a way to isolate the effect of the feature transformation. However, this creates an inherently unequal playing field. The baseline GBM parameters were optimized for the original feature space, which may be suboptimal (or even detrimental) for the transformed spaces. The result is that poor performance of PCA+GBM or LDA+GBM could reflect suboptimal tuning rather than any intrinsic limitation of the method. Only LdaBoost_tuned receives full independent tuning, making cross-method comparisons difficult to interpret cleanly. The authors should either tune all methods independently, or at minimum discuss this limitation more prominently and consistently.
 
Ok now we tune completely all parameters, also learning_rate ( BEFORE at 0.05) and the
number of observations for each stochastic gradient descent iteration (subsample)
RIFAI TUNANDO SOLO LEARNING RATE (SE CI METTE TROPPO  FAI UNA GRIGLIA ATTORNO A 0.05), subsample LASCIALO COM’ERA 0.6
 
 
2) While the authors tune the number of PCA components, they retain the maximum number of discriminant components c for LDA (C−1). This is theoretically motivated, but it means PCA is tuned while LDA is not, adding another asymmetry in the comparison. An ablation study retaining fewer LDA components would strengthen the empirical analysis.
 
SI PUO FARE rispetto a TABLE 1 COSA VIENE AL VARIARE per GBM+LDA (al variare di numero di var discriminanti PER Yeast (10 classi  c=1,5, 9 già c’è nel paper) e HAR (6 classi, c=1, 3, 5 già c’è nel paper)?  
 
Lo mettiamo in nota come commento
 
 
3) The paper relies on cross-validated means and standard deviations to compare methods, but never performs any formal statistical tests (e.g., Wilcoxon signed-rank test, Friedman test, or corrected paired t-tests). Given the overlapping standard deviations visible in Table 1, many of the claimed improvements may not be statistically significant. This is a standard expectation in comparative machine learning studies.
CHE ROMPICAZZO CHE è PUOI FARE QUALCHE TEST ???
 
 
4) LdaBoost refits LDA at every boosting iteration, which is considerably more expensive than standard GBM. The paper contains no discussion whatsoever of computational overhead — training times, scalability to large datasets, or memory footprint. This is a major omission for a method intended for practical use.
 
 
 
5) In binary classification (C=2), LDA produces exactly one discriminant component. Training GBM trees on a single feature per iteration is a severe constraint, and it is somewhat surprising LdaBoost performs competitively at all. The authors should discuss this case more carefully, since the theoretical motivation for LdaBoost is much more compelling in the multiclass setting.
 
CI PENSO IO
 
 
6) The related work focuses on feature extraction literature but does not engage with the rich literature on boosting variants. Methods such as LPBoost, FilterBoost, or boosting with kernel projections bear at least some conceptual similarity to LdaBoost. A brief discussion of how LdaBoost relates to or differs from these would strengthen the paper's positioning.
 
CI PENSO IO
 
 
7) The simulation study is a welcome addition, but the data-generating process (multivariate normal with class-specific means) is actually the setting where LDA is asymptotically optimal by design. Results here are therefore somewhat expected and may not generalize to real data with non-Gaussian features, nonlinear class boundaries, or mixed variable types. The authors should acknowledge this limitation and ideally complement the simulations with non-Gaussian scenarios.
 
CI PENSO IO, GLI DICIAMO CHE COSI ABBIAMO IL BAYES ERROR E LO METTIAO NEI PAPER LIMITATIONS
 
 
8) The fact that baseline GBM outperforms LdaBoost on cross-validated accuracy for the largest and most feature-rich dataset (HAR, 561 features, 10,299 observations) is actually the most informative result in the paper, yet it receives relatively little discussion. This is the setting where one might most expect LDA preprocessing to help with dimensionality reduction — why does it not? A more thorough analysis of this "failure mode" would be valuable.
 
HAI QUALCHE IPOTESI?ASPETTIAMO, MAGARI CAMBIA SE RITUNIAMO COME RICHIESTO AL PUNTO 1
 
 
Minor points
 
•​Figure captions are inconsistent: some figures in the body and appendix share nearly identical descriptions, making it hard to distinguish them.
•​The abstract claims results "show that applying LDA... significantly improves classification accuracy" — the word "significantly" implies statistical testing, which is not performed.
CORREGGI E LEVA OVERALL significantly O significant
 
•​The paper would benefit from a short limitations section before or in the conclusion.
CI PENSO IO,

 

 

Reviewing: 2
 
Major Concerns
 
1. Page 7 of 21, lines 22 through 27: The text implies that LDA is recalculated at each boosting round. However, the Fisher criterion is based on maximizing between-class scatter relative to within-class scatter. If residuals replace labels, the theoretical justification
for applying LDA in this way is not obvious. The manuscript should explain how the optimization problem remains valid under residual-based updates.
 
PUNTO CHIAVE è QUESTO … relabel each sample by the class with the largest positive
gradient, and refit LDA  ??? PENSACI  …PUOI TRATTRE SPINTO DAL GBM  CON TARGET BINARIO CHE LAVOR SUI LOGODDS O SUI RESIDUI  ETC ?
 
 
2. In the LdaBoost description, the authors recomputed LDA at each boosting round, a procedure that can become computationally expensive, particularly in high-dimensional settings. Please add few sentences discussing the scalability of this iterative approach
and compare it with using LDA once as a pre-processing step.
 
 
3. Steps 2 and 3 in the step-by-step algorithmic implementation of the LdaBoost procedure look fine and follow standard boosting ideas. The main concern is Step 1. LDA normally works with fixed class labels, but here the method tries to relabel samples using residuals and then refit LDA (as mentioned in 1). Residuals are not the same as labels, so it is unclear if this is a valid use of LDA. The validity of this step is unclear without additional justification. Hence, without a clear explanation, this part of the method feels uncertain. The authors should explain more clearly how residuals can be used in LDA and why this makes sense.
 
STESSA COSA DEL PUNTO 1
 
4. Step 4 in the algorithmic implementation of the LdaBoost procedure itself looks fine. However, it relies on Step 1 because the scores it updates come from that LDA step. If Step 1 is not clearly explained or justified, then Step 4 might end up relying on something that is not solid or reliable.
 
STESSA COSA DEL PUNTO 1
 

Minor Concerns
• Page 5 of 21 lines 47-48: yi(c) is introduced but not used in the PCA formulation. Since PCA is unsupervised, labels are irrelevant here and could be omitted or clarified. If you want to keep labels in the broader setup (because later you compare PCA with LDA, which is supervised), then add a clarifying sentence.
 
???
 
• Pages 6 of 21 lines 28:29: The statement “predictors with larger dissimilarities between class means will have larger weights” is oversimplified. In LDA, feature weights depend on both how far apart class means are and how consistent the feature is within each class. Please revise this explanation to reflect both components of the Fisher criterion so that the description is accurate and balanced.
CORREGGI
 
• Minor typographical errors (e.g., “repored” on page. 9 of 21 line 3).
CORREGGI
 
• Page 12 of 21 line 14 through 19: Please consider revising the sentence. You can start the sentence with ”We first begin our simulations. . . ”Also, please note that sample of 1,000 is not typically considered “small,” especially in simulation studies.
 
CORREGGI
 
 
This manuscript has potential to contribute to the literature on research in integrated learning models that integrate classical statistical discriminant methods with modern ensemble algorithms to improve classification performance. Fixing the methods section and a few smaller issues as mentioned above will significantly improve its rigor and readability. I recommend revision at this stage.