# Computing the fisher vectors
- Run the gm.py file that will compute the features from the images and generate a GMM fitted to the feature distribution
- Run the fisher_kernels.py: this will compute the features for training and test dataset and use the GMM to compute the associated fisher vectors
- Follow the training procedure with Linear kernel with the use_fisher_kernel parameter of the load_data function
# Training
For now the only model available is a MulticlassSVC using one versus rest (ovr) or one versus one (ovo) strategy.
- Tuning the parameters: 
To train the model, change the required kernel ("kernel_name", can be a composition of several kernels from the kernles.py file, separated by a "+"), you can add weights if you sum kernels, and hyper parameters (kernel_kwargs). Don't forget to change the save path

- Cache: to avoid recomputing the same kernels, files are created in the "kernel" folder. This is especially usefull for the histogram kernel that takes 30s to compute.

- Saving: The classifier can be saved. Basically it saves the computed coefficients alpha for each svc in the model in a npz file

# Evaluating
The evaluation is done in the evaluate.py file.
- The instanciated model should be the same as the one trained (with same kernels and prediction type ovo or ovr).

- Evaluation can take long since we have to compute the gram matrix for each svc with all the training data. One improvement in speed can be done by only keeping the samples associated to the most significant alphas at evaluation.

- Don't forget to change the loading path of the model and the output path

# Side notes
- I used cvxopt to solve the minimization problem since it is a quadratic programming problem and scipy optimize is shit (it sped up the computation time by two orders of magitude)

- I also used the python library 'fast-histogram' to compute the histogram kernel which is twice as faster as np.histogramdd

- In my experiments it now takes around 40min to train a new svc from scratch with the hist kernel and between 30s and 20min if gram matrices are already computed in the kernel folder for the ovo strategy. ovr is just to loong