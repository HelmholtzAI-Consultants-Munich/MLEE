# MLEE
This repo contains a Machine Learning Framework for Environmental Epidemiology (MLEE)

## 1. Overview
MLEE is a publicly available, generalizable, and user-friendly machine learning (ML) framework designed to support the analysis of environmental and health data. The framework was developed to address the current lack of reproducible ML workflows in environmental epidemiology, where applications of ML remain limited despite the growing availability of high-resolution environmental exposure data and large population cohorts.
The MLEE framework integrates data preprocessing, multiple machine learning classifiers, performance evaluation, and model explainability methods. MLEE enables researchers to identify and rank key individual, environmental, and neighborhood-level determinants of binary health outcomes while maintaining reproducibility throughout the process.
By combining predictive modeling with interpretable ML approaches, MLEE helps researchers explore complex, high-dimensional datasets and uncover important drivers of health outcomes. The framework is designed to complement traditional epidemiological methods and facilitate the use of ML in environmental epidemiology.


## 2. Quick start <!-- to my knowledge, this quick start is only for local machine set up. perhaps we should also create a quick start for cluster -->

If you already have **Git** and **Conda** installed, within your terminal, navigate to your local directory where you want to clone the MLEE repository and run:

##### 2.1 Windows and macOS
<!-- I think we should remove the step python.main.py--we don't need them to run the code yet. I would reccomend directing them to step 4. i.e., "Once this step is complete, you can proceed to **step 4**. -->
```bash
git clone https://github.com/HelmholtzAI-Consultants-Munich/MLEE.git
cd MLEE

conda env create --file environment.yaml
conda activate MLEE

python main.py
```

##### 2.2 Linux and cluster <!-- to my knowledge, this may be the quick start for cluster -->
 ```bash
git clone https://github.com/HelmholtzAI-Consultants-Munich/MLEE.git
cd MLEE

conda env create --file environment_cluster.yaml
conda activate MLEE
```

## 3. Detailed installation instruction

If this is your first time setting up the project or you need to configure SSH, Conda, or the data archive, continue with the detailed installation instructions below.


### 3.1 Prerequisites

Before setting up the project, ensure you have the following installed:

- Code editor. If you do not already have a preferred code editor or IDE, we recommend using **Visual Studio Code (VS Code)** which can be downloaded here: https://code.visualstudio.com/
- **Git** for cloning the repository, which can be downloaded here: https://git-scm.com/install/
  
**Conda** (Miniconda recommended, Anaconda also supported) is also required. We recommend using **Conda 23.10 or newer**, as newer versions include the `libmamba` dependency solver, which can significantly reduce environment creation time. If you do not have Conda installed, follow the installation instructions in the next section **(step 3.2.2)**.


### 3.2 Installation  <!-- I think it is slightly confusing to talk about Conda installation, clone the MLEE repository, and then provide instructions to clone Conda after. can we move this Installation step after Conda installation? -->

#### 3.2.1 Clone the MLEE repository

You only need to clone the repository once per local device. If you intend to run MLEE on the cluster, the file system is shared across login and compute nodes, so the repository also only needs to be cloned once.

##### Recommended: clone using HTTPS
Navigate to the local directory where you want to store the project and run:

```bash 
git clone https://github.com/HelmholtzAI-Consultants-Munich/MLEE.git
cd MLEE
```

##### Optional: clone using SSH

If you prefer to use SSH, first configure an SSH key for GitHub by following the official GitHub instructions:

https://docs.github.com/en/authentication/connecting-to-github-with-ssh

After adding your SSH key to your GitHub account, clone the repository with:

```bash
git clone git@github.com:HelmholtzAI-Consultants-Munich/MLEE.git
cd MLEE
```

On the cluster, if direct SSH access to GitHub is restricted, you can configure GitHub SSH to use port 443:

```text
Host github.com
    HostName ssh.github.com
    IdentityFile ~/.ssh/id_ed25519_github
    IdentitiesOnly yes
    Port 443
```

#### 3.2.2 Install Conda

If you already have **Conda** installed (Miniconda or Anaconda, version 23.10 or newer is recommended), you can proceed directly to **step 3.3**. Otherwise, follow **step 3.2.2.1** for Windows and macOS installation or **3.2.2.2** for Linux and cluster installation.

##### 3.2.2.1 Windows and macOS

Install the latest version of **Miniconda** (recommended) or **Anaconda** for your operating system and processor architecture.

- Miniconda: https://www.anaconda.com/download/success
- Anaconda: https://www.anaconda.com/download

##### 3.2.2.2 Linux and cluster

1. Create a `tools` directory in your home folder (if it does not already exist):
	```bash
	mkdir ~/tools
	cd ~/tools
	```

2. Create a temporary directory for the installation:
	```bash
	mkdir tmpconda
	cd ~/tools/tmpconda
	```

3. Download Miniconda:
	```bash
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	```

4. Install Miniconda:
	```bash
	TMPDIR=~/tools/tmpconda bash Miniconda3-latest-Linux-x86_64.sh -u
	```
	Accept the license agreement and complete the installation.

5. Configure your environment:
	```bash
	chmod +x ~/miniconda3/bin/*
	chmod +x ~/miniconda3/envs
	export LD_LIBRARY_PATH=~/miniconda3/lib:$LD_LIBRARY_PATH
	export PATH=~/miniconda3/bin:$PATH
	source ~/miniconda3/etc/profile.d/conda.sh
	~/miniconda3/bin/conda init bash
	```
	Log out and log in again after the installation.


### 3.3 Environment setup

#### 3.3.1 Verify the Conda installation
  Open your code editor (VS Code or similar) and check your Conda version by typing in the terminal (Command Prompt (CMD) terminal recommended):
	
```bash
conda --version
```

> **Troubleshooting:** If conda --version does not return a version number, open your Anaconda Prompt app and run:
>  	```bash
>	conda init cmd.exe
> 	```
**** 

Afterward, restart VS Code and open a new terminal session (Command Prompt (CMD) terminal recommended). 
	
   Conda **23.10 or newer** is recommended. Older versions may use the slower `classic` dependency solver.
	
   Check the configured solver:
   
```bash
conda config --show solver
```
	
The recommended output is:
	
```text
solver: libmamba
```
	
OPTIONAL: If the solver is `classic`, switch to `libmamba`:

```bash
conda config --set solver libmamba
```
	
Verify the change:

```bash
conda config --show solver
```
	
OPTIONAL: If `libmamba` is unavailable, update Conda before continuing:

```bash
conda update -n base conda
```

#### 3.3.2 Create the environment
Remain in the terminal of your code editor. Make sure you are in the root directory of the cloned `MLEE` repository. Follow instructions for either the local machine **3.3.2.1** or cluster **3.3.2.2**.

##### 3.3.2.1 Local machine

```bash
conda env create --file environment.yaml
```

##### 3.3.2.2 Cluster

```bash
conda env create --file environment_cluster.yaml
```

## 4. Usage
This section contains the instructions to run the framework.

#### 4.1 Activate the environment

```bash
conda activate MLEE
```

If you chose a different environment name, replace `MLEE` with the name you specified.

#### 4.2 OPTIONAL: Update the environment

If the environment definition has changed, update your existing environment.

##### 4.2.1 Local machine <!-- is this equivalent to Mac and OS? -->

```bash
conda env update --file environment.yaml --prune
```

##### 4.2.2 Cluster** <!-- is this equivalent to Linux and Cluster? -->

```bash
conda env update --file environment_cluster.yaml --prune
```

#### 4.3 OPTIONAL: Recreate the environment

If updating does not resolve dependency conflicts, remove and recreate the environment.

Delete the environment:

```bash
conda remove --name MLEE --all
```

Then recreate it by following the **Create the environment** section above **(3.2.2)**).

> **Note:** Creating the environment may take several minutes because Conda must resolve and download package dependencies. If it remains on **"Solving environment"** for an unusually long time, verify that you are using Conda 23.10 or newer with the `libmamba` solver enabled.

#### 4.4 Preparing Input Parameters
Open the **input_parameters.json** file in your preferred code editor. Using the outline of inputs below as a guide, update the file to reflect your dataset and variables.

- ```path_name```: Path to the directory containing the dataset, default: "./data/". <br />
- ```file_name```: Name of the dataset file <br />
- ```columns_to_keep```: Dictionary containing the list of numerical ("num") and categorical ("cat") columns to retain.  <br />
- ```binary_columns```: List with categorical features that are binary and do not need to be one hot encoded. <br />
- ```target```: Name of the target variable, default: "hypertension". <br />
- ```filters```: Dictionary specifying filters for age, bmi, and sex.
	- Keys correspond to the feature names.
	- If not specified, no filtering is applied.
	- For age and bmi, null is interpreted as 0 (lower bound) or ∞ (upper bound).
	- Options for sex: "M" or "m" (men), "W" or "w" (women).
- ```test_size```: Float number representing the proportion of the dataset used as the test set. <br />
- ```validation_size```: Float number representing proportion of the dataset used as the validation set. <br />
- ```feature_stratification```: List of features used for stratified train-test splitting. <br />
- ```features_to_drop```: List of features to drop after stratification (i.e., used for splitting but not for training). <br />
- ```imputation_strategy```: String representing the imputation technique to apply, options: "iterative", "mean". <br />
- ```models```: Dictionary containing the models to evaluate and their corresponding grid of hyperparameters. <br />
- ```n_boot_iterations```: Integer number of bootstrap iterations performed during model evaluation. <br />
- ```selection_metric```: String containing the metric used for model selection, options: "accuracy", "precision", "recall", "f1-score". <br />
- ```selection_cutoff```: Float number representing the P-value threshold above which the best-ranked model is automatically selected. <br />
- ```shap_output_prob```: Boolean indicating the output space of the SHAP values. If `true`, SHAP values are computed in probability space; if `false`, they are computed in log-odds space.
- ```subset_percentage```: Float number representing the proportion of the dataset to use, (default: 1.0, i.e. the full dataset. <br />

#### 4.5 Run the framework
#### 4.5.1 Local machine <!-- is this equivalent to Mac and OS? -->
After the MLEE environment is activated and you have updated the input parameters, you can just type the following command to run the framework:
```
python main.py
```
This will execute the whole pipeline with the input parameters you can set in the input_parameters.json (see 4.1).

#### 4.5.2 Cluster <!-- is this equivalent to Linux and Cluster? -->
1. First you have to log in to the cluster:
	```
	ssh username@hpc-build01
	```
	or 
	```
	ssh user.name@@hpc-submit03gui
	```
2. The cloning procedure on the cluster is the same as on any device and it is done once.
3. For the first time you need to transfer csv file to the folder named "data" and also encrypt it as mentioned above. Afterwards, each time you log in you just need to extract the data.7z and you delete the "data" folder once you don't need it to keep it secure, as mentioned in the previous sections.
4. There are two ways to run the code on the cluster, one is to submit a job via a slurm script and the second is to request an interactive session/job and run it on the assigned compute node.\
	a. submit a slurm script:
	you can run the script called run_pipeline.sbatch after navigating to MLEE folder by entering:
	```
 	sbatch run_pipeline.sbatch
 	```
 	this will request resources from slurm job scheduler and you will see the job allocation number. You can always check the running jobs by
	```
 	squeue -u user.name
 	```
 	replacing user.name with your username. Standard output and error logs are stored in the slurm_log directory.

	b. request a compute resource on some compute node:
	by entering this command in the cluster terminal you will get same amount of resource as the sbatch script but also you get an interactive session that helps with 	editing and running the code:
	```
	salloc -J N2N_pipeline -c 16 -p cpu_p --qos=cpu_normal --mem=128G -t 1-00:00:00
	```
	once you got to the compute node you then then activate the conda environment
	```
	conda activate MLEE
	```
	then run the code the same as local computers with:
	```
	python main.py
	```

### 4.6 Outputs <!-- generally speaking, I think it would be helpful to explain a bit more where to find each of the outputs. It doesn't have to have a lot of detail, but a bit more would be helpful -->
The result of each run will be saved as HTML report within the reports folder. Each experiment is saved in a different folder, where the name contains the timestamp of the run and a unique identifier. Each report will  be also saved as an archived zip file for ease of transfer. Also, the outputs are saved in the output folder as well as the resources in the reports folders under the experiment's folder. 

## 5. Contributing

Contributions are welcome! If you would like to improve the project, please follow these guidelines:

1. Create a new branch for your changes.
2. Keep commits focused and use descriptive commit messages.
3. Update the documentation if your changes affect the installation, usage, or outputs.
4. Ensure the pipeline runs successfully before submitting your changes.

When your work is ready, open a pull request and provide a clear description of:
- the motivation for the changes,
- the main modifications,
- any additional steps required to test them.

If you are unsure about a proposed change, feel free to open an issue or start a discussion before implementing it.

