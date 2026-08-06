# MLEE
This repo contains a Machine Learning Framework for Environmental Epidemiology (MLEE)

## Overview
MLEE is a publicly available, generalizable, and user-friendly machine learning (ML) framework designed to support the analysis of environmental and health data. The framework was developed to address the current lack of reproducible ML workflows in environmental epidemiology, where applications of ML remain relatively limited despite the growing availability of high-resolution environmental exposure data and large population cohorts.
MLEE framework integrates data preprocessing, multiple machine learning classifiers, performance evaluation, and model explainability methods. MLEE enables researchers to identify and rank key individual, environmental, and neighborhood-level determinants of binary health outcomes while maintaining reproducibility throughout the process.
By combining predictive modeling with interpretable AI approaches, MLEE helps researchers explore complex, high-dimensional datasets and uncover important drivers of health outcomes. The framework is designed to complement traditional epidemiological methods and facilitate the use of ML in environmental epidemiology.

## Quick start

If you already have **Git** and **Conda** installed, navigate to the directory where you want to clone the repository and run:

```bash
git clone git@github.com:HelmholtzAI-Consultants-Munich/MLEE.git
cd MLEE

conda env create --file environment.yaml
conda activate MLEE

python main.py
```

If this is your first time setting up the project or you need to configure SSH, Conda, or the data archive, continue with the detailed installation instructions below.

## Prerequisites

Before setting up the project, ensure you have the following installed:

- **Git** for cloning the repository.
- **Conda** (Miniconda recommended, Anaconda also supported).

We recommend using **Conda 23.10 or newer**, as newer versions include the `libmamba` dependency solver, which can significantly reduce environment creation time.

If you do not have Git or Conda installed, follow the installation instructions in the next section.

## Installation

### Clone the repository (Only required once)
If you want to clone our repo to any device, follow a similar procedure but only once per device. Notice that on the cluster the file system is shared with all nodes, including login or compute nodes, so you don't have to do it more than once.

1.  generate and add SSH key to your agent
First, follow the procedure in the following link to generate and add an SSH key to your GitHub account.
https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
\
**Important:** The name "id_ed25519_github" is what you have named your key in this step. I suggest you add this _github to that filename to know which key you are using for github.\
Also regarding the "Adding your SSH key to the ssh-agent" section, you can use the following template in the SSH config file in the ~/.ssh folder (create a file without an extension called "config" if there isn’t any):

	a. on the local computer:
    ```
    Host github.com
    	HostName github.com
    	PreferredAuthentications publickey
    	IdentityFile ~/.ssh/id_ed25519_github
    Host *
    	AddKeysToAgent yes
    	UseKeychain yes
    ```

	b. On the cluster:
    ```
    Host github.com
    	Hostname ssh.github.com
    	IdentityFile ~/.ssh/id_ed25519_github
    	IdentitiesOnly yes
    	port 443
    ```
2. Adding a new SSH key to your GitHub account
   Follow the procedure in the following link to add the SSH key you just created to your GitHub account:
   https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account
3. Navigate to a folder where you want to clone the repository. After entering the following command in the terminal a MLEE folder will be created within the current path and all the files will be downloaded within the MLEE folder:
```
git clone git@github.com:HelmholtzAI-Consultants-Munich/MLEE.git
```

Alternatively, you can add a personal token and use HTTPS.
https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens


### Install Conda

#### Windows and macOS

Install the latest version of **Miniconda** (recommended) or **Anaconda** for your operating system and processor architecture.

- Miniconda: https://www.anaconda.com/download/success
- Anaconda: https://www.anaconda.com/download

#### Linux and cluster

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

### Recommended development environment (optional)

If you do not already have a preferred code editor or IDE, we recommend using **Visual Studio Code (VS Code)** for local development.

Download VS Code from:

https://code.visualstudio.com/

## Data and privacy
Since our NAKO data requires strict privacy, you have to either keep it as an encrypted volume file or archive file. To be able to use it on any OS, here we suggest an encrypted 7z file called data.7z with your self-chosen password for protection. On any operating system (OS), you can achieve this by encrypting the "data" folder after you transfer and store the data file, e.g. "N2Nworkingdataset20230724update.csv" in there. 

### How to create an archive file from a folder
**Important:** It is only required once after you transferred the data to the data folder.

#### Installing 7zip on Linux and macOS console/terminal:
The cluster already has the 7z installed and you don't need to install this software, i.e. you can skip this section. Otherwise, if you need to install 7z on your local PC then follow the procedure below:

1. First you have to download .tar.xz version with the correct Architecture, e.g. 64-bit Linux x86-64 or macOS (arm64 / x86-64) from the link: https://www.7-zip.org/download.html
2. Extract the downloaded file and this will create a folder that has 7z or 7zz binary. Navigate to the created folder.
3. In case the binary file is called 7zz then rename it to 7z for ease of use otherwise in the next sections you have to change 7z with 7zz in the commands.
4. Copy the 7z binary to your binary path with the following command:
    ```
    sudo cp 7z /usr/local/bin/
    ```
5. To check whether it is already recognized type
    ```
    which 7z /usr/local/bin/
    ```
    the output should be /usr/local/bin/7z or otherwise restart the terminal.
   
#### Create 7z file on Linux (including the cluster's OS) or macOS terminal after installing 7z
After installing 7zip if it is unavailable on your system, navigate to the MLEE folder where you cloned the repository. Then create the data.7z archive file in the MLEE folder as follows:
```
7z a -p data.7z data/
```
-p option makes sure to encrypt the file. It will ask you to give a password for the file.

### How to extract the previously made archive file from a folder
**Important:** Repeat this step each time you run the pipeline.

#### Extracting 7z file on Linux (including the cluster's OS) or macOS terminal
To extract the archive, run:
```
7z x data.7z
```
The last command creates a folder called "data" and extracts the content into the created folder.

#### Create or extract archive file on Windows after installing 7z
Since 7z on Windows has a GUI you can use that to create or extract the archive file. However, there is also a similar process for creating 7z file using the command line interface (CLI).

#### **Important note regarding the privacy**
Please delete the csv file after usage. You can extract the encrypted file (data.7z) stored in the MLEE folder any time you need it. 
You can either do this by deleting the "data" folder if you are in MLEE folder:
```
rm -rf data/
```
or delete the csv if you are in data folder:
```
rm -rf *.csv
```

## Environment setup

### Verify the Conda installation

Check your Conda version:

```bash
conda --version
```

Conda **23.10 or newer** is recommended. Older versions may use the slower `classic` dependency solver.

Check the configured solver:

```bash
conda config --show solver
```

The recommended output is:

```text
solver: libmamba
```

If the solver is `classic`, switch to `libmamba`:

```bash
conda config --set solver libmamba
```

Verify the change:

```bash
conda config --show solver
```

If `libmamba` is unavailable, update Conda before continuing:

```bash
conda update -n base conda
```

### Create the environment

Make sure you are in the root directory of the cloned `MLEE` repository.

**Local machine**

```bash
conda env create --file environment.yaml
```

**Cluster**

```bash
conda env create --file environment_cluster.yaml
```

### Activate the environment

```bash
conda activate MLEE
```

If you chose a different environment name, replace `MLEE` with the name you specified.

### Update the environment

If the environment definition has changed, update your existing environment.

**Local machine**

```bash
conda env update --file environment.yaml --prune
```

**Cluster**

```bash
conda env update --file environment_cluster.yaml --prune
```

### Recreate the environment

If updating does not resolve dependency conflicts, remove and recreate the environment.

Delete the environment:

```bash
conda remove --name MLEE --all
```

Then recreate it by following the **Create the environment** section above.

> **Note:** Creating the environment may take several minutes because Conda must resolve and download package dependencies. If it remains on **"Solving environment"** for an unusually long time, verify that you are using Conda 23.10 or newer with the `libmamba` solver enabled.


## Usage
This section contains the instructions to run the framework.

### Local machine
After the MLEE environment is activated you can just type the following command to run the pipeline:
```
python main.py
```
This will execute the whole pipeline with the input parameter you can set in the input_parameters.json.

### Cluster
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


## Outputs
The result of each run will be saved as HTML report within the reports folder. Each experiment is saved in a different folder, where the name contains the timestamp of the run and a unique identifier. Each report will  be also saved as an archived zip file for ease of transfer. Also, the outputs are saved in the output folder as well as the resources in the reports folders under the experiment's folder. 


## Input Parameters
- ```path_name```: Path to the directory containing the dataset, default: "./data/". <br />
- ```file_name```: Name of the dataset file <br />
- ```columns_to_keep```: Dictionary containing the list of numerical ("num") and categorical ("cat") columns to retain.  <br />
- ```binary_columns```: List with categorical features that are binary and do not need to be one hot encoded. <br />
- ```target```: Name of the target variable, default: "hypertens". <br />
- ```filters```: Dictionary specifying filters for age, BMI, and sex.
	- Keys correspond to the feature names.
	- If not specified, no filtering is applied.
	- For age and BMI, null is interpreted as 0 (lower bound) or ∞ (upper bound).
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

## Contributing

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

