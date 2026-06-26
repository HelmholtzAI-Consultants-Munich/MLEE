# MLEE
This repo contains a Machine Learning Framwork for Environmental Epidiemology (MLEE)

## Description
The environment has major impacts on human health, and particularly adverse effects are projected to increase within urban environments. Therefore, sophisticated geostatistical and data science models are urgently needed to better reflect real-life exposures and understand the long-term impact of environmental factors on human health. By joining the complementary expertise and data of HMGU and DLR, these impacts are targeted exemplary in the domain of noise.\
First, spatial limitations of state-of-the-art noise maps are tackled using a unique noise mapping approach based on generous data augmentation and deep convolutional networks. Then, this data will be linked to socio-economic and demographic information from more than 200.000 participants of the German national cohort (NAKO) to identify vulnerable clusters in terms of noise and neighborhood factors for the risk of hypertension by exploring distribution regression networks. Then, these clusters will be predicted for the whole of Germany.\
Finally, this network will be enhanced by auxiliary individual socioeconomic and health data to investigate the interplay of noise levels, neighborhood characteristics, and individual risk factors for hypertension.
We therefore apply and advance machine learning techniques considering interpretable approaches. The innovative AI/ML methods developed within this project shall serve as case studies for the modeling of additional health endpoints as well as additional environmental parameters like air pollution, temperature/heat waves, and relative humidity as these data are currently only available with strong limitations.\


## Cloning the repository (Only required once)
If you want to clone our repo to any device you have to follow a similar procedure but only once per device. Notice that on the cluster the file system is shared with all nodes, including login or compute nodes, so you don't have to do it more than once.

1.  generate and add ssh key to your agent
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
   Follow the procedure in the following link to add the ssk key you just created to your GitHub account:
   https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account
3. Now, to clone the repo you have to first navigate to a folder that you want. After entering the following command in the terminal a MLEE folder will be created within the current path and all the files will be downloaded within the MLEE folder:
```
git clone git@github.com:HelmholtzAI-Consultants-Munich/MLEE.git
```

Alternatively, you can add a personal token and use HTTPS.
https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens

## DATA and its Privacy
Since our NAKO data requires strict privacy, you have to either keep it as an encrypted volume file or archive file. To be able to use it on any OS, here we suggest an encrypted 7z file called data.7z with your self-chosen password for protection. On any operating system (OS), you can achieve this by encrypting the "data" folder after you transfer and store the data file, e.g. "N2Nworkingdataset20230724update.csv" in there. 

### How to create an archive file from a folder ()
**Important:** It is only required once after you transferred the data to the data folder.

#### installing 7zip on Linux and macOS console/terminal:
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
**Important:** This needs to be done each time you want to run the pipeline.

#### Extracting 7z file on Linux (including the cluster's OS) or macOS terminal
To extract you have to enter the following command:
follows:
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
## installing conda
#### installation on Windows or macOS
Navigate to https://www.anaconda.com/download and download the correct version of anaconda software based on your processor's architecture, specially for the case of intel or M processors.

#### installation on the cluster or Linux 
1) Start with creating “tools” folder under your home directory: This will help to organize your home directory and store all the necessary files under one directory. If you already have tools directory, you do not need to create one, otherwise you may use mkdir command as follows (Be sure that you are sitting on your personal home folder (/home/<usergroup>/<username>)
```
mkdir tools
```
and then, get into the tools directory with the following command ;
```
 cd ~/tools
```
2) Create necessary folders under tools directory:

Create tmpconda folder under tools folder
```
mkdir tmpconda
```
3) Navigate to tools/tmpconda folder and
```
cd ~/tools/tmpconda
```
4) download miniconda with the following command : 
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
5) now make sure conda knows the temporary directory and install with bash:
```
TMPDIR=~/tools/tmpconda bash Miniconda3-latest-Linux-x86_64.sh -u
```
Please read license agreement and accept it, HIT enter key, then exit.

6) Give Required Permissions
```
chmod +x ~/miniconda3/bin/*
chmod +x ~/miniconda3/envs
export LD_LIBRARY_PATH=~/miniconda3/lib:$LD_LIBRARY_PATH
export PATH=~/miniconda3/bin:$PATH
PATH=$PATH:$HOME/miniconda3/bin
source ~/miniconda3/etc/profile.d/conda.sh
~/miniconda3/bin/conda init bash
```
---> After installation, log out and log in again for the change to make effect.

 5) Verify the Installation and setup
``` 
export LD_LIBRARY_PATH=~/miniconda3/lib:$LD_LIBRARY_PATH
export PATH=~/miniconda3/bin:$PATH
PATH=$PATH:$HOME/miniconda3/bin
conda -V
```

#### Creating environment:
6) After cloning this repository, navigate to MLEE folder, and create the conda environment ```MLEE``` by default from the environment.yaml file as follows (You can change the name of the conda environment by changing the name in this yaml file):
- Local machine:
```
conda env create --file environment.yaml
```
- Cluster:
```
conda env create --file environment_cluster.yaml
```

#### Activating environment:
then activate the conda environment
```
conda activate MLEE
```

#### Updating environment:
For each time there has been some new development in the code and if there is an error that tells a package is missing then please run the following to update your environment( notice here that the environment.yaml has a name argument inside that should be the exact same name as when you created the environment if you have changed before from default to something else):
- Local machine:
```
conda env update --file environment.yaml --prune
```
- Cluster:
```
conda env update --file environment_cluster.yaml --prune
```

#### Deleting and reinstalling environment:
sometimes updating the environment doesn't resolve conflict issues, then remove the environment and install it again:
Delete:
```
conda remove --name MLEE --all
```
Reinstall: Please follow the same procedure as creating the environment.

## Usage
This section contains the instructions to run the pipeline.\

### Run in your laptop
After the MLEE environment is activated you can just type the following command to run the pipeline:
```
python main.py
```
This will execute the whole pipeline with the input parameter you can set in the input_parameters.json.

### Run on the cluster
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
 	replacing user.name with your username. Also the normal terminal output logs are now stored in slurm_log folder with output as output_job_allocated_number.job and errors in error_job_allocated_number.job respectively.

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

## Revision:
### Installation details on windows machines (configuring ssh and cloning) TODO
1. First we created ssh key as described https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
2. New-Item for creating this config file
3. notepad.exe for config


## Input Parameters Description
- ```path_name```: Path to the directory containing the dataset, default: "./data/". <br />
- ```file_name```: Name of the dataset file., default: "N2Nworkingdataset20230724update.csv". <br />
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
