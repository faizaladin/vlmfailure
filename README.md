# VLM Safety Analysis

Faiz Aladin, Kaustav Chakraborty

University of Southern California

The purpose of this repository is to finetune LlaVa 1.5 on trajectories of cars under various weather conditions and determine whether the car will successfully stay on the road or fail by committing a lane violation or crashing. Additionally we use a fully automated approach by assigning binary labels while collecting data to avoid human labeled data. This way the VLM uses a scalable end-to-end training approach. 

To collect data for the VLM finetuning the repository here can be used: https://github.com/faizaladin/Neural-Net-Image-Driving

## How to Use

1. Install the Conda Environment
<pre>conda env create -f environment.yml
conda activate myenv
</pre>

2. Fine-Tune LlaVa 1.5
