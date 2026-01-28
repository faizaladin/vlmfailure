# VLM Safety Analysis

Faiz Aladin, Kaustav Chakraborty

University of Southern California

The purpose of this repository is to finetune LlaVa-Video on the initial trajectories of cars under various weather conditions and determine whether the car will successfully stay on the road or fail by committing a lane violation or crashing. Additionally we use a fully automated approach by assigning binary labels while collecting data to avoid human labeled data. This way the VLM uses a scalable end-to-end training approach. 

To collect data for the VLM finetuning, use this repository: https://github.com/faizaladin/Neural-Net-Image-Driving

## How to Use

1. Install the Conda Environment
<pre>conda env create -f environment.yml
conda activate myenv
</pre>

2. After collecting the data from the repository above, you will need to format it.
<pre>python format_data.py</pre>

3. Fine-Tune LlaVa-Video
<pre>python train.py</pre>

4. Evaluate on unseen trajectories
<pre>python inference.py </pre>

## Results

Prompt: This is a video sequence from a car's vision controller. This sequence is the trajectory of the car. Predict: **Success** (stays on road) or **Failure** (off-road or collision)

![Town 2 Sunny](assets/town5_run_134.gif)

ASSISTANT: The car is driving on a road, and there are no obstacles or other vehicles in its path. The weather appears to be clear, and the car is moving at a steady speed. Based on these factors, the car is likely to successfully navigate the road and reach its destination.

![Town 2 Sunny](assets/town3_run_131.gif)

ASSISTANT: The car is driving on a wet road, which can make the road slippery and increase the risk of accidents. The car is also driving at a high speed, which can make it more difficult to control and react to any obstacles or changes in the road conditions. Additionally, the car is driving in a residential area, which may have parked cars, pedestrians, or other obstacles that the car must navigate around. Based on these factors, it is likely that the car will experience a failure, as it may collide with an obstacle or lose control due to the wet road conditions and high speed.

    
