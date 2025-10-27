"""
Test script to print collision object map and count from llava_input.json
"""
from vlm_train import LlavaSequenceDataset

dataset = LlavaSequenceDataset("llava_input.json")
# The print statements are already in the __init__, so running this will show the output.
