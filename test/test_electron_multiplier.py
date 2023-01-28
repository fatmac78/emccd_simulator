from emccd_simulator.electron_multiplier import ElectronMultiplier
import numpy as np
from pyspark import SparkContext, SparkConf

em_stages = 10
em_gain = 1000
prob_cic = 0.0
random_seed = 1234
readout_mu = 0
readout_sigma = 0

conf = (SparkConf()
    )
    
sc = SparkContext(conf=conf)

def test_register_amplification_shift_no_input():
    emccd=ElectronMultiplier(em_stages, em_gain, prob_cic, random_seed, 
                             readout_mu, readout_sigma, sc)
    output_electrons=emccd.register_amplification_shift(0)
    assert output_electrons == 0

def test_register_amplification_shift_prob_mult_1():
    # Guarantee multiplication at each stage
    em_stages = 10
    em_gain = 1000
    emccd=ElectronMultiplier(em_stages, em_gain, prob_cic, random_seed, 
                             readout_mu, readout_sigma, sc)
    output_electrons=emccd.register_amplification_shift(1)
    assert output_electrons == 2

def test_cic_shift_no_prob():
    emccd=ElectronMultiplier(em_stages, em_gain, prob_cic, random_seed, 
                             readout_mu, readout_sigma, sc)
    generated_cic = emccd.register_cic_shift()
    assert generated_cic == 0
    prob_cic_certain=1.0

def test_cic_shift_certain():
    prob_cic_certain=1.0
    emccd=ElectronMultiplier(em_stages, em_gain, prob_cic_certain, random_seed, 
                             readout_mu, readout_sigma, sc)
    generated_cic = emccd.register_cic_shift()
    assert generated_cic == 1

def test_pixel_shift_out_no_input_no_cic():
    emccd=ElectronMultiplier(em_stages, em_gain, prob_cic, random_seed, 
                             readout_mu, readout_sigma, sc)
    output_electrons=emccd.pixel_shift_out(0)
    assert output_electrons == 0

def test_pixel_shift_out_1_input_no_cic():
    em_stages = 100
    em_gain = 20
    simulations = 5000
    emccd=ElectronMultiplier(em_stages, em_gain, prob_cic, random_seed, 
                             readout_mu, readout_sigma, sc)
    output_electrons = 0
    for i in range(0,simulations):
        output_electrons = output_electrons + emccd.pixel_shift_out(1)
    average_output_electrons = output_electrons/simulations
    print(average_output_electrons)
    assert average_output_electrons > em_gain-1 and average_output_electrons < em_gain+1   

def test_pixel_shift_out_1_input_no_cic():
    em_stages = 100
    em_gain = 10
    simulations = 5000
    emccd=ElectronMultiplier(em_stages, em_gain, prob_cic, random_seed, 
                             readout_mu, readout_sigma, sc)
    output_electrons = 0
    for i in range(0, simulations):
        output_electrons = output_electrons + emccd.pixel_shift_out(1)
    average_output_electrons = output_electrons/simulations

    assert average_output_electrons > em_gain-1 and average_output_electrons < em_gain+1   

def test_stack_shift_spark_parallel_1photon_input_no_cic():
    em_stages = 100
    em_gain = 10

    emccd=ElectronMultiplier(em_stages, em_gain, prob_cic, random_seed, 
                             readout_mu, readout_sigma, sc)
    
    # General test input image stack of 10 100x100 frames, 1 photon in each (100k pixels)
    input_image_stack=np.full((10, 100, 100), 1)   

    output_image_stack = emccd.simulate_stack_spark(input_image_stack)
    
    average_output_electrons=np.mean(output_image_stack)
    
    assert average_output_electrons > em_gain-1 and average_output_electrons < em_gain+1   
