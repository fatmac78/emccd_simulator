import numpy as np
from pyspark import SparkContext

class ElectronMultiplier:

    def __init__(self, stages, gain, prob_cic, random_seed, 
                 readout_mu, readout_sigma, spark_context,  **kwargs):
        self.stages: int = stages
        self.prob_multiplication: float = gain**(1/stages)-1
        self.prob_cic: float = prob_cic
        self.readout_mu: float = readout_mu
        self.readout_sigma: float = readout_sigma
        self.sc = spark_context
        np.random.seed(random_seed)

    def register_amplification_shift(self,
                                     register_input_electron_count: int)\
                                     -> int:
        """Simulate cascade process in single electron shift

        Returns:
            Integer photon count at end of register
        """
        register_output_electron_count = register_input_electron_count

        for electron in range(0, register_input_electron_count):
            register_output_electron_count += \
                np.random.binomial(1, self.prob_multiplication)
        return register_output_electron_count

    def register_cic_shift(self) -> int:
        """Simulate cascade process in single electron shift
       
        Returns:
            Integer photon count at end of register
        """
        register_output_cic_electron_count = 0
        register_output_cic_electron_count = \
             np.random.binomial(1, self.prob_cic)
        return register_output_cic_electron_count

    def pixel_shift_out(self, input_electron_count: int) -> int: 
        """Shift an input pixel electron count through the EM register

        Returns:
            Integer photon count at end of register
        """    
        output_electron_count = input_electron_count

        for stage in range(1, self.stages):
            output_electron_count = \
                self.register_amplification_shift(output_electron_count)
            output_electron_count += self.register_cic_shift()
        return output_electron_count
   
    def image_shift_out(self, input_electron_map: np.ndarray) -> np.ndarray:    
        """Shift an entire image through the EM register

        Returns:
            Integer photon count at end of register
        """
        shift_pixel_out_vectorized = np.vectorize(self.pixel_shift_out)
        return shift_pixel_out_vectorized(input_electron_map)

    def simulate_stack(self, image_stack: np.ndarray) -> np.ndarray:
        """Shift a stack of input photon frames through the EM register,
         adding readout noise

        Returns:
            Output image stack
        """
        def simulate_image(image: np.ndarray) -> np.ndarray:
            output_image = self.image_shift_out(image) + \
                             (self.readout_sigma*np.random.randn(image.shape[1], \
                                 image.shape[1])) \
                              + self.readout_mu
            return output_image

        out_image_stack = []

        for frame in range(0, image_stack.shape[0]):
            output_image = simulate_image(image_stack[frame, :, :])
            out_image_stack.append(output_image)
        
        out_image_stack = np.stack(out_image_stack, axis=0)  
        
        return out_image_stack
    
    def simulate_stack_spark(self, image_stack) -> np.ndarray:
        """Shift a stack of input photon frames through the EM register,
         adding readout noise. Utilise spark framework for parallel processing
        Returns:
            Output image stack
        """
        
        def register_amplification_shift(register_input_electron_count: int,
                                         prob_multiplication: float)\
                                     -> int:
            """Simulate cascade process in single electron shift

        Returns:
            Integer photon count at end of register
        """

            register_output_electron_count = register_input_electron_count

            for electron in range(0, register_input_electron_count):
                register_output_electron_count += \
                    np.random.binomial(1, prob_multiplication)
            return register_output_electron_count

        def register_cic_shift(prob_cic) -> int:
            """Simulate cascade process in single electron shift
       
        Returns:
            Integer photon count at end of register
        """
            register_output_cic_electron_count = 0

            register_output_cic_electron_count = \
                 np.random.binomial(1, prob_cic)
            return register_output_cic_electron_count

        def pixel_shift_out(input_electron_count: int,  stages: int, \
            prob_mult: float, prob_cic: float) -> int: 
            """Shift an input pixel electron count through the EM register

        Returns:
            Integer photon count at end of register
        """    
            output_electron_count = input_electron_count

            for stage in range(1, stages):
                output_electron_count = \
                    register_amplification_shift(output_electron_count, prob_mult)
                output_electron_count += register_cic_shift(prob_cic)
            return output_electron_count
       
        def simulate_pixel(pixel: int, readout_sigma: float, 
                           readout_mu: float, stages: int, 
                           prob_mult: float, prob_cic: float) -> int:
            """Shift an input pixel electron count through the EM register and
               add readout bias and noise

            Returns:
               Integer photon count at output of EMCCD
            """   
            output_pixel = pixel_shift_out(pixel, stages, prob_mult, prob_cic)\
                 + (readout_sigma*np.random.randn(1)) \
                + readout_mu
            return output_pixel

        input_pixels = self.sc.parallelize(image_stack.flatten(), 200)

        mu = self.readout_mu
        sigma = self.readout_sigma
        stages = self.stages
        prob_mult = self.prob_multiplication
        prob_cic = self.prob_cic

        output_pixels = input_pixels.map(lambda x: simulate_pixel(x, sigma, mu, 
                                        stages, prob_mult, prob_cic))

        output_pixels_numpy = np.array(output_pixels.collect())
             
        return output_pixels_numpy
