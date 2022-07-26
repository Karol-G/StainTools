import numpy as np
import staintools


data = np.random.randint(low=0, high=255, size=(512, 512, 3), dtype=np.uint8)

standardizer = staintools.LuminosityStandardizer
augmentor = staintools.StainAugmentor(method='vahadane', sigma1=0.2, sigma2=0.2)

transformed_array = standardizer.standardize(data)
augmentor.fit(transformed_array)
transformed_array = augmentor.pop()