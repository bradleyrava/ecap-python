from distutils.core import setup
setup(
  name = 'ecap',         
  packages = ['ecap'],   
  version = '0.2',
  license='GPLv3',       
  description = 'ECAP Procedure in Python',
  long_description = 'Implements the Excess Certainty Adjusted Probability adjustment procedure as described in the paper "Irrational Exuberance: Correcting Bias in Probability Estimates" by Gareth James, Peter Radchenko, and Bradley Rava (Journal of the American Statistical Association, 2020; <doi:10.1080/01621459.2020.1787175>). The package includes a function that preforms the ECAP adjustment and a function that estimates the parameters needed for implementing ECAP.',
  author = 'Bradley Rava',                  
  author_email = 'brava@marshall.usc.edu',     
  url = 'https://github.com/bradleyrava',   
  download_url = 'https://github.com/bradleyrava/ecap-python/archive/ecap-0.2.tar.gz',
  keywords = ['Excess Certainty', 'Emperical Bayes', 'probability', 'calibration', 'statistics'],  
  install_requires=[            
          'pandas',
          'numpy',
          'sklearn',
          'quadprog',
          'statsmodels',
          'patsy'
          ],
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',  
    'Topic :: Software Development :: Build Tools',
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",   
    'Programming Language :: Python :: 3.7',
  ],
)

