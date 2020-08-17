from distutils.core import setup
setup(
  name = 'ecap',         
  packages = ['ecap'],   
  version = '0.1.95',     
  license='GPLv3',       
  description = 'Implements the Excess Certainty Adjusted Probability adjustment procedure.',
  author = 'Bradley Rava',                  
  author_email = 'brava@marshall.usc.edu',     
  url = 'https://github.com/bradleyrava',   
  download_url = 'https://github.com/bradleyrava/ecap-python/archive/0.1.95.tar.gz',   
  keywords = ['Excess Certainty', 'Emperical Bayes', 'probability', 'calibration', 'statistics'],  
  install_requires=[            
          'pandas',
          'numpy',
          'sklearn',
          'quadprog',
          'statsmodels',
          'patsy',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',  
    'Topic :: Software Development :: Build Tools',
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",   
    'Programming Language :: Python :: 3.7',
  ],
)

