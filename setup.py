from distutils.core import setup
setup(
  name = 'ecap',         # How you named your package folder (MyLib)
  packages = ['ecap'],   # Chose the same as "name"
  version = '0.1.1',      # Start with a small number and increase it with every change you make
  license='GPL-3',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Implements the Excess Certainty Adjusted Probability adjustment procedure as described in the paper "Irrational Exuberance: Correcting Bias in Probability Estimates" by Gareth James, Peter Radchenko, and Bradley Rava (Journal of the American Statistical Association, 2020; <doi:10.1080/01621459.2020.1787175>). The package includes a function that preforms the ECAP adjustment and a function that estimates the parameters needed for implementing ECAP. For testing and reproducibility, the ESPN and FiveThirtyEight data used in the paper are also included.',   # Give a short description about your library
  author = 'Bradley Rava',                   # Type in your name
  author_email = 'brava@marshall.usc.edu',      # Type in your E-Mail
  url = 'https://github.com/bradleyrava',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/bradleyrava/ecap-python/archive/0.1.tar.gz',    # I explain this later on
  keywords = ['Excess Certainty', 'Emperical Bayes', 'Probability', 'Calibration'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'pandas',
          'numpy',
          'sklearn',
          'quadprog',
          'abc',
          'statsmodels',
          'patsy',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: GPL-3 License',   # Again, pick a license
    'Programming Language :: Python :: 3.7.3',
  ],
)