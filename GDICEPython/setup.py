from setuptools import setup, find_packages


setup(
    name='GDICE_Python',
    version='0.1.2',
    packages=find_packages(),
    install_requires=['numpy', 'gym', 'rl_parsers', 'gym_pomdps', 'gym_dpomdps', 'filelock'],
    scripts=['testUAV.py', 'generalGDICE.py', 'cleanTempResults.py', 'clearFinalResults.py', 'PlottingScript.py']
)