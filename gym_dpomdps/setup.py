from setuptools import setup, find_packages


setup(
    name='gym_dpomdps',
    version='0.1.0',
    packages=find_packages(),
    package_data={'': ['*.dpomdp']},
    install_requires=['numpy', 'gym'],
    test_suite='tests',
)