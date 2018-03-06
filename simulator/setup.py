from setuptools import setup, find_packages

setup(
    name='simulator',
    version='0.1',
    description='End-to-end Resistive RAM-based Simulator',
    author='THU.IME & THU.DCST & ASU',
    license='BSD',
    install_requires=['numpy', 'theano'],
    include_package_data=True,
    packages = find_packages(),
    )
